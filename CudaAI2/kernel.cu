
#include "cuda_runtime.h"
#include "device_launch_parameters.h"




#include <stdio.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <random>
#include <cublas_v2.h>

//eval_grad definition : 
// Suppose current node is f(g(x1,x2,...)), where f is parent, g is current node and x1,x2,... are children
// parent_grad is the gradient of the parent node evaluated at g(x1,x2,...)
// eval_grad should compute the partial differential of f(g) with respect to x_i, where x_i is an input of the current node, and pass it to the child node i

static constexpr bool PRINT_LOSS = true;
cublasHandle_t cublas_handle;


constexpr size_t next_power_of_two(size_t v)
{
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v |= v >> 32;
    return v + 1;
}



__global__ void elementwise_tanh(const float* input, float* result, uint32_t n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n)
		result[i] = tanhf(input[i]);
}
//f' = 1 - f^2
__global__ void tanh_grad(const float* parent_grad, const float* tanh_x, float* result, uint32_t n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n)
        result[i] += parent_grad[i] * (1.0f - tanh_x[i] * tanh_x[i]);   
}


//tanh(x)
template<typename OP>
struct matrix_tanh
{
    static constexpr int rows = OP::rows;
    static constexpr int cols = OP::cols;

    matrix_tanh(OP& o) : op(o)
    {
        cudaMalloc(&gpu_data, OP::rows * OP::cols * sizeof(float));
        cudaMalloc(&grad_data, OP::rows * OP::cols * sizeof(float));
        op.num_parents++;
    }
    ~matrix_tanh()
    {
        cudaFree(&gpu_data);
        cudaFree(&grad_data);
    }

    void eval()
    {
        op.eval();
        size_t size = OP::rows * OP::cols;
        constexpr int threads_per_block = 64;
        int num_blocks = 1 > size / threads_per_block ? 1 : size / threads_per_block;
        elementwise_tanh << < dim3(num_blocks, 1, 1), dim3(threads_per_block, 1, 1) >> > (op.gpu_data, gpu_data, size);

    }

    void zero_grad()
    {
		proccessed_parents = 0;
		cudaMemset(grad_data, 0, OP::rows * OP::cols * sizeof(float));
		op.zero_grad();
    }

    void eval_grad(float* parent_grad)
    {
        size_t size = OP::rows * OP::cols;
        constexpr int threads_per_block = 64;
        int num_blocks = 1 > size / threads_per_block ? 1 : size / threads_per_block;
        tanh_grad << < dim3(num_blocks, 1, 1), dim3(threads_per_block, 1, 1) >> > (parent_grad, gpu_data, grad_data, size);
        if (++proccessed_parents == num_parents)
            op.eval_grad(grad_data);
    }

    size_t num_parents = 0;
    size_t proccessed_parents = 0;
    OP& op;
    float* gpu_data;
    float* grad_data;
};

//1/2 * ||op||^2
template<typename OP>
struct matrix_l2
{
    static constexpr int rows = 1;
    static constexpr int cols = 1;

    matrix_l2(OP& o) : op(o)
    {
        cudaMalloc(&gpu_data, sizeof(float));
        cudaMalloc(&grad_data, OP::rows * OP::cols * sizeof(float));
        op.num_parents++;
    }
    ~matrix_l2()
    {
        cudaFree(&gpu_data);
        cudaFree(&grad_data);
    }

    void eval()
    {
        op.eval();

        cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_DEVICE);
        cublasSnrm2(cublas_handle, OP::rows * OP::cols, op.gpu_data, 1, gpu_data);
        cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_HOST);

		float alpha = 0.5f;
        cublasSscal(cublas_handle, 1, &alpha, gpu_data, 1);

        if constexpr (PRINT_LOSS)
        {
            float x;
            cudaMemcpy(&x, gpu_data, sizeof(float), cudaMemcpyDeviceToHost);
            std::cout << "Error : " << x << '\n';
        }
    }

    void zero_grad()
    {
        proccessed_parents = 0;
        cudaMemset(grad_data, 0, OP::rows * OP::cols * sizeof(float));
        op.zero_grad();
    }

    void eval_grad()
    {
        op.eval_grad(op.gpu_data);
    }

    void eval_grad(float* parent_grad)
	{
        cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_DEVICE);
		cublasSaxpy(cublas_handle, OP::rows * OP::cols, parent_grad, op.gpu_data, 1, grad_data, 1);
		cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_HOST);

		if (++proccessed_parents == num_parents)
			op.eval_grad(grad_data);
	}

    size_t num_parents = 0;
    size_t proccessed_parents = 0;
    OP& op;
    float* gpu_data;
    float* grad_data;
};

//Matrix product
template<typename OP1, typename OP2>
requires (OP1::cols == OP2::rows)
struct matrix_product
{
public:
    static constexpr size_t rows = OP1::rows;
    static constexpr size_t cols = OP2::cols;
    matrix_product(OP1& o1, OP2& o2) : op1(o1), op2(o2)
    {
        cudaMalloc(&gpu_data, rows * cols * sizeof(float));
        cudaMalloc(&op1_grad, OP1::rows * OP1::cols * sizeof(float));
        cudaMalloc(&op2_grad, OP2::rows * OP2::cols * sizeof(float));
        op1.num_parents++;
        op2.num_parents++;
    }
    ~matrix_product()
    {
        cudaFree(&gpu_data);
        cudaFree(&op1_grad);
        cudaFree(&op2_grad);
    }

    void eval()
    {
        op1.eval();
        op2.eval();

        float alpha = 1.0f;
        float beta = 0.0f;
        cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, cols, rows, OP1::cols, &alpha, op2.gpu_data, cols, op1.gpu_data, OP1::cols, &beta, gpu_data, cols);
    }

    void zero_grad()
    {
        proccessed_parents = 0;
        cudaMemset(op1_grad, 0, OP1::rows * OP1::cols * sizeof(float));
        cudaMemset(op2_grad, 0, OP2::rows * OP2::cols * sizeof(float));
        op1.zero_grad();
        op2.zero_grad();
    }

    void eval_grad(float* parent_grad)
    {
        int n = op1.rows;
        int k = op1.cols;
        int m = op2.cols;

        float alpha = 1.0f;
		float beta = 1.0f;

        cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, k, n, m, &alpha, op2.gpu_data, m, parent_grad, m, &beta, op1_grad, k);
        cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, m, k, n, &alpha, parent_grad, m, op1.gpu_data, k, &beta, op2_grad, m);

        if (++proccessed_parents == num_parents)
		{
			op1.eval_grad(op1_grad);
			op2.eval_grad(op2_grad);
		}
    }

    size_t num_parents = 0;
    size_t proccessed_parents = 0;
    OP1& op1;
    OP2& op2;
    float* gpu_data;
    float* op1_grad;
    float* op2_grad;
};

//Sums a matrix and a column, the column is added to all columns of the matrix. 
template<typename OP1, typename OP2>
requires (OP1::rows == OP2::rows && OP2::cols == 1)
struct matrix_sum_all_cols
{
public:
    static constexpr size_t rows = OP1::rows;
    static constexpr size_t cols = OP1::cols;

    matrix_sum_all_cols(OP1& o1, OP2& o2) : op1(o1), op2(o2)
    {
        int max_size = std::max(rows, cols);
        cudaMalloc(&gpu_data, rows * cols * sizeof(float));
        cudaMalloc(&op1_grad, OP1::rows * OP1::cols * sizeof(float));
        cudaMalloc(&op2_grad, OP2::rows * OP2::cols * sizeof(float));
        cudaMalloc(&all_one_vector, max_size * sizeof(float));
        op1.num_parents++;
        op2.num_parents++;

        
        //used for computing gradients
        float* all_one_vector_cpu = new float[max_size];
        for (size_t i = 0; i < max_size; i++)
        {
            all_one_vector_cpu[i] = 1.0f;
        }
        cudaMemcpy(all_one_vector, all_one_vector_cpu, max_size * sizeof(float), cudaMemcpyHostToDevice);
        delete[] all_one_vector_cpu;
    };
    ~matrix_sum_all_cols()
    {
        cudaFree(&gpu_data);
        cudaFree(&op1_grad);
        cudaFree(&op2_grad);
        cudaFree(&all_one_vector);
    }
    void eval()
    {
        op1.eval();
        op2.eval();
        
        float alpha = 1.0f;
		float beta = 0.0f;
        cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, cols, rows, 1, &alpha, all_one_vector, cols, op2.gpu_data, 1, &beta, gpu_data, cols);
        beta = 1.0f;
        cublasSgeam(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, cols, rows, &alpha, gpu_data, cols, &beta, op1.gpu_data, cols, gpu_data, cols);
    }

    void zero_grad()
    {
        proccessed_parents = 0;
        cudaMemset(op1_grad, 0, OP1::rows * OP1::cols * sizeof(float));
        cudaMemset(op2_grad, 0, OP2::rows * OP2::cols * sizeof(float));
        op1.zero_grad();
        op2.zero_grad();
    }

    void eval_grad(float* parent_grad)
    {

        float alpha = 1.0f;
		float beta = 1.0f;
    	cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, OP1::rows, OP1::cols, &alpha, all_one_vector, 1, parent_grad, OP1::cols, &beta, op2_grad, 1);
        cublasSgeam(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, cols, rows, &alpha, parent_grad, cols, &beta, op1_grad, cols, op1_grad, cols);
        if (++proccessed_parents == num_parents)
        {
            op1.eval_grad(op1_grad);
            op2.eval_grad(op2_grad);
        }
    }

    size_t num_parents = 0;
    size_t proccessed_parents = 0;
    float* gpu_data;
    float* op1_grad;
    float* op2_grad;
    float* all_one_vector;
private:
    OP1& op1;
    OP2& op2;
   
};

//Regular sum
template<typename OP1, typename OP2>
requires (OP1::cols == OP2::cols && OP1::rows == OP2::rows)
struct matrix_sum
{
public:
    static constexpr size_t rows = OP1::rows;
    static constexpr size_t cols = OP2::cols;
    matrix_sum(OP1& o1, OP2& o2) : op1(o1), op2(o2)
    {
        cudaMalloc(&gpu_data, rows * cols * sizeof(float));
        cudaMalloc(&grad_data, rows * cols * sizeof(float));
        op1.num_parents++;
        op2.num_parents++;
    }
    ~matrix_sum()
    {
        cudaFree(&gpu_data);
        cudaFree(&grad_data);
    }

    void eval()
    {
        op1.eval();
        op2.eval();

        float alpha = 1.0f;
        float beta = 1.0f;
		cublasSgeam(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, cols, rows, &alpha, op1.gpu_data, cols, &beta, op2.gpu_data, cols, gpu_data, cols);
    }

    void zero_grad()
    {
        proccessed_parents = 0;
        cudaMemset(grad_data, 0, rows * cols * sizeof(float));
        op1.zero_grad();
        op2.zero_grad();
    }

    void eval_grad(float* parent_grad)
    {
        float alpha = 1.0f;
        float beta = 1.0f;
        cublasSgeam(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, cols, rows, &alpha, parent_grad, cols, &beta, grad_data, cols, grad_data, cols);
        if (++proccessed_parents == num_parents)
		{
			op1.eval_grad(grad_data);
			op2.eval_grad(grad_data);
		}
    }

    size_t num_parents = 0;
    size_t proccessed_parents = 0;
    OP1& op1;
    OP2& op2;
    float* gpu_data;
    float* grad_data;
};

//A matrix where data from the CPU can be loaded
template<size_t n, size_t m>
struct input_matrix
{
public:
	static constexpr int rows = n;
	static constexpr int cols = m;

    size_t num_vectors;

	input_matrix(size_t n_vec) : num_vectors(n_vec), data(std::vector<float>(rows * cols * num_vectors, 0))
	{
		cudaMalloc(&full_data, rows * cols * num_vectors * sizeof(float));
        gpu_data = full_data;
        current_data_idx = 0;
        dirty = true;
	}
	~input_matrix()
	{
		cudaFree(&full_data);
	}

	void eval()
	{
        if (dirty)
		    cudaMemcpy(full_data, data.data(), rows * cols * num_vectors * sizeof(float), cudaMemcpyHostToDevice);
        dirty = false;
	}

    void zero_grad()
    {

    }

	void eval_grad(float* parent_grad)
	{
		
	}

    void next_data()
    {
        current_data_idx = (current_data_idx + 1) % num_vectors;
		gpu_data = full_data + current_data_idx * rows * cols;
    }

    size_t num_parents = 0;
	float* gpu_data;
    float* full_data;
	std::vector<float> data;
    
private:
    int current_data_idx;
    bool dirty = true;
};

//A matrix that can be modified by gradient descent
template<size_t n, size_t m>
struct weight_matrix
{
public:
    static constexpr int rows = n;
    static constexpr int cols = m;

    weight_matrix() : data(std::vector<float>(rows* cols, 0))
    {
        cudaMalloc(&gpu_data, rows * cols * sizeof(float));
        cudaMalloc(&grad_data, rows * cols * sizeof(float));
        dirty = true;
    }
    ~weight_matrix()
    {
        cudaFree(&gpu_data);
    }

    void download()
    {
        cudaMemcpy(data.data(), gpu_data, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
    }

    void eval()
    {
        if (dirty)
            cudaMemcpy(gpu_data, data.data(), rows * cols * sizeof(float), cudaMemcpyHostToDevice);
        dirty = false;
    }

    void zero_grad()
    {
		cudaMemset(grad_data, 0, rows * cols * sizeof(float));
    }
    void eval_grad(float* parent_grad)
    {
        float alpha = 1.0f;
        float beta = 1.0f;
        cublasSgeam(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, cols, rows, &alpha, parent_grad, cols, &beta, grad_data, cols, grad_data, cols);
    }

    void gradient_descent(float alpha)
    {
        float minus_alpha = -alpha;
        cublasSaxpy(cublas_handle, rows * cols, &minus_alpha, grad_data, 1, gpu_data, 1);
    }

    size_t num_parents = 0;
    float* gpu_data;
    float* grad_data;
    std::vector<float> data;
    bool dirty;
};


void mlp_example()
{
    constexpr size_t layer_size = 8;
    constexpr size_t batch_size = 8;

    auto W1 = weight_matrix<layer_size, 2>();
    auto b1 = weight_matrix<layer_size, 1>();
    auto W2 = weight_matrix<1, layer_size>();
    auto b2 = weight_matrix<1, 1>();
    
    //moon dataset
    auto device = std::random_device();
    auto generator = std::mt19937(device());
    auto distribution = std::uniform_real_distribution<float>(-1.0f, 1.0f);

    auto x = input_matrix<2, 2>(256);
    auto v1 = std::vector<int>(128);
    auto v2 = std::vector<int>(128);
    std::iota(v1.begin(), v1.end(), 0);
    std::iota(v2.begin(), v2.end(), 0);
    std::shuffle(v1.begin(), v1.end(), generator);
    std::shuffle(v2.begin(), v2.end(), generator);

    for (size_t i = 0; i < 128; i++)
	{
		x.data[i * 4] = 0.5f * cosf(3.1415f * v1[i] / 256.0f);
		x.data[i * 4 + 2] = 0.5f * sinf(3.1415f * v1[i] / 256.0f);
        x.data[i * 4 + 1] = 0.5f * cosf(3.1415f * v2[i] / 256.0f) + 0.5f;
        x.data[i * 4 + 3] = -0.5f * sinf(3.1415f * v2[i] / 256.0f) + 0.3f;
	}
    auto y = input_matrix<1, 2>(256);
    for (size_t i = 0; i < 256; i++)
	{
		y.data[i] = (i % 2 == 0) ? -1.0f : 1.0f;
	}
    
    auto Wx = matrix_product(W1, x);
    auto Wx_plus_b = matrix_sum_all_cols(Wx, b1);
    auto tanh1 = matrix_tanh(Wx_plus_b);
    auto W2_tanh1 = matrix_product(W2, tanh1);
    auto W2_tanh1_plus_b = matrix_sum_all_cols(W2_tanh1, b2);
    auto final_res = matrix_tanh(W2_tanh1_plus_b);
    auto diff = matrix_sum(final_res, y);
    auto sqr = matrix_l2(diff);

    for (size_t i = 0; i < W1.cols * W1.rows; i++)
    {
		W1.data[i] = distribution(generator);
    }
    for (size_t i = 0; i < W2.cols * W2.rows; i++)
    {
        W2.data[i] = distribution(generator);
    }
    for (size_t i = 0; i < b1.cols * b1.rows; i++)
	{
		b1.data[i] = distribution(generator);
	}
    for (size_t i = 0; i < b2.cols * b2.rows; i++)
    {
        b2.data[i] = distribution(generator);
    }

    //training
    float lr = 0.06f;
    
    for (size_t i = 0; i < 20000; i++)
    {
        sqr.zero_grad();
        sqr.eval();
        sqr.eval_grad();
        
        W1.gradient_descent(lr);
        W2.gradient_descent(lr);
        b1.gradient_descent(lr);
        b2.gradient_descent(lr);

        y.next_data();
        x.next_data();
    }

}

void matrix_inversion_example()
{
    constexpr size_t matrix_size = 128;
    auto x = weight_matrix<matrix_size, matrix_size>();
    auto y = input_matrix<matrix_size, matrix_size>(1);
    auto minus_eye = input_matrix<matrix_size, matrix_size>(1);
    
    std::random_device device;
    std::mt19937 generator(device());
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);

    //fill matrices
    for (size_t i = 0;i < matrix_size * matrix_size; i++)
	{
		y.data[i] = distribution(generator);
	}

    for (size_t i = 0; i < matrix_size; i++)
    {
        y.data[i + ((i + 1) % matrix_size) * matrix_size] = 1.0f;
        x.data[i * (1 + matrix_size)] = 15.0f;
        minus_eye.data[i * (1 + matrix_size)] = -1.0f;
    }
    auto prod = matrix_product(x, y);
    auto sum = matrix_sum(prod, minus_eye);
    //computes ||x*y - I||^2, which we'll minimize
    auto sqr = matrix_l2(sum);
    //do  gradient descent
    //clock
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < 10000; i++)
    {
        sqr.zero_grad();
        sqr.eval();
        sqr.eval_grad();
        x.gradient_descent(0.0004f);
        if (i % 1000 == 0)
        {
			std::cout << "Iteration " << i << '\n';
            //compute time taken
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            std::cout << "Time taken : " << duration.count() << " microseconds\n";
            start = std::chrono::high_resolution_clock::now();
        }
    }
    
    //download result from gpu and print it
    x.download();
    for (size_t i = 0; i < matrix_size; i++)
    {
        for (size_t j = 0; j < matrix_size; j++)
        {
            std::cout << x.data[j + i * matrix_size] << ", ";
        }
        std::cout << '\n';

    }
    
}

void test_1()
{
    auto inp_1 = weight_matrix<2, 2>();
    auto inp_2 = weight_matrix<2, 2>();
    auto sum = matrix_sum(inp_1, inp_2);
    auto prod = matrix_product(inp_1, inp_2);
    auto l2 = matrix_l2(prod);
    for (size_t i = 0; i < 4; i++)
	{
		inp_1.data[i] = i;
		inp_2.data[i] = i;
	}
    sum.eval();
	prod.eval();
    l2.eval();
    l2.eval_grad();
    float res_sum[4];
    float res_prod[4];
    float res_grad_left[4];
    float res_grad_right[4];
    float norm;
    cudaMemcpy(res_sum, sum.gpu_data, 4 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(res_prod, prod.gpu_data, 4 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&norm, l2.gpu_data, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(res_grad_left, inp_1.grad_data, 4 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(res_grad_right, inp_2.grad_data, 4 * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Sum : \n";
    for (size_t i = 0; i < 2; i++)
    {
        for (size_t j = 0; j < 2; j++)
        {
            std::cout << res_sum[j + i * 2] << ", ";
        }
        std::cout << '\n';
    }
    std::cout << "Prod : \n";
    for (size_t i = 0; i < 2; i++)
    {
        for (size_t j = 0; j < 2; j++)
        {
            std::cout << res_prod[j + i * 2] << ", ";
        }
        std::cout << '\n';
    }

    std::cout << "Norm : " << norm << '\n';

    std::cout << "Grad left : \n";
    for (size_t i = 0; i < 2; i++)
    {
        for (size_t j = 0; j < 2; j++)
        {
            std::cout << res_grad_left[j + i * 2] << ", ";
        }
        std::cout << '\n';
    }
    std::cout << "Grad right : \n";
	for (size_t i = 0; i < 2; i++)
	{
		for (size_t j = 0; j < 2; j++)
		{
			std::cout << res_grad_right[j + i * 2] << ", ";
		}
		std::cout << '\n';
	}

}

void test_2()
{
    auto inp_1 = weight_matrix<2, 3>();
    auto inp_2 = weight_matrix<2, 1>();
    for (size_t i = 0; i < 6; i++)
    {
        inp_1.data[i] = i;
    }
    inp_2.data[0] = 1;
    inp_2.data[1] = 2;
    auto sum = matrix_sum_all_cols(inp_1, inp_2);
    auto norm = matrix_l2(sum);
    norm.eval();
    norm.eval_grad();
    float sum_result[6];
    float res_grad[6];
    cudaMemcpy(res_grad, inp_1.grad_data, 6 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(sum_result, sum.gpu_data, 6 * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Sum : \n";
    for (size_t i = 0; i < 2; i++)
	{
		for (size_t j = 0; j < 3; j++)
		{
			std::cout << sum_result[j + i * 3] << ", ";
		}
		std::cout << '\n';
	}
    std::cout << "Grad : \n";
    for (size_t i = 0; i < 2; i++)
    {
        for (size_t j = 0; j < 3; j++)
        {
            std::cout << res_grad[j + i * 3] << ", ";
        }
        std::cout << '\n';
    }
    float res_grad_2[2];
    cudaMemcpy(res_grad_2, inp_2.grad_data, 2 * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Grad 2 : \n";
    for (size_t i = 0; i < 2; i++)
    {
        std::cout << res_grad_2[i] << "\n";
    }
}

int main()
{
    cublasCreate(&cublas_handle);
    matrix_inversion_example();
    //mlp_example();
    //test_1();
    //test_2();
    cublasDestroy(cublas_handle);
}
