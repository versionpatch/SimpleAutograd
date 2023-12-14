
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <random>


//eval_grad definition : 
// Suppose current node is f(g(x1,x2,...)), where f is parent, g is current node and x1,x2,... are children
// parent_grad is the gradient of the parent node evaluated at g(x1,x2,...)
// eval_grad should compute the partial differential of f(g) with respect to x_i, where x_i is an input of the current node, and pass it to the child node i

static constexpr bool PRINT_LOSS = true;

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


//Could be improved by using shared memory and improving memory coalescing, or just by using cuBLAS
__global__ void mat_mul(const float* a, const float* b, float* result, int n, int mid, int m)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n && j < m)
    {
        float sum = 0;
        for (int k = 0; k < mid; k++)
            sum += a[k + i * mid] * b[j + k * m];
        result[j + i * m] = sum;
    }
}

// parent_grad * op2_data^T
__global__ void mat_mul_grad_left(const float* parent_grad, const float* op2_data, float* left_grad_out, int n, int m, int cols)
{
    int i_row = blockIdx.y * blockDim.y + threadIdx.y;
    int i_col = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_row < n && i_col < m)
    {
        float sum = 0;
        for (int c = 0; c < cols; c++)
        {
            sum += parent_grad[c + i_row * cols] * op2_data[c + i_col * cols];
        }
        left_grad_out[i_col + m * i_row] += sum;
    }
}

// op1_data^T * parent_grad
__global__ void mat_mul_grad_right(const float* parent_grad, const float* op1_data, float* op2_grad, int n, int m, int rows)
{
    int i_row = blockIdx.y * blockDim.y + threadIdx.y;
    int i_col = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_row < n && i_col < m)
    {
        float sum = 0;
        for (int r = 0; r < rows; r++)
            sum += parent_grad[i_col + r * m] * op1_data[i_row + r * n];
        op2_grad[i_col + m * i_row] += sum;
    }
}

__global__ void mat_sum(const float* a, const float* b, float* result, int n, int m)
{
    int i_row = blockIdx.y * blockDim.y + threadIdx.y;
    int i_col = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_row < n && i_col < m)
    {
        result[i_col + m * i_row] = a[i_col + m * i_row] + b[i_col + m * i_row];
    }
}

__global__ void mat_sum_all_cols(const float* a, const float* b, float* result, int n, int m)
{
    int i_row = blockIdx.y * blockDim.y + threadIdx.y;
    int i_col = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_row < n && i_col < m)
    {
        result[i_col + m * i_row] = a[i_col + m * i_row] + b[i_row];
    }
}

__global__ void mat_sum_all_cols_grad(const float* parent_grad, float* output_grad, int n, int m)
{
	int i_row = blockIdx.x * blockDim.x + threadIdx.x;
	if (i_row < n)
	{
		float sum = 0;
		for (int i = 0; i < m; i++)
			sum += parent_grad[i + m * i_row];
		output_grad[i_row] += sum;
	}
}

__global__ void mat_grad_desc(const float* grad, float* parameters, float alpha, int n, int m)
{
    int i_row = blockIdx.y * blockDim.y + threadIdx.y;
    int i_col = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_row < n && i_col < m)
    {
        parameters[i_col + m * i_row] -= alpha * grad[i_col + m * i_row];
    }
}

__global__ void elementwise_square_plus_one_stride(const float* input, float* output, uint32_t stride , uint32_t n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < stride && i + stride < n)
        output[i] = 0.5f * (input[i] * input[i] + input[i + stride] * input[i + stride]);
    else if (i < stride && i < n)
        output[i] = 0.5f * (input[i] * input[i]);

}

__global__ void l2_grad(const float* parent_grad, float* child_value, float* out, int n, int m)
{
    int i_row = blockIdx.y * blockDim.y + threadIdx.y;
    int i_col = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_row < n && i_col < n)
	{
		out[i_col + n * i_row] += parent_grad[0] * child_value[i_col + m * i_row];
	}
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

__global__ void reduce(float* result, uint32_t stride, uint32_t n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < stride && i + stride < n)
    {
        result[i] += result[i + stride];
    }
}

__global__ void direct_square_reduce(const float* input, float* result, uint32_t n)
{
    result[0] = 0;
    for (int i = 0; i < n; i++)
    {
        result[0] += 0.5f * input[i] * input[i];
    }
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
        cudaMalloc(&gpu_data, OP::rows * OP::cols * sizeof(float));
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
        size_t size = OP::rows * OP::cols;
        size_t stride = next_power_of_two(size) / 2;
        constexpr int threads_per_block = 64;
        if (stride < threads_per_block)
		{
		    direct_square_reduce << < dim3(1, 1, 1), dim3(1, 1, 1) >> > (op.gpu_data, gpu_data, size);
		}
        else
        {
            int num_blocks_sqr = (stride > threads_per_block) ? stride / threads_per_block : 1;
            elementwise_square_plus_one_stride << < dim3(num_blocks_sqr, 1, 1), dim3(threads_per_block, 1, 1) >> > (op.gpu_data, gpu_data, stride, size);
            for (stride /= 2; stride > 0; stride /= 2)
            {
                int num_blocks = (stride > threads_per_block) ? stride / threads_per_block : 1;
                reduce << < num_blocks, threads_per_block >> > (gpu_data, stride, size);
            }
        }
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
        int blocks_x = cols > 16 ? cols / 16 : 1;
        int blocks_y = rows > 16 ? cols / 16 : 1;
        l2_grad << < dim3(blocks_x, blocks_y), dim3(16, 16) >> > (parent_grad, op.gpu_data, grad_data, rows, cols);
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
        int blocks_x = cols > 16 ? cols / 16 : 1;
        int blocks_y = rows > 16 ? cols / 16 : 1;
        mat_mul << < dim3(blocks_x, blocks_y), dim3(16, 16) >> > (op1.gpu_data, op2.gpu_data, gpu_data, rows, OP1::cols, cols);
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
        int blocks_x = OP1::cols > 16 ? OP1::cols / 16 : 1;
        int blocks_y = OP1::rows > 16 ? OP1::rows / 16 : 1;
        mat_mul_grad_left << < dim3(blocks_x, blocks_y), dim3(16, 16) >> > (parent_grad, op2.gpu_data, op1_grad, OP1::rows, OP1::cols, cols);
        
        blocks_x = OP2::cols > 16 ? OP2::cols / 16 : 1;
        blocks_y = OP2::rows > 16 ? OP2::rows / 16 : 1;
        mat_mul_grad_right << < dim3(blocks_x, blocks_y), dim3(16, 16) >> > (parent_grad, op1.gpu_data, op2_grad, OP2::rows, OP2::cols, rows);
        
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
        cudaMalloc(&gpu_data, rows * cols * sizeof(float));
        cudaMalloc(&op1_grad, OP1::rows * OP1::cols * sizeof(float));
        cudaMalloc(&op2_grad, OP2::rows * OP2::cols * sizeof(float));
        op1.num_parents++;
        op2.num_parents++;
    };
    ~matrix_sum_all_cols()
    {
        cudaFree(&gpu_data);
        cudaFree(&op1_grad);
        cudaFree(&op2_grad);
    }
    void eval()
    {
        op1.eval();
        op2.eval();
        int blocks_x = cols > 16 ? cols / 16 : 1;
        int blocks_y = rows > 16 ? cols / 16 : 1;
        mat_sum_all_cols << < dim3(blocks_x, blocks_y), dim3(16, 16) >> > (op1.gpu_data, op2.gpu_data, gpu_data, rows, cols);
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
        int blocks = rows > 16 ? rows / 16 : 1;
        mat_sum_all_cols_grad << < dim3(blocks, 1), dim3(1, 1) >> > (parent_grad, op2_grad, rows, cols);
        int blocks_x = cols > 16 ? cols / 16 : 1;
        int blocks_y = rows > 16 ? cols / 16 : 1;
        mat_sum << < dim3(blocks_x, blocks_y), dim3(16, 16) >> > (op1_grad, parent_grad, op1_grad, rows, cols);
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
        int blocks_x = cols > 16 ? cols / 16 : 1;
        int blocks_y = rows > 16 ? cols / 16 : 1;
        mat_sum << < dim3(blocks_x, blocks_y), dim3(16, 16) >> > (op1.gpu_data, op2.gpu_data, gpu_data, rows, cols);
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
        int blocks_x = cols > 16 ? cols / 16 : 1;
        int blocks_y = rows > 16 ? cols / 16 : 1;
		mat_sum << < dim3(blocks_x, blocks_y), dim3(16, 16) >> > (parent_grad, grad_data, grad_data, rows, cols);
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
        int blocks_x = cols > 16 ? cols / 16 : 1;
        int blocks_y = rows > 16 ? cols / 16 : 1;
        mat_sum << < dim3(blocks_x, blocks_y), dim3(16, 16) >> > (parent_grad, grad_data, grad_data, rows, cols);
    }

    void gradient_descent(float alpha)
    {
        int blocks_x = cols > 16 ? cols / 16 : 1;
        int blocks_y = rows > 16 ? cols / 16 : 1;
        mat_grad_desc << < dim3(blocks_x, blocks_y), dim3(16, 16) >> > (grad_data, gpu_data, alpha, rows, cols);
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
    
    for (size_t i = 0; i < 10000; i++)
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
    constexpr size_t matrix_size = 32;
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
    for (size_t i = 0; i < 10000; i++)
    {
        sqr.zero_grad();
        sqr.eval();
        sqr.eval_grad();
        x.gradient_descent(0.04f);
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

int main()
{
    //matrix_inversion_example();
    mlp_example();
}
