
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>


//eval_grad definition : 
// Suppose current node is f(g(x1,x2,...)), where f is parent, g is current node and x1,x2,... are children
// parent_grad is the gradient of the parent node evaluated at g(x1,x2,...)
// eval_grad should compute the partial differential of f(g) with respect to x_i, where x_i is an input of the current node, and pass it to the child node i


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
__global__ void mat_mul(const float* a, const float* b, float* c, int n, int mid, int m)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n && j < m)
    {
        float sum = 0;
        for (int k = 0; k < mid; k++)
            sum += a[k + i * mid] * b[j + k * m];
        c[j + i * m] = sum;
    }
}

__global__ void mat_mul_grad_left(const float* parent_grad, const float* op2_data, float* op1_grad, int n, int m, int cols)
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
        op1_grad[i_col + m * i_row] = sum;
    }
}

__global__ void mat_mul_grad_right(const float* parent_grad, const float* op1_data, float* op2_grad, int n, int m, int rows)
{
    int i_row = blockIdx.y * blockDim.y + threadIdx.y;
    int i_col = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_row < n && i_col < m)
    {
        float sum = 0;
        for (int r = 0; r < rows; r++)
            sum += parent_grad[i_col + r * m] * op1_data[i_row + r * n];
        op2_grad[i_col + m * i_row] = sum;
    }
}

__global__ void mat_sum(const float* a, const float* b, float* c, int n, int m)
{
    int i_row = blockIdx.y * blockDim.y + threadIdx.y;
    int i_col = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_row < n && i_col < m)
    {
        c[i_col + m * i_row] = a[i_col + m * i_row] + b[i_col + m * i_row];
    }
}

__global__ void mat_sum_all_cols(const float* a, const float* b, float* c, int n, int m)
{
    int i_row = blockIdx.y * blockDim.y + threadIdx.y;
    int i_col = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_row < n && i_col < m)
    {
        c[i_col + m * i_row] = a[i_col + m * i_row] + b[i_row];
    }
}

__global__ void mat_sum_all_cols_grad(const float* parent_grad, float* a, int n, int m)
{
	int i_row = blockIdx.x * blockDim.x + threadIdx.x;
	if (i_row < n)
	{
		float sum = 0;
		for (int i = 0; i < m; i++)
			sum += parent_grad[i + m * i_row];
		a[i_row] = sum;
	}
}

__global__ void mat_grad_desc(const float* grad, float* val, float alpha, int n, int m)
{
    int i_row = blockIdx.y * blockDim.y + threadIdx.y;
    int i_col = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_row < n && i_col < m)
    {
        val[i_col + m * i_row] -= alpha * grad[i_col + m * i_row];
    }
}

__global__ void elementwise_square_plus_one_stride(const float* o, float* a, uint32_t stride , uint32_t n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < stride && i + stride < n)
        a[i] = 0.5f * (o[i] * o[i] + o[i + stride] * o[i + stride]);
    else if (i < stride && i < n)
        a[i] = 0.5f * (o[i] * o[i]);

}

__global__ void elementwise_tanh(const float* o, float* a, uint32_t n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n)
		a[i] = tanhf(o[i]);
}

//f' = 1 - f^2
__global__ void tanh_grad(const float* parent_grad, const float* o, float* a, uint32_t n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n)
        a[i] = parent_grad[i] * (1.0f - o[i] * o[i]);   
}

__global__ void reduce(float* a, uint32_t stride, uint32_t n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < stride && i + stride < n)
    {
        a[i] += a[i + stride];
    }
}

__global__ void direct_square_reduce(const float* o, float* a, uint32_t n)
{
    a[0] = 0;
    for (int i = 0; i < n; i++)
    {
        a[0] += 0.5f * o[i] * o[i];
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
        elementwise_tanh << < dim3(num_blocks, 1, 1), dim3(num_blocks, 1, 1) >> > (op.gpu_data, gpu_data, size);
    }

    void eval_grad(float* parent_grad)
    {
        size_t size = OP::rows * OP::cols;
        constexpr int threads_per_block = 64;
        int num_blocks = 1 > size / threads_per_block ? 1 : size / threads_per_block;
        tanh_grad << < dim3(1, 1, 1), dim3(OP::rows, OP::cols, 1) >> > (parent_grad, op.gpu_data, grad_data, size);
        op.eval_grad(grad_data);
    }


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
    }
    ~matrix_l2()
    {
        cudaFree(&gpu_data);
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
        float x;
        cudaMemcpy(&x, gpu_data, sizeof(float), cudaMemcpyDeviceToHost);
        std::cout << "Error : " << x << '\n';
    }

    void eval_grad()
    {
        op.eval_grad(op.gpu_data);
    }


    OP& op;
    float* gpu_data;
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

    void eval_grad(float* parent_grad)
    {
        int blocks_x = OP1::cols > 16 ? OP1::cols / 16 : 1;
        int blocks_y = OP1::rows > 16 ? OP1::rows / 16 : 1;
        mat_mul_grad_left << < dim3(blocks_x, blocks_y), dim3(16, 16) >> > (parent_grad, op2.gpu_data, op1_grad, OP1::rows, OP1::cols, cols);
        op1.eval_grad(op1_grad);
        blocks_x = OP2::cols > 16 ? OP2::cols / 16 : 1;
        blocks_y = OP2::rows > 16 ? OP2::rows / 16 : 1;
        mat_mul_grad_right << < dim3(blocks_x, blocks_y), dim3(16, 16) >> > (parent_grad, op1.gpu_data, op2_grad, OP2::rows, OP2::cols, rows);
        op2.eval_grad(op2_grad);
    }


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
        cudaMalloc(&op2_grad, OP2::rows * OP2::cols * sizeof(float));
    };
    ~matrix_sum_all_cols()
    {
        cudaFree(&gpu_data);
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
    void eval_grad(float* parent_grad)
    {
        int blocks = rows > 16 ? rows / 16 : 1;
        mat_sum_all_cols_grad << < dim3(blocks, 1), dim3(1, 1) >> > (parent_grad, op2_grad, rows, cols);
        op2.eval_grad(op2_grad);
        op1.eval_grad(parent_grad);
    }

    float* gpu_data;
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
    }
    ~matrix_sum()
    {
        cudaFree(&gpu_data);
    }

    void eval()
    {
        op1.eval();
        op2.eval();
        int blocks_x = cols > 16 ? cols / 16 : 1;
        int blocks_y = rows > 16 ? cols / 16 : 1;
        mat_sum << < dim3(blocks_x, blocks_y), dim3(16, 16) >> > (op1.gpu_data, op2.gpu_data, gpu_data, rows, cols);
    }

    void eval_grad(float* parent_grad)
    {
        op1.eval_grad(parent_grad);
        op2.eval_grad(parent_grad);
    }


    OP1& op1;
    OP2& op2;
    float* gpu_data;
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

	void eval_grad(float* parent_grad)
	{
		
	}

    void next_data()
    {
        current_data_idx = (current_data_idx + 1) % num_vectors;
		gpu_data = full_data + current_data_idx * rows * cols;
    }

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

    void eval_grad(float* parent_grad)
    {
        grad_data = parent_grad;
    }

    void gradient_descent(float alpha)
    {
        int blocks_x = cols > 16 ? cols / 16 : 1;
        int blocks_y = rows > 16 ? cols / 16 : 1;
        mat_grad_desc << < dim3(blocks_x, blocks_y), dim3(16, 16) >> > (grad_data, gpu_data, alpha, rows, cols);
    }

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
    
    auto x = input_matrix<2, 2>(256);
    for (size_t i = 0; i < 128; i++)
	{
		x.data[i * 4] = 0.5f * cosf(3.1415f * i / 256.0f);
		x.data[i * 4 + 2] = 0.5f * sinf(3.1415f * i / 256.0f);
        x.data[i * 4 + 1] = 0.5f * cosf(3.1415f * i / 256.0f) + 0.5f;
        x.data[i * 4 + 3] = -0.5f * sinf(3.1415f * i / 256.0f) + 0.3f;
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

    //training
    float lr = 0.6f;
    
    for (size_t i = 0; i < 1000; i++)
    {
        sqr.eval();
        sqr.eval_grad();
        
        W1.gradient_descent(lr);
        W2.gradient_descent(lr);
        b1.gradient_descent(lr);
        b2.gradient_descent(lr);
        y.next_data();
        x.next_data();

        lr -= 0.0005f;
        if (lr < 0.001f)
            lr = 0.001f;
    }
    


}

void matrix_inversion_example()
{
    constexpr size_t matrix_size = 32;
    auto x = weight_matrix<matrix_size, matrix_size>();
    auto y = input_matrix<matrix_size, matrix_size>(1);
    auto minus_eye = input_matrix<matrix_size, matrix_size>(1);
    
    //fill matrices
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
    for (size_t i = 0; i < 1000; i++)
    {
        sqr.eval();
        sqr.eval_grad();
        x.gradient_descent(0.02f);
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
