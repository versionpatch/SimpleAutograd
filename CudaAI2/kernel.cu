
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>

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

__global__ void mat_grad_desc(const float* grad, float* val, float alpha, int n, int m)
{
    int i_row = blockIdx.y * blockDim.y + threadIdx.y;
    int i_col = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_row < n && i_col < m)
    {
        val[i_col + m * i_row] -= alpha * grad[i_col + m * i_row];
    }
}

__global__ void elementwise_square(const float* o, float* a, uint32_t stride , uint32_t n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < stride && i + stride < n)
        a[i] = 0.5f * (o[i] * o[i] + o[i + stride] * o[i + stride]);
    else if (i < stride)
        a[i] = 0.5f * (o[i] * o[i]);

}


__global__ void reduce(float* a, uint32_t stride, uint32_t n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < stride)
    {
        a[i] += a[i + stride];
    }
}

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
        int num_blocks_sqr = (stride > threads_per_block) ? stride / threads_per_block : 1;
        elementwise_square << < dim3(num_blocks_sqr, 1, 1), dim3(threads_per_block, 1, 1) >> > (op.gpu_data, gpu_data, stride, size);
        for (stride /= 2; stride != threads_per_block / 2; stride /= 2)
        {
            int num_blocks = (stride > threads_per_block) ? stride / threads_per_block : 1;
            reduce << < num_blocks, threads_per_block >> > (gpu_data, stride, size);
        }
    }

    void eval_grad()
    {
        op.eval_grad(op.gpu_data);
    }


    OP& op;
    float* gpu_data;
};

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

template<size_t n, size_t m>
struct input_matrix
{
public:
    static constexpr int rows = n;
    static constexpr int cols = m;

    input_matrix() : data(std::vector<float>(rows* cols, 0))
    {
        cudaMalloc(&gpu_data, rows * cols * sizeof(float));
        dirty = true;
    }
    ~input_matrix()
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


int main()
{
    constexpr size_t matrix_size = 1024;
    auto x = input_matrix<matrix_size, matrix_size>();
    auto y = input_matrix<matrix_size, matrix_size>();
    auto minus_eye = input_matrix<matrix_size, matrix_size>();
    //example, computing inverse of 1024x1024 matrix
    for (size_t i = 0; i < matrix_size; i++)
    {
        x.data[i * (1 + matrix_size)] = 2.0f;
        y.data[i * (1 + matrix_size)] = 1.0f;
        minus_eye.data[i * (1 + matrix_size)] = -1.0f;
    }
    auto prod = matrix_product(x, y);
    auto sum = matrix_sum(prod, minus_eye);
    auto sqr = matrix_l2(sum);

    auto start = std::chrono::system_clock::now();
    for (size_t i = 0; i < 1000; i++)
    {
        sqr.eval();
        sqr.eval_grad();
        x.gradient_descent(0.01f);
        if (i % 100 == 0 && i > 0)
        {
            auto end = std::chrono::system_clock::now();
            auto time = std::chrono::duration<double>(end - start);
            std::cout << "Finished 100 iterations in " << time.count();
            start = end;
            std::cout << '\n';
        }
    }
    
    x.download();
    for (size_t i = 0; i < matrix_size; i++)
    {
        for (size_t j = 0; j < matrix_size; j++)
        {
            std::cout << (int)x.data[j + i * matrix_size] << ", ";
        }
        std::cout << '\n';
        
    }
    

    std::cout << '\n';

}
