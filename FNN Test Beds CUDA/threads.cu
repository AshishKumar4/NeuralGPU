#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "thrust/device_vector.h"
#include "thrust/host_vector.h"

#include "thrust/random/normal_distribution.h"
#include "thrust/random/linear_congruential_engine.h"

#include "sm_35_atomic_functions.h"

#include <stdio.h>
#include <iostream>
#include <vector>

#define ACTIVATION_SIGMOID_2R
#define ACTIVATION_DIFF_TANH

#ifdef MNIST_DIGIT_RECOG
#include "idx.h"
#endif

#include "Neuron.h"

__device__ double sigmoid(double in)
{
	return  1 / (1 + expf(-in));
}

__device__ double sigmoid2r(double in)
{
	return (2*(1 / (1 + expf(-in)))) - 1;
}

__global__ void cuda_dot_product_vec_matrixTranspose(float* output, float* x, float* y, int n, int m)
{
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if( i < n )
	{
		int row = i;
		{
			output[row] = 0;
			for(int k = 0; k < m; k++)
			{
				output[row] += x[ k ] * y[ row*m + k ];//interm[row*m + k];
			}
		}
	}
}

__global__ void cuda_dot_product_vec_matrixDirect(float* output, float* x, float* y, int n, int m)
{
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if( i < n )
	{
		int row = i;//i / m;
		output[row] = 0;
		for(int k = 0; k < m; k++)
		{
			output[row] += x[ k ] * y[ k*n + row ];//interm[row*m + k];
		}
	}
}

// __global__ void cuda_dot_productTranspose(float* output, float* x, float* y)
// {
// 	extern __shared__ float interm[];
// 	int i = blockIdx.x;
// 	int j = threadIdx.x;
// 	interm[ i*blockDim.x + j ] = x[j] * y[ i*blockDim.x + j ];
//     __syncthreads();
//     if(j == 0)
//     {
// 		output[i] = 0;
//         for(int k = 0; k < blockDim.x; k++)
//         {
//             output[i] += 1;//interm[ i*blockDim.x + k ];
//         }
//     }
// }

// __global__ void cuda_dot_product(float* output, float* x, float* y)
// {
// 	extern __shared__ float interm[];
// 	int i = blockIdx.x;
// 	int j = threadIdx.x;
// 	interm[ i*blockDim.x + j ] = x[j] * y[ j*gridDim.x + i ];
//     __syncthreads();
//     if(j == 0)
//     {
// 		output[i] = 0;
//         for(int k = 0; k < blockDim.x; k++)
//         {
//             output[i] += interm[ i*blockDim.x + k ];
//         }
//     }
// }

__global__ void cuda_cross_product(float* output, float* x, float *y, int xdim, int ydim)
{
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(i < xdim*ydim)
	{
		int column = i % ydim;
		int row = i / ydim;
		output[ i ] = x[row] * y[column];
	}
}

__global__ void cuda_hadamard_product(float* output, float* x, float* y, int n)
{
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(i < n)
		output[i] = x[i] * y[i];
}

__global__ void cuda_subtract(float* output, float* x, float* y, int n)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(i < n)
		output[i] = x[i] - y[i];
}

__global__ void cuda_vector_scalar_multiply(float* output, float* x, float y, int n)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(i < n)
		output[i] = x[i] * y;
}

__global__ void cuda_activation_RELU(float* output, float* input, int n)
{
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(i < n)
	{
		if(input[i] > 0)
			output[i] = input[i];
		else output[i] = 0;
	}
}

__global__ void cuda_activation_Linear(float* output, float* input, int n)
{
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(i < n)
		output[i] = input[i];
}

__global__ void cuda_activation_Sigmoid(float* output, float* input, int n)
{
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(i < n)
		output[i] = sigmoid(input[i]);
}

__global__ void cuda_activation_TANH(float* output, float* input, int n)
{
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(i < n)
		output[i] = tanhf(input[i]);
}

__global__ void cuda_activation_SOFTMAX_SUM(float* output, float* input, int n)
{
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	if(i < n)
	{
		__shared__ float sum;
		extern __shared__ float tmps[];
		tmps[i] = powf(2, input[i]);
		__syncthreads();
		if(i == 0)
		{
			sum = 0;
			for(int j = 0; j < blockDim.x * gridDim.x; j++)
			{
				sum += tmps[j];
			}
			if(sum == 0) sum = 1;
			if(isnan(sum)) sum = 1;
		}
		__syncthreads();
		output[i] = tmps[i] / sum;
	}
}

__global__ void cuda_activation_SOFTMAX(float* output, float* input, int n)
{
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	__shared__ float sum;
	extern __shared__ float interm[];
	if(i < n)
	{
		interm[i] = powf(12, input[i]);
		__syncthreads();
		if(i == 0)
		{
			sum = 0;
			for(int j = 0; j < n; j++)
			{
				sum += interm[j];
			}
			if(sum == 0) sum = 1;
		}
		__syncthreads();
		output[i] = interm[i] / sum;
	}
}

// __global__ void cuda_activation_SOFTMAX(float* output, float* input, int n)
// {
// 	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	
// 	if(i == 0)
// 	{
// 		float sum = 0;
// 		for(int j = 0; j < n; j++)
// 		{
// 			sum += powf(12, input[j]);
// 		}
// 		if(sum == 0) sum = 1;
// 		for(int j = 0; j < n; j++)
// 		{
// 			output[j] = powf(12, input[j]) / sum;
// 		}
// 	}
// }

__global__ void cuda_derivative_RELU(float* output, float* input, int n)
{
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(i < n)
	{
		if(input[i] > 0)
			output[i] = 1;
		else output[i] = 0;
	}
}

__global__ void cuda_derivative_Linear(float* output, float* input, int n)
{
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(i < n)
		output[i] = 1;
}

__global__ void cuda_derivative_Sigmoid(float* output, float* input, int n)
{
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	if(i < n)
		output[i] = input[i] * (1 - input[i]);
}

__global__ void cuda_derivative_TANH(float* output, float* input, int n)
{
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(i < n)
		output[i] = 1 - input[i] * input[i];
}

__global__ void cuda_derivative_SOFTMAX(float* output, float* input, int n)
{
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(i < n)
		output[i] = 1;//input[i];
}

/****************************************************************************************************/

__global__ void cuda_Loss_LC(float* output, float* x, float* y, int n)
{
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(i < n)
		output[i] = y[i] - x[i];
}


__global__ void cuda_identity(float* output, float* input, int n)
{
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(i < n)
		output[i] = input[i];
}