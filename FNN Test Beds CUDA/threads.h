#pragma once
#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "thrust/device_vector.h"
#include "thrust/host_vector.h"
#include "sm_35_atomic_functions.h"

#include "idx.h"
#include "Neuron.h"

__global__ void cuda_dot_product_vec_matrixTranspose(float* output, float* x, float* y, int n, int m);
__global__ void cuda_dot_product_vec_matrixDirect(float* output, float* x, float* y, int n, int m);
// __global__ void cuda_dot_productTranspose(float* output, float* x, float* y);
// __global__ void cuda_dot_product(float* output, float* x, float* y);
__global__ void cuda_cross_product(float* output, float* x, float *y, int xdim, int ydim);
__global__ void cuda_hadamard_product(float* output, float* x, float* y, int n);
__global__ void cuda_subtract(float* output, float* x, float* y, int n);
__global__ void cuda_vector_scalar_multiply(float* output, float* x, float y, int n);

__global__ void cuda_activation_RELU(float* output, float* input, int n);
__global__ void cuda_activation_Linear(float* output, float* input, int n);
__global__ void cuda_activation_Sigmoid(float* output, float* input, int n);
__global__ void cuda_activation_TANH(float* output, float* input, int n);
__global__ void cuda_activation_SOFTMAX_SUM(float* output, float* input, float* sum, int n);
__global__ void cuda_activation_SOFTMAX(float* output, float* input, int n);//, float* sum, int n);
__global__ void cuda_derivative_RELU(float* output, float* input, int n);
__global__ void cuda_derivative_Linear(float* output, float* input, int n);
__global__ void cuda_derivative_Sigmoid(float* output, float* input, int n);
__global__ void cuda_derivative_TANH(float* output, float* input, int n);
__global__ void cuda_derivative_SOFTMAX(float* output, float* input, int n);
__global__ void cuda_Loss_LC(float* output, float* x, float* y, int n);
__global__ void cuda_identity(float* output, float* input, int n);