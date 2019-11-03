#pragma once

#include "stdlib.h"
#include "stdio.h"
#include "stdint.h"
#include "string.h"
#include "iostream"

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

#define MNIST_DIGIT_RECOG

#define THREADS_PER_BLOCK 128

// Round up to the nearest multiple of n
#define ROUNDUP(a, n)     ({uint64_t __n = (uint64_t)(n);(typeof(a))(ROUNDDOWN((uint64_t)(a) + __n - 1, __n));})

float* oneHotEncode(uint8_t value, int sz = 10);
uint8_t oneHotDecode(float* val, int sz = 10);

float calcMean(uint8_t* arr, int n);
float standardDeviation(uint8_t* arr, int n);
float* normalizeArray(uint8_t* arr, int n);

void printVector(float* vec, int size);
void itoa(unsigned i,char* buf, unsigned base);
float* NN_FeedEncoder(int n, int base, int sz, float lower_bound, float upper_bound);
int NN_FeedDecoder(float* val, int base, int sz, float lower_bound, float upper_bound);	// Currently only binary