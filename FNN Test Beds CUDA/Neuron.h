#pragma once

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
#include <random>

#include "threads.h"

#include "usableLib.h"

#ifdef MNIST_DIGIT_RECOG
#include "idx.h"
#endif

using namespace std;
using namespace thrust;

float* NN_FeedEncoder(int n, int base, int sz, float lower_bound = - 1, float upper_bound = 1);
int NN_FeedDecoder(float* val, int base, int sz, float lower_bound = - 1, float upper_bound = 1);	// Currently only binary

class NeuralLayer
{
public:
	float* output;
	float* sum;
	float* derivative;

	NeuralLayer* nextLayer;
	NeuralLayer* lastLayer;
	
	int activation_type;
	int size;

	virtual float* ForwardProp(float* input) = 0;
	virtual float* BackwardProp(float* delta) = 0;
	virtual float* UpdateParam(float* lastOutput, float learning_rate) = 0;

	NeuralLayer(int size, int activationType) : size(size), activation_type(activationType)
	{
		nextLayer = NULL;
		lastLayer = NULL;

		cudaMalloc(&output, sizeof(float) * size);
		cudaMalloc(&sum, sizeof(float) * size);
		cudaMalloc(&derivative, sizeof(float) * size);
	}

	NeuralLayer(int size, int activationType, NeuralLayer* lastLayer) : size(size), activation_type(activationType), lastLayer(lastLayer)
	{
		lastLayer->nextLayer = this;
		cudaMalloc(&output, sizeof(float) * size);
		cudaMalloc(&sum, sizeof(float) * size);
		cudaMalloc(&derivative, sizeof(float) * size);
	}
};

enum ACTIVATION_FUNCTIONS 
{
	RELU = 0,
	LINEAR,
	SIGMOID,
	TANH,
	SOFTMAX
};

enum LOSS_FUNCTIONS
{
	LOSS_CROSSENTROPY = 0,
	MEAN_SQUARED
};

class DenseLayer : public NeuralLayer
{
public:
	float* weights;
	float* deltas;
	float* errors;
	float* backError;
	float* bias;

	int prevSize;

	float* ForwardProp(float* input);
	float* BackwardProp(float* delta);
	float* UpdateParam(float* lastOutput, float learning_rate);

	DenseLayer(int size, int activationType, NeuralLayer* lastLayer = NULL) : NeuralLayer(size, activationType, lastLayer)
	{
		cudaMalloc(&errors, sizeof(float) * size);
		cudaMalloc(&bias, sizeof(float) * size);
		if(lastLayer != NULL)
		{
			printf("\nInitializing Weights %dx%d...", lastLayer->size, size);
			cudaMalloc(&backError, sizeof(float) * lastLayer->size);
			cudaMalloc(&weights, sizeof(float) * lastLayer->size * size);
			cudaMalloc(&deltas, sizeof(float) * lastLayer->size * size);
			prevSize = lastLayer->size;

			float* _w = new float[lastLayer->size * size];
			
			std::minstd_rand rng;
			std::normal_distribution<double> dist(0, 1 / powf(lastLayer->size, 0.5));
			for(int i = 0; i < lastLayer->size * size; i++)
			{
				_w[i] = dist(rng);
			}

			cudaMemcpy(weights, _w, sizeof(float) * lastLayer->size * size, cudaMemcpyHostToDevice);
		}
		else 
		{
			perror("\nERROR! Attempt to initialize Dense Layer as Input Layer is not allowed!");
			exit(0);
		}
	}

	void setWeights(float* weightMatrix)
	{
		printf("\nWeight Size %d", lastLayer->size * size);
		float* _w = new float[lastLayer->size * size];
		for(int i = 0; i < lastLayer->size * size; i++)
		{
			_w[i] = weightMatrix[i];
		}

		cudaMemcpy(this->weights, _w, sizeof(float) * lastLayer->size * size, cudaMemcpyHostToDevice);
	}
};

class InputLayer : public NeuralLayer
{
public:

	float* ForwardProp(float* input)
	{
		// printf("{{%d}}", this->size);
		int blocks = this->size / THREADS_PER_BLOCK;
		cuda_identity <<< blocks+1, THREADS_PER_BLOCK >>> (this->output, input, this->size);
		return this->output;
	}
	float* BackwardProp(float* delta)
	{
		return delta;
	}
	float* UpdateParam(float* lastOutput, float learning_rate)
	{
		return lastOutput;
	}

	InputLayer(int size, int activationType) : NeuralLayer(size, activation_type)
	{
		
	}
};

class NeuralNet_FF
{
public:

	float* error;

	InputLayer* input;
	NeuralLayer* output;

	float learning_rate;
	float gamma;
	float delta;

	int loss_function;

	NeuralNet_FF(InputLayer* input, NeuralLayer* output, float learningRate, int lossFunction, float gamma, float delta): 
	input(input), output(output), learning_rate(learningRate), loss_function(lossFunction), gamma(gamma), delta(delta)
	{
		cudaMalloc(&error, sizeof(float) * output->size);
	}

	float* ForwardProp(float* in);
	void BackwardProp(float* error);
	void UpdateParams();

	float* calcError(float* output, float* expected);
};

class NeuralEngine
{
	void EngineSetup(float** _in, float** _out, int _samples);

	float** in_vec;
	float** out_vec;

	float** in_vec_orig;
	float** out_vec_orig;

public:
	NeuralNet_FF * nn;
	int in_sz;
	int out_sz;

	NeuralEngine(NeuralNet_FF* _nn) : nn(_nn)
	{

	}
	
	void Train(float** _in, float** _out, int _samples, int _epoch, bool _shuffle = true);
	std::tuple<float**, float*, float*, float, float> Test(float** _in, float** _out, int _samples);
	void Save_Data(std::tuple<float*, float*, float*, float*, float> mtp, int _samples);
};


void dot_productTranspose(float* output, float* x, float* y, int n, int m);
void dot_product(float* output, float* x, float* y, int n, int m);
void hadamard_product(float* output, float* x, float* y, int n, int m);
void cross_product(float* output, float* x, float* y, int xdim, int ydim);
void vector_subtract(float *output, float* x, float* y, int n, int m);