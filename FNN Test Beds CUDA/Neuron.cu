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

#include "usableLib.h"
#include "Neuron.h"
#include "threads.h"

#include <algorithm>    // std::shuffle

#include <random>       // std::default_random_engine
#include <chrono>       // std::chrono::system_clock

#include <fstream>
#include "cstdlib"
#include <functional>


using namespace std;

void vector_identity(float* output, float* input, int size)
{
	int blocks = size / THREADS_PER_BLOCK;
	cuda_identity <<< blocks+1, THREADS_PER_BLOCK >>> (output, input, size);
}

// n --> smallest dimension, m --> largest dimension
// ASSERTION : For Dot Product, the matrices shall meet critera of one dimension being equal
// DOT PRODUCT of two matrices
void dot_productTranspose(float* output, float* x, float* y, int n, int m)
{
	// Dot product = x . transpose(Y)
	// x dimension is 'm', y dimension is 'n x m'
	// Output dimension is n, input dimension is m
	// cuda_dot_productTranspose <<< n, m, n*m*sizeof(float) >>> (output, x, y);
	int blocks = n / THREADS_PER_BLOCK;
	cuda_dot_product_vec_matrixTranspose <<< blocks + 1, THREADS_PER_BLOCK >>> (output, x, y, n, m);
	
	cudaDeviceSynchronize();
}

// DOT PRODUCT of two matrices
void dot_product(float* output, float* x, float* y, int n, int m)
{
	// x dimension is 'm', y dimension is 'n x m'
	// Output dimension is n, input dimension is m
	// cuda_dot_product <<< n, m, n*m*sizeof(float) >>> (output, x, y);int blocks = n / THREADS_PER_BLOCK;
	int blocks = n / THREADS_PER_BLOCK;
	cuda_dot_product_vec_matrixDirect <<< blocks + 1, THREADS_PER_BLOCK >>> (output, x, y, n, m);

	cudaDeviceSynchronize();
}

void hadamard_product(float* output, float* x, float* y, int n, int m)
{
	// Element-wise multiplication
	int blocks = (n*m) / THREADS_PER_BLOCK;
	cuda_hadamard_product <<< blocks + 1, THREADS_PER_BLOCK >>> (output, x, y, n*m);
	cudaDeviceSynchronize();
}

void cross_product(float* output, float* x, float* y, int xdim, int ydim)
{
	// Output dimension is xdim, ydim
	
	int blocks = (xdim*ydim) / THREADS_PER_BLOCK;
	cuda_cross_product <<< blocks + 1, THREADS_PER_BLOCK >>> (output, x, y, xdim, ydim);
	cudaDeviceSynchronize();
}

void vector_subtract(float *output, float* x, float* y, int n, int m)
{
	// = x - y
	int blocks = (n*m) / THREADS_PER_BLOCK;
	cuda_subtract <<< blocks + 1, THREADS_PER_BLOCK >>> (output, x, y, n*m);

	cudaDeviceSynchronize();	
}

void vector_scalar_multiply(float* output, float* x, float val, int size)
{
	int blocks = (size) / THREADS_PER_BLOCK;
	cuda_vector_scalar_multiply <<< blocks + 1, THREADS_PER_BLOCK >>> (output, x, val, size);
	cudaDeviceSynchronize();
}

/****************************************************************************************************/

void activationRELU(float* output, float* input, int size)
{
	int blocks = (size) / THREADS_PER_BLOCK;
	cuda_activation_RELU <<< blocks + 1, THREADS_PER_BLOCK >>> (output, input, size);
	cudaDeviceSynchronize();
}

void activationLinear(float* output, float* input, int size)
{
	int blocks = (size) / THREADS_PER_BLOCK;
	cuda_activation_Linear <<< blocks + 1, THREADS_PER_BLOCK >>> (output, input, size);
	cudaDeviceSynchronize();
}

void activationSigmoid(float* output, float* input, int size)
{
	int blocks = (size) / THREADS_PER_BLOCK;
	cuda_activation_Sigmoid <<< blocks + 1, THREADS_PER_BLOCK >>> (output, input, size);
	cudaDeviceSynchronize();
}

void activationTANH(float* output, float* input, int size)
{
	int blocks = (size) / THREADS_PER_BLOCK;
	cuda_activation_TANH <<< blocks + 1, THREADS_PER_BLOCK >>> (output, input, size);
	cudaDeviceSynchronize();
}

float* softmaxSum;

void activationSOFTMAX(float* output, float* input, int size)
{
	int blocks = (size) / THREADS_PER_BLOCK;
	// cuda_activation_SOFTMAX <<< 1, 1 >>> (output, input, size);
	cuda_activation_SOFTMAX <<< blocks + 1, THREADS_PER_BLOCK, size*sizeof(float) >>> (output, input, size);
	cudaDeviceSynchronize();
}

std::function<void(float*, float*, int)> activationFunction[] = {	\
												&activationRELU, 	\
												&activationLinear, 	\
												&activationSigmoid,	\
												&activationTANH, 	\
												&activationSOFTMAX 	\
												};

/****************************************************************************************************/

void derivativeRELU(float* output, float* input, int size)
{
	int blocks = (size) / THREADS_PER_BLOCK;
	cuda_derivative_RELU <<< blocks + 1, THREADS_PER_BLOCK >>> (output, input, size);
	cudaDeviceSynchronize();
}

void derivativeLinear(float* output, float* input, int size)
{
	int blocks = (size) / THREADS_PER_BLOCK;
	cuda_derivative_Linear <<< blocks + 1, THREADS_PER_BLOCK >>> (output, input, size);
	cudaDeviceSynchronize();
}

void derivativeSigmoid(float* output, float* input, int size)
{
	int blocks = (size) / THREADS_PER_BLOCK;
	cuda_derivative_Sigmoid <<< blocks + 1, THREADS_PER_BLOCK >>> (output, input, size);
	cudaDeviceSynchronize();
}

void derivativeTANH(float* output, float* input, int size)
{
	int blocks = (size) / THREADS_PER_BLOCK;
	cuda_derivative_TANH <<< blocks + 1, THREADS_PER_BLOCK >>> (output, input, size);
	cudaDeviceSynchronize();
}

void derivativeSOFTMAX(float* output, float* input, int size)
{
	int blocks = (size) / THREADS_PER_BLOCK;
	cuda_derivative_SOFTMAX <<< blocks + 1, THREADS_PER_BLOCK >>> (output, input, size);
	cudaDeviceSynchronize();
}

std::function<void(float*, float*, int)> activationDerivative[] = {	\
												&derivativeRELU, 	\
												&derivativeLinear, 	\
												&derivativeSigmoid,	\
												&derivativeTANH, 	\
												&derivativeSOFTMAX, 	\
												};

/****************************************************************************************************/

void loss_LC_RELU(float* output, float* x, float* y, int size)
{
	int blocks = (size) / THREADS_PER_BLOCK;
	cuda_Loss_LC <<< blocks + 1, THREADS_PER_BLOCK >>> (output, x, y, size);
	cudaDeviceSynchronize();
}

void loss_LC_Linear(float* output, float* x, float* y, int size)
{
	int blocks = (size) / THREADS_PER_BLOCK;
	cuda_Loss_LC <<< blocks + 1, THREADS_PER_BLOCK >>> (output, x, y, size);
	cudaDeviceSynchronize();
}

void loss_LC_Sigmoid(float* output, float* x, float* y, int size)
{
	int blocks = (size) / THREADS_PER_BLOCK;
	cuda_Loss_LC <<< blocks + 1, THREADS_PER_BLOCK >>> (output, x, y, size);
	cudaDeviceSynchronize();
}

void loss_LC_TANH(float* output, float* x, float* y, int size)
{
	int blocks = (size) / THREADS_PER_BLOCK;
	cuda_Loss_LC <<< blocks + 1, THREADS_PER_BLOCK >>> (output, x, y, size);
	cudaDeviceSynchronize();
}

void loss_LC_SOFTMAX(float* output, float* x, float* y, int size)
{
	int blocks = (size) / THREADS_PER_BLOCK;
	cuda_Loss_LC <<< blocks + 1, THREADS_PER_BLOCK >>> (output, x, y, size);
	cudaDeviceSynchronize();
}

std::function<void(float*, float*, float*, int)> loss_derivative[][5]	= { \
												{
													&loss_LC_RELU, 		\
													&loss_LC_Linear, 	\
													&loss_LC_Sigmoid, 	\
													&loss_LC_TANH, 		\
													&loss_LC_SOFTMAX, 	\
												}
												};										
/****************************************************************************************************/

void printVector(float* vec, int size)
{
	float* _out = new float[size];
	cudaMemcpy(_out, vec, sizeof(float)*size, cudaMemcpyDeviceToHost);
	for(int k = 0; k < size; k++)
	{
		printf("[%f]", _out[k]);
	}
	printf("\n==>\n");
}

float* DenseLayer::ForwardProp(float* input)
{
	dot_productTranspose(this->sum, input, this->weights, this->size, this->prevSize);
	activationFunction[this->activation_type](this->output, this->sum, this->size);

	return this->output;
}

float* DenseLayer::BackwardProp(float* delta)
{
	vector_identity(this->errors, delta, this->size);
	dot_product(this->backError, delta, this->weights, this->prevSize, this->size);
	activationDerivative[this->lastLayer->activation_type](this->lastLayer->derivative, this->lastLayer->output, this->lastLayer->size);
	hadamard_product(this->backError, this->backError, this->lastLayer->derivative, this->lastLayer->size, 1);

	// activationDerivative[this->activation_type](this->derivative, this->output, this->size);
	// hadamard_product(this->errors, delta, this->derivative, this->size, 1);
	// dot_productTranspose(this->backError, delta, this->weights, this->prevSize, this->size);

	return this->backError;
}

float* DenseLayer::UpdateParam(float* lastOutput, float learning_rate)
{
	cross_product(this->deltas, this->errors, lastOutput, this->size, this->prevSize);
	vector_scalar_multiply(this->deltas, this->deltas, learning_rate, this->size * this->prevSize);
	vector_subtract(this->weights, this->weights, this->deltas, this->size, this->prevSize);
	
	return this->output;
}

/****************************************************************************************************/

void NeuralNet_FF::BackwardProp(float* error)
{
	NeuralLayer* lyr = this->output;
	float* interm = error;
	while(lyr != NULL )
	{
		interm = lyr->BackwardProp(interm);
		lyr = lyr->lastLayer; 
	}
}

void NeuralNet_FF::UpdateParams()
{
	NeuralLayer* lyr = this->input;
	float* interm = lyr->output;
	lyr = lyr->nextLayer;
	while(lyr != NULL )
	{
		interm = lyr->UpdateParam(interm, this->learning_rate);
		lyr = lyr->nextLayer;
	}
}

float* NeuralNet_FF::ForwardProp(float* in)
{
	// All data should already be present in the GPU. These pointers hold GPU memory!
	NeuralLayer* lyr = this->input;
	float* interm = in;
	
	while(lyr != NULL )
	{
		interm = lyr->ForwardProp(interm);
		lyr = lyr->nextLayer;
	}
	return interm;
}

float* NeuralNet_FF::calcError(float* output, float* expected)
{
	loss_derivative[this->loss_function][this->output->activation_type](this->error, expected, output, this->output->size);
	return this->error;
}

/****************************************************************************************************/

void NeuralEngine::EngineSetup(float** _in, float** _out, int _samples)
{
	in_sz = nn->input->size;
	out_sz = nn->output->size;

	in_vec_orig = _in;
	out_vec_orig = _out;

	in_vec = new float*[_samples];
	out_vec = new float*[_samples];

	for (int i = 0; i < _samples; i++)
	{
		cudaMalloc(&in_vec[i], sizeof(float)*in_sz);
		cudaMalloc(&out_vec[i], sizeof(float)*out_sz);

		cudaMemcpy(in_vec[i], _in[i], sizeof(float)*in_sz, cudaMemcpyHostToDevice);
		cudaMemcpy(out_vec[i], _out[i], sizeof(float)*out_sz, cudaMemcpyHostToDevice);
	}
}

void NeuralEngine::Train(float** _in, float** _out, int _samples, int _epoch, bool _shuffle)
{
	cout << "Trainer Setting up...\n";
	EngineSetup(_in, _out, _samples);

	cout << "Training Started...\n";
	
	for (int i = 0; i < _epoch; i++)
	{
		if (_shuffle)
		{
			shuffle(&in_vec[0], &in_vec[0] + _samples, default_random_engine(i));
			shuffle(&out_vec[0], &out_vec[0] + _samples, default_random_engine(i));
		}
		for (int j = 0; j < _samples; j++)
		{
			if( j % 10000 == 0)
				std::cout<<"\rStep "<< j ;
			// printf("\nForward-->");
			float* out = nn->ForwardProp(in_vec[j]);
			// printf("\nOutput-->");
			// printVector(out, nn->output->size);
			// printf("\nExpected-->");
			// printVector(out_vec[j], nn->output->size);
			float* error = nn->calcError(out, out_vec[j]);
			// printVector(error, nn->output->size);
			// printf("\nBackward-->");
			nn->BackwardProp(error);
			// printf("\nUpdate-->");
			nn->UpdateParams();
		}
		cudaDeviceSynchronize();
		cout << "\n\nEpoch " << i << " Completed\n";
	}
}

std::tuple<float**, float*, float*, float, float> NeuralEngine::Test(float** _in, float** _out, int _samples)
{
	cout << "Testing Setting up...\n";
	EngineSetup(_in, _out, _samples);
	float* outP;
	cudaMalloc(&outP, sizeof(float) * out_sz);
	int effec = 0;

	float* preds = new float[_samples];
	float* acts = new float[_samples];
	cout << "Testing Started...\n";

	float sum_digErrors = 0;

	for (int j = 0; j < _samples; j++)
	{
		outP = nn->ForwardProp(in_vec[j]);

		float* oop = new float[out_sz];
		cudaMemcpy((void*)oop, (void*)outP, sizeof(float) * out_sz, cudaMemcpyDeviceToHost);
		
		preds[j] = oneHotDecode(oop);
		acts[j] = oneHotDecode(_out[j]);
		printVector(outP, nn->output->size);
		printf("\nExpected-->");
		printVector(out_vec[j], nn->output->size);

		std::cout << "Prediction: " << preds[j] << ", Actual: " << acts[j];
		if( preds[j] == acts[j]) 
		{
			++effec;
		}
	}
	cudaDeviceSynchronize();
	float acc = 100 * (((float)effec) / ((float)_samples));
	printf("\n\nTotal Samples = %d, Correct Results = %d\nEfficiency = %f, Mean Digit Errors: %f", _samples, effec, acc, 100*(sum_digErrors/_samples));

	return {_in, acts, preds, effec, acc};
}

void NeuralEngine::Save_Data(std::tuple<float*, float*, float*, float*, float> mtp, int _samples)
{
}