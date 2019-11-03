
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
#include <algorithm>
#include <random>

#ifdef MNIST_DIGIT_RECOG
	#include "idx.h"
#endif

#include "Neuron.h"

#include "threads.h"
//#include "dot_adder.h"

using namespace std;
//using namespace thrust;

idx_img* imgs, *imgs2;
idx_labels* lbl, *lbl2;

double Func(double x)
{
	return 2;
}

std::tuple<float**, float**, float**, float**, int, int> productDataset_MNIST()
{
	int n = 60000;
	float** ins_train = (float**)new float*[n];
	float** outs_train = (float**)new float*[n];
	
	int m = 10000;
	float** ins_test = new float*[m];
	float** outs_test = new float*[m];

	/*let the dataset's function be even numbers from 1 to 4096*/

	idx_img* imgs, *imgs2;
	idx_labels* lbl, *lbl2;

	lbl = new idx_labels("../digits/trainlabel.bin");
	imgs = new idx_img("../digits/trainimg.bin", 60000);

	for(int i = 0; i < n; i++)
	{
		float* tmp = oneHotEncode(lbl->labels.values[i]);
		ins_train[i] = normalizeArray(imgs->imgs[i].values, 28*28);
		outs_train[i] = new float[10];
		for(int j = 0; j < 10; j++)
		{
			outs_train[i][j] = tmp[j];
		}
	}
	
	lbl2 = new idx_labels("../digits/testlabel.bin");
	imgs2 = new idx_img("../digits/testimg.bin", 10000);

	for(int i = 0; i < m; i++)
	{
		float* tmp = oneHotEncode(lbl2->labels.values[i]);
		ins_test[i] = normalizeArray(imgs2->imgs[i].values, 28*28);
		outs_test[i] = new float[10];
		for(int j = 0; j < 10; j++)
		{
			outs_test[i][j] = tmp[j];
		}
	}

	return {ins_train, outs_train, ins_test, outs_test, n, m};
}

void mathsApprox()
{
	int epoch = 10;

	float** ii, **oo, **iit, **oot;
	int tot_samples_train, tot_samples_test;

	std::tie(ii, oo, iit, oot, tot_samples_train, tot_samples_test) = productDataset_MNIST();

	auto inputLayer = new InputLayer(28*28, RELU);
	auto x = new DenseLayer(200, RELU, inputLayer);
	x = new DenseLayer(50, TANH, x);
	auto outputLayer = new DenseLayer(10, SOFTMAX, x);

	NeuralNet_FF* nn = new NeuralNet_FF(inputLayer, outputLayer, 0.01, LOSS_CROSSENTROPY, 0.01, 0.01);
	NeuralEngine engine(nn);

	for(int i = 1; i <= epoch; i++)
	{
		engine.Train(ii, oo, tot_samples_train, 1, true);
		engine.Test(iit, oot, tot_samples_test);
		printf("\nValidation Completed! epoch %d ", i);
	}
}

void test_function();

int main()
{
	//cudaSetDevice(0);

	//digit_recog();
	mathsApprox();
	// test_function();
	return 0;
}

void test_function()
{
	int epoch = 1;

	float** ii, **oo, **iit, **oot;
	int tot_samples_train, tot_samples_test;

	float test[4] = {1,2,3,4};
	float testout[3] = {0, 1, 0};

	ii = new float*[1];
	oo = new float*[1];
	iit = new float*[1];
	oot = new float*[1];

	ii[0] = test;
	oo[0] = testout;
	iit[0] = test;
	oot[0] = testout;
	tot_samples_train = 1;
	tot_samples_test = 1;

	auto inputLayer = new InputLayer(4, LINEAR);
	auto x = new DenseLayer(5, LINEAR, inputLayer);
	auto y = new DenseLayer(5, LINEAR, x);
	auto outputLayer = new DenseLayer(3, LINEAR, y);

	float tw1[] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20};
	float tw2[] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25};
	float tw3[] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
	
	x->setWeights(tw1);
	y->setWeights(tw2);
	outputLayer->setWeights(tw3);

	NeuralNet_FF* nn = new NeuralNet_FF(inputLayer, outputLayer, 0.01, LOSS_CROSSENTROPY, 0.01, 0.01);
	NeuralEngine engine(nn);

	engine.Train(ii, oo, tot_samples_train, epoch);
	// engine.Test(iit, oot, tot_samples_test);
}
