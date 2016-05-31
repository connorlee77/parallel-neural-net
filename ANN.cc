#include <cstdio>
#include <cstdlib>
#define _USE_MATH_DEFINES
#include <cmath>
#include <vector>
#include <fstream>
#include <assert.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <curand.h>
#include <algorithm>
#include <time.h>

#include "./headers/ANN.cuh"
#include "./headers/ANN.h"
#include "./headers/parse.h"
#include "./headers/helpers.h"


#define NOMINMAX
#include <windows.h>
#include <string>
#include <random>

#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(
    cudaError_t code,
    const char *file,
    int line,
    bool abort = true)
{
	if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %s %d\n",
		        cudaGetErrorString(code), file, line);
		exit(code);
	}
}

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { printf("Error at %s:%d\n",__FILE__,__LINE__); exit( EXIT_FAILURE);}} while(0)
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { printf("Error at %s:%d\n",__FILE__,__LINE__); exit( EXIT_FAILURE);}} while(0)

unsigned int blocks = 512;
unsigned int threads = 200;

/**
	 * @param shape					array 	[input, hidden ... , output]
	 * @param layer_count			int 	layer count
	 * @param cost_function 		string	"quadratic, entropy"
	 * @param activation_function	string 	"ReLu, softmax, tanh"
	 * @param regularization		int 	"0 - None, 1 - L1, 2 - L2"
	 */
ANN::ANN(int *shape, int layer_count) {

	this->shape = shape;
	this->layer_count = layer_count;

	this->weights = new float*[layer_count];
	this->bias = new float*[layer_count];
	this->delta_weights = new float*[layer_count];
	this->delta_bias = new float*[layer_count];


	initWeights(this->weights, this->bias, this->delta_weights, this->delta_bias, layer_count, shape);

	this->deltas = new float*[this->layer_count];
	this->input_layers = new float*[layer_count];
	this->output_layers = new float*[layer_count];
	initLayers(this->input_layers, this->output_layers, this->deltas, layer_count, shape);
}

ANN::~ANN() {


	for (int layer = 0; layer < this->layer_count - 1; layer++) {
		cudaFree(this->weights[layer]);
		cudaFree(this->delta_weights[layer]);
	}

	delete [] this->weights;
	delete [] this->delta_weights;


	for (int layer = 0; layer < layer_count - 1; layer++) {
		cudaFree(this->bias[layer]);
		cudaFree(this->delta_bias[layer]);
		cudaFree(this->input_layers[layer]);
		cudaFree(this->output_layers[layer]);
		cudaFree(this->deltas[layer]);
	}

	cudaFree(input_layers[this->layer_count - 1]);
	cudaFree(output_layers[this->layer_count - 1]);
	cudaFree(this->deltas[this->layer_count - 1]);

	delete [] this->bias;
	delete [] this->delta_bias;

	delete [] this->input_layers;
	delete [] this->output_layers;
}


void ANN::initLayers(float **input_layers, float **output_layers, float **deltas, int layer_count, int *shape) {

	for (int layer = 0; layer < layer_count; layer++) {

		int layer_shape = shape[layer];

		gpuErrChk( cudaMalloc((void **) &deltas[layer], layer_shape * sizeof(float)) );
		gpuErrChk( cudaMalloc((void **) &input_layers[layer], layer_shape * sizeof(float)) );
		gpuErrChk( cudaMalloc((void **) &output_layers[layer], layer_shape * sizeof(float)) );
	}
}

void printDevice(float *dev, int size) {
	float *test = (float *) malloc(sizeof(float) * size);
	gpuErrChk( cudaMemcpy(test, dev, size * sizeof(float), cudaMemcpyDeviceToHost) );
	for (int i = 0; i < size; i++) {
		printf("%.3f ", test[i]);
	}
	printf("\n");
	free(test);
}

void ANN::initWeights(float **weights, float **bias, float **delta_w, float **delta_b, int layer_count, int *shape) {

	curandGenerator_t gen;
	CURAND_CALL( curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) );
	CURAND_CALL( curandSetPseudoRandomGeneratorSeed(gen, 1234ULL) );

	// Fill layer weights
	for (int layer = 0; layer < layer_count - 1; layer++) {

		int curr_size = shape[layer];
		int next_size = shape[layer + 1];

		gpuErrChk( cudaMalloc((void **) &delta_w[layer], curr_size * next_size * sizeof(float)) );
		gpuErrChk( cudaMalloc((void **) &weights[layer], curr_size * next_size * sizeof(float)) );

		float std = 1.0 / sqrt(next_size);
		CURAND_CALL( curandGenerateNormal(gen, weights[layer], curr_size * next_size * sizeof(float), 0.0, std) );

	}


	// Fill bias weights
	for (int layer = 0; layer < layer_count - 1; layer++) {

		int layer_size = shape[layer + 1];

		gpuErrChk( cudaMalloc((void **) &bias[layer], layer_size * sizeof(float)) );
		gpuErrChk( cudaMalloc((void **) &delta_b[layer], layer_size * sizeof(float)) );

		CURAND_CALL( curandGenerateNormal(gen, bias[layer], layer_size * sizeof(float), 0.0, 1) );

	}

	CURAND_CALL(curandDestroyGenerator(gen));
}

float* ANN::feedforward(float *input_data, int size) {

	float **input_layer = this->input_layers;
	float **output_layer = this->output_layers;

	gpuErrChk( cudaMemcpy(input_layer[0], input_data, sizeof(float) * size, cudaMemcpyHostToDevice) );
	gpuErrChk( cudaMemcpy(output_layer[0], input_data, sizeof(float) * size, cudaMemcpyHostToDevice) );

	for (int i = 1; i < this->layer_count; i++) {

		// Compute x * w + b
		gpuErrChk( cudaMemset(input_layer[i], 0, sizeof(float) * (this->shape[i])) );

		dotVectorToMatrix(blocks, threads, input_layer[i], output_layer[i - 1], this->weights[i - 1], this->shape[i - 1], this->shape[i - 1], this->shape[i]);
		addVectors(blocks, threads, output_layer[i], input_layer[i], this->bias[i - 1], this->shape[i]);
	}

	return output_layer[this->layer_count - 1];
}


void ANN::backpropogate(float label) {

	int last_layer = this->layer_count - 1;

	// Calculate last layer delta
	delta(this->deltas[last_layer], this->output_layers[last_layer], label, this->shape[last_layer]);

	for (int i = last_layer - 1; i >= 0; i--) {

		calculateDeltas(blocks, threads,
		                this->deltas[i], this->deltas[i + 1],
		                this->weights[i], this->input_layers[i],
		                this->shape[i + 1], this->shape[i],
		                this->shape[i + 1]);
	}

	for (int i = last_layer; i > 0; i--) {
		dotVectorTransposeToVector(blocks, threads, this->delta_weights[i - 1], this->output_layers[i - 1], this->deltas[i], this->shape[i - 1], this->shape[i]);
		gpuErrChk( cudaMemcpy(this->delta_bias[i - 1], this->deltas[i], sizeof(float) * (this->shape[i]), cudaMemcpyDeviceToDevice) );
	}
}

void ANN::sgd(float **training_data, float* training_labels, float **testing_data, float* testing_labels, int size, float gamma, float alpha, int epochs, int input_data_size, int test_size) {

	for (int epoch = 0; epoch < epochs; epoch++) {
		
		for (int i = 0; i < size; i++) {
			
			float *image = training_data[i];
			float label = training_labels[i];

			this->feedforward(image, input_data_size);
			this->backpropogate(label);			
			
			for (int layer = 0; layer < this->layer_count - 1; layer++) {
				updateBias(blocks, threads, this->bias[layer], gamma, this->delta_bias[layer], this->shape[layer + 1]);
				updateWeights(blocks, threads, this->weights[layer], gamma, this->delta_weights[layer], this->shape[layer], this->shape[layer + 1], alpha);

				
			}

			
			if (i % 10000 == 0)
				printf("%d\n", i);
		}

		int correct = this->evaluate(testing_data, testing_labels, test_size, input_data_size);
		gamma *= 0.99;
		alpha *= 0.99;
		printf("Epoch %d: %.3f\n", epoch, (float) correct / (float) test_size);
	}

}

int ANN::evaluate(float **testing_data, float *testing_labels, int size, int input_data_size) {
	int sum = 0;
	float *prediction = (float *) malloc(sizeof(float) * 10);
	for (int i = 0; i < size; i++) {
		float *image = testing_data[i];
		float label = testing_labels[i];

		float *temp = this->feedforward(image, input_data_size);
		gpuErrChk( cudaMemcpy(prediction, temp, sizeof(float) * 10, cudaMemcpyDeviceToHost) )
		float max = 0.0;
		int argmax = 0;
		for (int k = 0; k < 10; k++) {
			//printf("%0.3f, ", prediction[k]);
			if (prediction[k] > max) {
				max = prediction[k];
				argmax = k;
			}
		}
		//printf("\n\n");
		sum += ((int) label - argmax == 0);
	}

	free(prediction);

	return sum;
}

int main(int argc, char const *argv[]) {
	float* trainLabeldata = read_mnist_labels(PATH + trainLabels, 60000);
	float** trainImagedata = read_mnist_images(PATH + trainImages, 60000, 784);
	float* testLabeldata = read_mnist_labels(PATH + testLabels, 10000);
	float** testImagedata = read_mnist_images(PATH + testImages, 10000, 784);

	int size = 3;
	int *shape = new int[size];
	shape[0] = 784;
	shape[1] = 300;
	shape[2] = 10;


	ANN *ann = new ANN(shape, size);
	ann->sgd(trainImagedata, trainLabeldata, testImagedata, testLabeldata, 60000, 0.05, 1e-10, 10, 784, 10000);

	// delete ann;
	// delete [] shape;

	return 0;
}

