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

	int size = 0;
	this->bias_index = new int[layer_count - 1];
	this->io_index = new int[layer_count];
	for(int i = 0; i < layer_count; i++) {
		io_index[i] = size;	
		size += shape[i];
	}

	this->bias_size = 0;
	for(int i = 1; i < layer_count; i++) {
		this->bias_index[i - 1] = bias_size;
		bias_size += shape[i];
	}

	this->network_size = size;
	assert(this->bias_size ==this->network_size - shape[0]);

	this->sum_weights = 0;
	this->weight_shapes = new int[layer_count - 1];
	this->weight_index = new int[layer_count - 1];

	for(int layer = 0; layer < layer_count - 1; layer++) {
		int curr_size = shape[layer];
		int next_size = shape[layer + 1];
		int total_size = curr_size * next_size;
		
		this->weight_index[layer] = this->sum_weights;
		this->sum_weights += total_size;
		this->weight_shapes[layer] = total_size;
	}

	gpuErrChk( cudaMalloc((void **) &this->delta_w, sum_weights * sizeof(float)) );
	gpuErrChk( cudaMalloc((void **) &this->weights, sum_weights * sizeof(float)) );

	gpuErrChk( cudaMalloc((void **) &this->bias, this->bias_size * sizeof(float)) );
	gpuErrChk( cudaMalloc((void **) &this->delta_b, this->bias_size * sizeof(float)) );


	initWeights(layer_count);

	gpuErrChk( cudaMalloc((void **) &(this->deltas), (this->network_size) * sizeof(float)) );
	gpuErrChk( cudaMalloc((void **) &(this->input_layers), (this->network_size) * sizeof(float)) );
	gpuErrChk( cudaMalloc((void **) &(this->output_layers), (this->network_size) * sizeof(float)) );
}

ANN::~ANN() {
	gpuErrChk( cudaFree(this->delta_w) );
	gpuErrChk( cudaFree(this->delta_b) );
	gpuErrChk( cudaFree(this->weights) );
	gpuErrChk( cudaFree(this->bias) );
	gpuErrChk( cudaFree(this->deltas) );
	gpuErrChk( cudaFree(this->input_layers) );
	gpuErrChk( cudaFree(this->output_layers) );

	free(this->bias_index);
	free(this->weight_shapes);
	free(this->weight_index);
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

void ANN::initWeights(int layer_count) {

	curandGenerator_t gen;
	CURAND_CALL( curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) );
	CURAND_CALL( curandSetPseudoRandomGeneratorSeed(gen, 1234ULL) );

	// Fill layer weights
	for (int layer = 0; layer < layer_count - 1; layer++) {

		int layer_size = this->weight_shapes[layer];
		int index = this->weight_index[layer];

		float std = 1.0 / sqrt(this->shape[layer + 1]);
		CURAND_CALL( curandGenerateNormal(gen, &(this->weights[index]), layer_size * sizeof(float), 0.0, std) );
	}

	// Fill bias weights
	CURAND_CALL( curandGenerateNormal(gen, this->bias, (this->bias_size) * sizeof(float), 0.0, 1) );
	
	CURAND_CALL(curandDestroyGenerator(gen));
}

float* ANN::feedforward(float *input_data, int size) {

	float *input_layer = this->input_layers;
	float *output_layer = this->output_layers;

	gpuErrChk( cudaMemcpy(input_layer, input_data, sizeof(float) * size, cudaMemcpyHostToDevice) );
	gpuErrChk( cudaMemcpy(output_layer, input_data, sizeof(float) * size, cudaMemcpyHostToDevice) );

	for (int i = 1; i < this->layer_count; i++) {

		// Compute x * w + b
		int curr_io_index = this->io_index[i];
		int prev_io_index = this->io_index[i - 1];
		int prev_weight_index = this->weight_index[i - 1];
		int prev_bias_index = this->bias_index[i - 1];

		gpuErrChk( cudaMemset(&input_layer[curr_io_index], 0, sizeof(float) * (this->shape[i])) );

		dotVectorToMatrix(blocks, threads, &input_layer[curr_io_index], &output_layer[prev_io_index], &this->weights[prev_weight_index], this->shape[i - 1], this->shape[i - 1], this->shape[i]);
		addVectors(blocks, threads, &output_layer[curr_io_index], &input_layer[curr_io_index], &this->bias[prev_bias_index], this->shape[i]);
	}

	return &output_layer[this->io_index[this->layer_count - 1]];
}


void ANN::backpropogate(float label) {

	int last_layer = this->layer_count - 1;

	// Calculate last layer delta
	int last_layer_index = this->io_index[last_layer];
	delta(&this->deltas[last_layer_index], &this->output_layers[last_layer_index], label, this->shape[last_layer]);

	for (int i = last_layer - 1; i >= 0; i--) {

		int next_io_index = this->io_index[i + 1];
		int curr_io_index = this->io_index[i];
		int curr_weight_index = this->weight_index[i];

		calculateDeltas(blocks, threads,
		                &this->deltas[curr_io_index], &this->deltas[next_io_index],
		                &this->weights[curr_weight_index], &this->input_layers[curr_io_index],
		                this->shape[i + 1], this->shape[i],
		                this->shape[i + 1]);
	}

	for (int i = last_layer; i > 0; i--) {
		
		int curr_io_index = this->io_index[i];
		int prev_io_index = this->io_index[i - 1];
		int prev_weight_index = this->weight_index[i - 1];
		int prev_bias_index = this->bias_index[i - 1];

		dotVectorTransposeToVector(blocks, threads, &this->delta_w[prev_weight_index], &this->output_layers[prev_io_index], &this->deltas[curr_io_index], this->shape[i - 1], this->shape[i]);
		gpuErrChk( cudaMemcpy(&this->delta_b[prev_bias_index], &this->deltas[curr_io_index], sizeof(float) * (this->shape[i]), cudaMemcpyDeviceToDevice) );
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
				
				int curr_bias_index = this->bias_index[layer];
				int curr_weight_index = this->weight_index[layer];

				updateBias(blocks, threads, &this->bias[curr_bias_index], gamma, &this->delta_b[curr_bias_index], this->shape[layer + 1]);
				updateWeights(blocks, threads, &this->weights[curr_weight_index], gamma, &this->delta_w[curr_weight_index], this->shape[layer], this->shape[layer + 1], alpha);
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

	int size = 4;
	int *shape = new int[size];
	shape[0] = 784;
	shape[1] = 300;
	shape[2] = 50;
	shape[3] = 10;


	ANN *ann = new ANN(shape, size);
	ann->sgd(trainImagedata, trainLabeldata, testImagedata, testLabeldata, 60000, 0.05, 1e-10, 10, 784, 10000);

	// delete ann;
	// delete [] shape;

	return 0;
}

