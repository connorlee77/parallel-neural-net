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
#include "./headers/helper.h"

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
const int sb_count = 8;
const int batch_size = 150;
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
	int normal_size = 0;
	this->bias_index = new int[layer_count - 1];
	this->io_index = new int[layer_count];
	this->normal_io_index = new int[layer_count];
	for (int i = 0; i < layer_count; i++) {
		io_index[i] = size;
		normal_io_index[i] = normal_size;
		normal_size += shape[i];
		if (i == 0)
			size += shape[i] * batch_size * sb_count;
		else
			size += shape[i];
	}
	this->network_size = size;
	this->normal_size = normal_size;

	this->bias_size = 0;
	for (int i = 1; i < layer_count; i++) {
		this->bias_index[i - 1] = this->bias_size;
		this->bias_size += shape[i];
	}


	assert(this->bias_size == this->network_size - shape[0] * batch_size * sb_count);

	this->sum_weights = 0;
	this->weight_shapes = new int[layer_count - 1];
	this->weight_index = new int[layer_count - 1];

	for (int layer = 0; layer < layer_count - 1; layer++) {
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

	gpuErrChk( cudaMalloc((void **) & (this->deltas), (this->network_size) * sizeof(float)) );
	gpuErrChk( cudaMalloc((void **) & (this->input_layers), (this->network_size) * sizeof(float)) );
	gpuErrChk( cudaMalloc((void **) & (this->output_layers), (this->network_size) * sizeof(float)) );

	this->input_host = (float *) malloc(sizeof(float) * shape[0] * batch_size * sb_count);
	this->labels = (float *) malloc(sizeof(float) * batch_size * sb_count);

	this->eval_input = new float[this->normal_size];
	this->eval_output = new float[this->normal_size];
	this->host_weights = new float[sum_weights];
	this->host_bias = new float[this->bias_size];
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
	free(this->input_host);

	delete [] this->eval_output;
	delete [] this->eval_input;
	delete [] this->host_weights;
	delete [] this->host_bias;
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
		CURAND_CALL( curandGenerateNormal(gen, &(this->weights[index]), layer_size, 0.0, std) );
	}

	// Fill bias weights
	CURAND_CALL( curandGenerateNormal(gen, this->bias, (this->bias_size), 0.0, 1) );

	CURAND_CALL(curandDestroyGenerator(gen));
}

float* ANN::feedforward(int data_idx, cudaStream_t s) {

	float *input_layer = this->input_layers;
	float *output_layer = this->output_layers;

	for (int i = 1; i < this->layer_count; i++) {

		// Compute x * w + b
		int curr_io_index = this->io_index[i];
		int prev_io_index = this->io_index[i - 1];
		int prev_weight_index = this->weight_index[i - 1];
		int prev_bias_index = this->bias_index[i - 1];

		if (i == 1) {
			dotVectorToMatrix(blocks, threads, &input_layer[curr_io_index], &output_layer[data_idx], &this->weights[prev_weight_index], this->shape[i - 1], this->shape[i - 1], this->shape[i], s);
		} else {
			dotVectorToMatrix(blocks, threads, &input_layer[curr_io_index], &output_layer[prev_io_index], &this->weights[prev_weight_index], this->shape[i - 1], this->shape[i - 1], this->shape[i], s);
		}

		addVectors(blocks, threads, &output_layer[curr_io_index], &input_layer[curr_io_index], &this->bias[prev_bias_index], this->shape[i], s);
	}

	return &output_layer[this->io_index[this->layer_count - 1]];
}


void ANN::backpropogate(float label, int data_idx, cudaStream_t s) {

	int last_layer = this->layer_count - 1;

	// Calculate last layer delta
	int last_layer_index = this->io_index[last_layer];
	delta(blocks, threads, &this->deltas[last_layer_index],
	      &this->output_layers[last_layer_index], label,
	      this->shape[last_layer], s);

	for (int i = last_layer - 1; i >= 0; i--) {

		int next_io_index = this->io_index[i + 1];
		int curr_io_index = this->io_index[i];
		int curr_weight_index = this->weight_index[i];

		if (i == 0) {
			calculateDeltas(blocks, threads,
			                &this->deltas[curr_io_index], &this->deltas[next_io_index],
			                &this->weights[curr_weight_index], &this->input_layers[data_idx],
			                this->shape[i + 1], this->shape[i],
			                this->shape[i + 1], s);
		} else {
			calculateDeltas(blocks, threads,
			                &this->deltas[curr_io_index], &this->deltas[next_io_index],
			                &this->weights[curr_weight_index], &this->input_layers[curr_io_index],
			                this->shape[i + 1], this->shape[i],
			                this->shape[i + 1], s);
		}

	}

	for (int i = last_layer; i > 0; i--) {

		int curr_io_index = this->io_index[i];
		int prev_io_index = this->io_index[i - 1];
		int prev_weight_index = this->weight_index[i - 1];
		int prev_bias_index = this->bias_index[i - 1];

		// Calculate weight deltas and bias deltas at the same time
		if (i == 1) {
			dotVectorTransposeToVector(blocks, threads, &this->delta_w[prev_weight_index],
			                           &this->output_layers[data_idx], &this->deltas[curr_io_index], &this->delta_b[prev_bias_index],
			                           this->shape[i - 1], this->shape[i], s);
		} else {
			dotVectorTransposeToVector(blocks, threads, &this->delta_w[prev_weight_index],
			                           &this->output_layers[prev_io_index], &this->deltas[curr_io_index], &this->delta_b[prev_bias_index],
			                           this->shape[i - 1], this->shape[i], s);
		}

	}
}

void ANN::sgd(float **training_data, float* training_labels, float **testing_data, float* testing_labels, int size, float gamma, float alpha, int epochs, int input_data_size, int test_size) {

	int data_idx, label_idx, stream_idx, batch_start_idx, label_start_idx;

	float time;
	cudaEvent_t start, stop;

	cudaStream_t s[sb_count];
	for (int i = 0; i < sb_count; i++) {
		cudaStreamCreate(&s[i]);
	}


	for (int epoch = 0; epoch < epochs; epoch++) {

		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);

		for (int i = 0; i < size; i++) {

			// Get input data index
			data_idx = (i % (batch_size * sb_count)) * input_data_size;
			label_idx = i % (batch_size * sb_count);

			// Retrieve image and label
			float *image = training_data[i];
			float label = training_labels[i];

			// Fill buffers
			this->labels[label_idx] = label;
			std::memcpy(&this->input_host[data_idx], image, sizeof(float) * (this->shape[0]));

			// Start processing batch
			if (i % batch_size == batch_size - 1) {

				stream_idx = (i / batch_size) % sb_count;
				batch_start_idx = stream_idx * batch_size * input_data_size;
				label_start_idx = stream_idx * batch_size;

				gpuErrChk(
				    cudaMemcpyAsync(&input_layers[batch_start_idx],
				                    &this->input_host[batch_start_idx],
				                    batch_size * input_data_size * sizeof(float),
				                    cudaMemcpyHostToDevice, s[stream_idx])
				);

				gpuErrChk(
				    cudaMemcpyAsync(&output_layers[batch_start_idx],
				                    &this->input_host[batch_start_idx],
				                    batch_size * input_data_size * sizeof(float),
				                    cudaMemcpyHostToDevice, s[stream_idx])
				);

				// Synchronize streams
				for (int z = 0; z < sb_count; z++) {
					if (z != stream_idx) {
						cudaStreamSynchronize(s[z]);
					}
				}

				for (int x = 0; x < batch_size; x++) {

					int index = batch_start_idx + x * input_data_size;
					int l_idx = label_start_idx + x;

					this->feedforward(index, s[stream_idx]);
					this->backpropogate(this->labels[l_idx], index, s[stream_idx]);

					for (int layer = 0; layer < this->layer_count - 1; layer++) {

						int curr_bias_index = this->bias_index[layer];
						int curr_weight_index = this->weight_index[layer];

						updateBias(blocks, threads,
						           &this->bias[curr_bias_index],
						           gamma,
						           &this->delta_b[curr_bias_index],
						           this->shape[layer + 1],
						           s[stream_idx]);

						updateWeights(blocks, threads,
						              &this->weights[curr_weight_index],
						              gamma,
						              &this->delta_w[curr_weight_index],
						              this->shape[layer],
						              this->shape[layer + 1],
						              alpha,
						              s[stream_idx]);
					}
				}

			}

			if (i % 10000 == 0)
				printf("%d\n", i);
		}

		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time, start, stop);
		printf("Epoch time:  %3.1f ms \n", time);
		int correct = this->evaluate(testing_data, testing_labels, test_size, input_data_size);
		gamma *= 0.99;
		alpha *= 0.99;
		printf("Epoch %d: %.3f\n", epoch, (float) correct / (float) test_size);
	}

	for (int i = 0; i < sb_count; i++) {
        cudaStreamSynchronize(s[i]);
        cudaStreamDestroy(s[i]);
    }
}

float* ANN::cpu_feedforward(float *input_data, int size) {

	float *host_input_layer = this->eval_input;
	float *host_output_layer = this->eval_output;

	std::memcpy(host_input_layer, input_data, sizeof(float) * size);
	std::memcpy(host_output_layer, input_data, sizeof(float) * size);

	for (int i = 1; i < this->layer_count; i++) {

		int curr_io_index = this->normal_io_index[i];
		int prev_io_index = this->normal_io_index[i - 1];
		int prev_weight_index = this->weight_index[i - 1];
		int prev_bias_index = this->bias_index[i - 1];

		// Compute x * w + b
		std::memset(&host_input_layer[curr_io_index], 0, sizeof(float) * (this->shape[i]));

		cpu_dotVectorToMatrix(&host_input_layer[curr_io_index], 
			&host_output_layer[prev_io_index], 
			&this->host_weights[prev_weight_index], 
			this->shape[i - 1], this->shape[i - 1], 
			this->shape[i]);

		cpu_addVectors(&host_input_layer[curr_io_index], &this->host_bias[prev_bias_index], this->shape[i]);
		
		std::memcpy(&host_output_layer[curr_io_index], 
			&host_input_layer[curr_io_index], 
			sizeof(float) * (this->shape[i]));

		cpu_sigmoid(&host_output_layer[curr_io_index], this->shape[i]);
	}

	return &host_output_layer[this->normal_io_index[this->layer_count - 1]];
}

int ANN::evaluate(float **testing_data, float *testing_labels, int size, int input_data_size) {
	int sum = 0;

	gpuErrChk( cudaMemcpy(this->host_weights, 
		this->weights, 
		sizeof(float) * this->sum_weights, 
		cudaMemcpyDeviceToHost) );

	gpuErrChk( cudaMemcpy(this->host_bias, 
		this->bias, 
		sizeof(float) * this->bias_size, 
		cudaMemcpyDeviceToHost) );

	for (int i = 0; i < size; i++) {
		float *image = testing_data[i];
		float label = testing_labels[i];

		float *prediction = this->cpu_feedforward(image, input_data_size);

		float max = 0.0;
		int argmax = 0;
		for (int k = 0; k < 10; k++) {
			if (prediction[k] > max) {
				max = prediction[k];
				argmax = k;
			}
		}

		sum += ((int) label - argmax == 0);
	}

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

