#include <cstdio>
#include <cstdlib>
#define _USE_MATH_DEFINES
#include <cmath>
#include <vector>
#include <fstream>
#include <assert.h>
#include <cuda_runtime.h>
#include <algorithm>

#include "./headers/ANN.cuh"
#include "./headers/ANN.h"
#include "./headers/parse.h"
#include "./headers/helpers.h"

#define NOMINMAX
#include <windows.h>
#include <string>
#include <random>

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

	this->weights = new float**[layer_count];
	this->bias = new float*[layer_count];
	this->delta_weights = new float**[layer_count];
	this->delta_bias = new float*[layer_count];
	initWeights(this->weights, this->bias, this->delta_weights, this->delta_bias, layer_count, shape);


	this->input_layers = new float*[layer_count];
	this->output_layers = new float*[layer_count];
	initLayers(this->input_layers, this->output_layers, layer_count, shape);
}

ANN::~ANN() {


	for (int layer = 0; layer < this->layer_count - 1; layer++) {
		int curr_size = this->shape[layer];
		int next_size = this->shape[layer + 1];

		for (int i = 0; i < curr_size; i++) {
			delete [] this->weights[layer][i];
			delete [] this->delta_weights[layer][i];
		}

		delete [] this->weights[layer];
		delete [] this->delta_weights[layer];
	}
	delete [] this->weights;
	delete [] this->delta_weights;


	for (int layer = 0; layer < layer_count - 1; layer++) {
		delete [] bias[layer];
		delete [] delta_bias[layer];
		delete [] input_layers[layer];
		delete [] output_layers[layer];
	}

	delete [] input_layers[this->layer_count - 1];
	delete [] output_layers[this->layer_count - 1];

	delete [] bias;
	delete [] delta_bias;

	delete [] input_layers;
	delete [] output_layers;
}



void ANN::printWeights() {
	for (int layer = 0; layer < layer_count - 1; layer++) {
		int layer_size = this->shape[layer];
		for (int c = 0; c < layer_size; c++) {
			printf("%.3f\n", this->bias[layer][c]);
		}

	}
}

void ANN::initLayers(float **input_layers, float **output_layers, int layer_count, int *shape) {

	for (int layer = 0; layer < layer_count; layer++) {

		int layer_shape = shape[layer];

		input_layers[layer] = new float[layer_shape];
		output_layers[layer] = new float[layer_shape];
	}
}

void ANN::initWeights(float ***weights, float **bias, float ***delta_w, float **delta_b, int layer_count, int *shape) {

	std::default_random_engine generator;

	// Fill layer weights
	for (int layer = 0; layer < layer_count - 1; layer++) {

		int curr_size = shape[layer];
		int next_size = shape[layer + 1];

		float std = 1.0 / sqrt(next_size);
		std::normal_distribution<float> distribution(0, std);

		float **nodes = new float*[curr_size];
		float **d_nodes = new float*[curr_size];

		for (int i = 0; i < curr_size; i++) {

			nodes[i] = new float[next_size];
			d_nodes[i] = new float[next_size];

			for (int c = 0; c < next_size; c++) {
				nodes[i][c] = distribution(generator);
			}
		}
		delta_w[layer] = d_nodes;
		weights[layer] = nodes;
	}

	// Fill bias weights
	for (int layer = 0; layer < layer_count - 1; layer++) {

		int layer_size = shape[layer + 1];
		std::normal_distribution<float> distribution(0, 1);

		bias[layer] = new float[layer_size];
		delta_b[layer] = new float[layer_size];

		for (int c = 0; c < layer_size; c++) {
			bias[layer][c] = distribution(generator);
		}

	}
}

float* ANN::feedforward(float *input_data, int size) {

	float **input_layer = this->input_layers;
	float **output_layer = this->output_layers;

	std::memcpy(input_layer[0], input_data, sizeof(float) * size);
	std::memcpy(output_layer[0], input_data, sizeof(float) * size);

	for (int i = 1; i < this->layer_count; i++) {

		// Compute x * w + b
		std::memset(input_layer[i], 0, sizeof(float) * (this->shape[i]));

		dotVectorToMatrix(input_layer[i], output_layer[i - 1], weights[i - 1], this->shape[i - 1], this->shape[i - 1], this->shape[i]);

		addVectors(input_layer[i], this->bias[i - 1], this->shape[i]);
		
		// sigma(x * w + b)
		std::memcpy(output_layer[i], input_layer[i], sizeof(float) * (this->shape[i]));
		sigmoid(output_layer[i], this->shape[i]);
	}

	return output_layer[this->layer_count - 1];
}


void ANN::backpropogate(float label) {

	float **deltas = new float*[this->layer_count];


	int last_layer = this->layer_count - 1;

	// Calculate last layer delta
	deltas[last_layer] = new float[this->shape[last_layer]];
	delta(deltas[last_layer], this->output_layers[last_layer], label, this->shape[last_layer]);

	for (int i = last_layer - 1; i >= 0; i--) {

		deltas[i] = new float[this->shape[i]];
		std::memset(deltas[i], 0, sizeof(float) * this->shape[i]);

		dotVectorToMatrixTranspose(deltas[i], deltas[i + 1], this->weights[i], this->shape[i + 1], this->shape[i], this->shape[i + 1]);

		sigmoid_dx(this->input_layers[i], this->shape[i]);

		hadamardVector(deltas[i], this->input_layers[i], this->shape[i]);
	}

	for (int i = last_layer; i > 0; i--) {

		dotVectorTransposeToVector(this->delta_weights[i - 1], this->output_layers[i - 1], deltas[i], this->shape[i - 1], this->shape[i]);

		std::memcpy(this->delta_bias[i - 1], deltas[i], sizeof(float) * this->shape[i]);
		
		delete [] deltas[i];
	}
	
	delete [] deltas[0];
	delete [] deltas;
}

void ANN::sgd(float **training_data, float* training_labels, float **testing_data, float* testing_labels, int size, float gamma, float alpha, int epochs, int input_data_size, int test_size) {

	for (int epoch = 0; epoch < epochs; epoch++) {

		for (int i = 0; i < size; i++) {
			float *image = training_data[i];
			float label = training_labels[i];

			this->feedforward(image, input_data_size);
			this->backpropogate(label);

			for (int layer = 0; layer < this->layer_count - 1; layer++) {

				updateBias(this->bias[layer], gamma, this->delta_bias[layer], this->shape[layer + 1]);

				updateWeights(this->weights[layer], gamma, this->delta_weights[layer], this->shape[layer], this->shape[layer + 1], alpha);
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

	for (int i = 0; i < size; i++) {
		float *image = testing_data[i];
		float label = testing_labels[i];

		float *prediction = this->feedforward(image, input_data_size);

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

	return sum;
}

int main(int argc, char const *argv[]) {
	float* trainLabeldata = read_mnist_labels(PATH + trainLabels, 60000);
	float** trainImagedata = read_mnist_images(PATH + trainImages, 60000, 784);
	float* testLabeldata = read_mnist_labels(PATH + testLabels, 10000);
	float** testImagedata = read_mnist_images(PATH + testImages, 10000, 784);


	int *shape = new int[5];
	shape[0] = 784;
	shape[1] = 300;
	shape[2] = 200;
	shape[3] = 100;
	shape[4] = 10;


	ANN *ann = new ANN(shape, 5);
	ann->sgd(trainImagedata, trainLabeldata, testImagedata, testLabeldata, 60000, 0.05, 1e-10, 10, 784, 10000);

	delete ann;
	delete [] shape;

	return 0;
}