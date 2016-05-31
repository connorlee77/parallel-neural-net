#include <cstdio>
#include <cuda_runtime.h>
#include "./headers/ANN.cuh"


/**
 *  Feedforward Kernels
 */

__global__
void callDotVectorToMatrix(float *ans, float *vector, float *matrix, int w_row_dim, int w_col_dim) {

	int i = threadIdx.x + blockDim.x * blockIdx.x;

	while(i < w_row_dim) {

		float x_elem = vector[i];

		for(int col = 0; col < w_col_dim; col++) {
			float addend = x_elem * matrix[i * w_col_dim + col];
			atomicAdd(&ans[col], addend);
		}

		i += blockDim.x * gridDim.x;
	}
}



void dotVectorToMatrix(unsigned int maxBlocks, unsigned int threadsPerBlock, float *ans, float *vector, float *matrix, int x_col_dim, int w_row_dim, int w_col_dim) {

	callDotVectorToMatrix<<<maxBlocks, threadsPerBlock>>>(ans, vector, matrix, w_row_dim, w_col_dim);
}




__global__
void callAddVectors(float *output, float *input, float *arr, int size) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;

	while(i < size) {
		float sum = input[i] + arr[i];
		float sigmoid = 1.0 / (1.0 + exp(-sum)); 
		output[i] = sigmoid;

		i += blockDim.x * gridDim.x;
	}
}


void addVectors(unsigned int maxBlocks, unsigned int threadsPerBlock, float *output, float *input, float *arr, int size) {

	callAddVectors<<<maxBlocks, threadsPerBlock>>>(output, input, arr, size);
}



__global__
void callSigmoid(float *output, float *input, int size) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;

	while(i < size) {
		output[i] = 1.0 / (1.0 + exp(-input[i])); 

		i += blockDim.x * gridDim.x;
	}
}

void sigmoid(unsigned int maxBlocks, unsigned int threadsPerBlock, float *output, float *input, int size) {
	callSigmoid<<<maxBlocks, threadsPerBlock>>>(output, input, size);
}










/**
 *  Backpropogate Kernels and helper functions
 */

__global__
void callCalculateDeltas(float *ans, float *vector, float *matrix, float *input, int w_row_dim, int w_col_dim) {

	int i = threadIdx.x + blockDim.x * blockIdx.x;

	while(i < w_row_dim) {

		// Compute delta_{i+1} * W^T
		float sum = 0;
		for (int col = 0; col < w_col_dim; col++) {
			sum += vector[col] * matrix[i * w_col_dim + col];
		}

		// Compute the sigmoid of input vector.
		float temp = 1.0 / (1.0 + exp(-input[i]));
		float sigmoid_dx = (1 - temp) * temp;

		// Compute Hadamard product. 
		ans[i] = sum * sigmoid_dx;

		i += blockDim.x * gridDim.x;
	}
}

void calculateDeltas(unsigned int maxBlocks, 
	unsigned int threadsPerBlock, 
	float *ans, float *vector, 
	float *matrix, float *input, int x_col_dim, 
	int w_row_dim, int w_col_dim) {

	callCalculateDeltas<<<maxBlocks, threadsPerBlock>>>(ans, vector, matrix, input, w_row_dim, w_col_dim);
}



__global__
void callDotVectorTransposeToVector(float *ans, float *vector1, float *vector2, int col_dim1, int col_dim2) {

	int i1 = threadIdx.x + blockDim.x * blockIdx.x;

	while (i1 < col_dim1) {
		for (int i2 = 0; i2 < col_dim2; i2++) {
			ans[i1 * col_dim2 + i2] = vector1[i1] * vector2[i2];
		}

		i1 += blockDim.x * gridDim.x;
	}
}


void dotVectorTransposeToVector(unsigned int maxBlocks, 
	unsigned int threadsPerBlock,
	float *ans, float *vector1, 
	float *vector2, int col_dim1, 
	int col_dim2) {

	callDotVectorTransposeToVector<<<maxBlocks, threadsPerBlock>>>(ans, vector1, vector2, col_dim1, col_dim2);
}




/**
 *  Gradient Descent kernels
 */
__global__
void callUpdateBias(float *bias, float gamma, float *delta_bias, int size) {

	int i = threadIdx.x + blockDim.x * blockIdx.x;

	while(i < size) {
		bias[i] -= gamma * delta_bias[i];

		i += blockDim.x * gridDim.x;
	}
}



void updateBias(unsigned int maxBlocks, 
	unsigned int threadsPerBlock,
	float *bias, float gamma, 
	float *delta_bias, int size) {

	callUpdateBias<<<maxBlocks, threadsPerBlock>>>(bias, gamma, delta_bias, size);
}


__global__
void callUpdateWeights(float *weights, float gamma, float *delta_weights, int row_dim, int col_dim, float alpha) {

	int i = threadIdx.x + blockDim.x * blockIdx.x;

	while(i < row_dim) {
		for (int col = 0; col < col_dim; col++) {
			weights[i * col_dim + col] -= gamma * delta_weights[i * col_dim + col] - alpha * weights[i * col_dim + col]; 
		}

		i += blockDim.x * gridDim.x;
	}
}

void updateWeights(unsigned int maxBlocks, 
	unsigned int threadsPerBlock,
	float *weights, float gamma, 
	float *delta_weights, int row_dim, 
	int col_dim, float alpha) {

	callUpdateWeights<<<maxBlocks, threadsPerBlock>>>(weights, gamma, delta_weights, row_dim, col_dim, alpha);
}










