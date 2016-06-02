#include <cstdio>
#include <cuda_runtime.h>
#include "./headers/ANN.cuh"


/**
 *  Feedforward Kernels
 */

__global__
void callDotVectorToMatrix(float *ans, float *vector, float *matrix, int w_row_dim, int w_col_dim) {

	int i = threadIdx.x + blockDim.x * blockIdx.x;

	while(i < w_col_dim) {

		float sum = 0.0;
		for(int row = 0; row < w_row_dim; row++) {
			sum += vector[row] * matrix[row * w_col_dim + i];
		}

		ans[i] = sum;

		i += blockDim.x * gridDim.x;
	}
}



void dotVectorToMatrix(unsigned int maxBlocks, unsigned int threadsPerBlock, float *ans, float *vector, float *matrix, int x_col_dim, int w_row_dim, int w_col_dim, cudaStream_t stream) {

	callDotVectorToMatrix<<<maxBlocks, threadsPerBlock, 0, stream>>>(ans, vector, matrix, w_row_dim, w_col_dim);
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


void addVectors(unsigned int maxBlocks, unsigned int threadsPerBlock, float *output, float *input, float *arr, int size, cudaStream_t stream) {

	callAddVectors<<<maxBlocks, threadsPerBlock, 0, stream>>>(output, input, arr, size);
}





/**
 *  Backpropogate Kernels and helper functions
 */

__global__
void callCalculateDeltas(float *ans, float *vector, float *matrix, float *input, int w_row_dim, int w_col_dim) {

	int i = threadIdx.x + blockDim.x * blockIdx.x;

	while(i < w_row_dim) {

		// Compute delta_{i+1} * W^T
		
		float sum = 0.0;
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
	int w_row_dim, int w_col_dim, cudaStream_t stream) {

	callCalculateDeltas<<<maxBlocks, threadsPerBlock, 0, stream>>>(ans, vector, matrix, input, w_row_dim, w_col_dim);
}



__global__
void callDotVectorTransposeToVector(float *ans, float *vector1, float *vector2, float *delta_b, int col_dim1, int col_dim2) {

	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int x = threadIdx.x;
	int v1_idx, v2_idx;
	float v2_temp;

	extern __shared__ float v2[];

	while(x < col_dim2) {

		v2[x] = vector2[x];
		x += blockDim.x;
	}

	__syncthreads();

	while(i < col_dim1 * col_dim2) {

		v1_idx = (i / col_dim2) % col_dim1;
		v2_idx = i % col_dim2;
		v2_temp = v2[v2_idx];

		ans[i] = vector1[v1_idx] * v2_temp;

		if (i < col_dim2) {
			delta_b[i] = v2_temp;
		}

		i += blockDim.x * gridDim.x;
	}
}


void dotVectorTransposeToVector(unsigned int maxBlocks, 
	unsigned int threadsPerBlock,
	float *ans, float *vector1, 
	float *vector2, float *delta_b,
	int col_dim1, int col_dim2, cudaStream_t stream) {

	callDotVectorTransposeToVector<<<maxBlocks, threadsPerBlock, col_dim2 * sizeof(float), stream>>>(ans, vector1, vector2, delta_b, col_dim1, col_dim2);
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
	float *delta_bias, int size, cudaStream_t stream) {

	callUpdateBias<<<maxBlocks, threadsPerBlock, 0, stream>>>(bias, gamma, delta_bias, size);
}


__global__
void callUpdateWeights(float *weights, float gamma, float *delta_weights, int row_dim, int col_dim, float alpha) {

	int i = threadIdx.x + blockDim.x * blockIdx.x;

	while(i < row_dim * col_dim) {

		float w = weights[i];
		weights[i] = w - gamma * delta_weights[i] - alpha * w; 

		i += blockDim.x * gridDim.x;
	}
}

void updateWeights(unsigned int maxBlocks, 
	unsigned int threadsPerBlock,
	float *weights, float gamma, 
	float *delta_weights, int row_dim, 
	int col_dim, float alpha, cudaStream_t stream) {

	callUpdateWeights<<<maxBlocks, threadsPerBlock, 0, stream>>>(weights, gamma, delta_weights, row_dim, col_dim, alpha);
}




__global__
void callDelta(float *ans, float *predicted, int label, int size) {

	int i = threadIdx.x + blockDim.x * blockIdx.x;

	while(i < size) {
		
		if (i == (int) label) 
			ans[i] = predicted[i] - 1;
		else 
			ans[i] = predicted[i];

		i += gridDim.x * blockDim.x;
	}
}


void delta(unsigned int maxBlocks, 
	unsigned int threadsPerBlock,
	float *ans, float *predicted, 
	float label, int size, cudaStream_t stream) {
	
	callDelta<<<maxBlocks, threadsPerBlock, 0, stream>>>(ans, predicted, (int) label, size);
}




