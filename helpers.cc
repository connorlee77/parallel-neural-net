#include "./headers/helpers.h"
#include <cstdio>
#include <cstdlib>
#define _USE_MATH_DEFINES
#include <cmath>
#include <vector>
#include <fstream>
#include <assert.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <minmax.h>

#define NOMINMAX
#include <windows.h>
#include <string>
#include <random>


void dotVectorToMatrix(float *ans, float *vector, float **matrix, int x_col_dim, int w_row_dim, int w_col_dim) {

	assert(x_col_dim == w_row_dim);

	for (int row = 0; row < w_row_dim; row++) {

		float x_elem = vector[row];

		for (int col = 0; col < w_col_dim; col++) {
			ans[col] += x_elem * matrix[row][col];
		}
	}
}

void dotVectorToMatrixTranspose(float *ans, float *vector, float **matrix, int x_col_dim, int w_row_dim, int w_col_dim) {

	for (int row = 0; row < w_row_dim; row++) {

		for (int col = 0; col < w_col_dim; col++) {
			ans[row] += vector[col] * matrix[row][col];
		}
	}
}

void dotVectorTransposeToVector(float **ans, float *vector1, float *vector2, int col_dim1, int col_dim2) {

	for (int i1 = 0; i1 < col_dim1; i1++) {
		for (int i2 = 0; i2 < col_dim2; i2++) {
			ans[i1][i2] = vector1[i1] * vector2[i2];
		}
	}
}

void hadamardVector(float *ans, float *vector, int size) {

	for (int i = 0; i < size; i++) {
		ans[i] *= vector[i];
	}
}

void updateBias(float *bias, float gamma, float *delta_bias, int size) {

	for(int i = 0; i < size; i++) {
		bias[i] -= gamma * delta_bias[i];
	}
}

void updateWeights(float **weights, float gamma, float **delta_weights, int row_dim, int col_dim, float alpha) {
	for (int row = 0; row < row_dim; row++) {
		for (int col = 0; col < col_dim; col++) {
			weights[row][col] -= gamma * delta_weights[row][col] - alpha * weights[row][col]; 
		}
	}
}

void addVectors(float *ans, float *arr, int size) {

	for (int i = 0; i < size; i++) {
		ans[i] += arr[i];
	}
}

void sigmoid(float *ans, int size) {

	for (int i = 0; i < size; i++) {
		ans[i] = 1.0 / (1.0 + exp(-ans[i])); 
	}

}

void sigmoid_dx(float *ans, int size) {

	for (int i = 0; i < size; i++) {
		float temp = 1.0 / (1.0 + exp(-ans[i]));
		ans[i] = (1 - temp) * temp;
	}
}

void delta(float *ans, float *predicted, float label, int size) {

	std::memcpy(ans, predicted, sizeof(float) * size);
	ans[(int)label] -= 1.0;

}