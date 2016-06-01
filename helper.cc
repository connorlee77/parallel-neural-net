#include "./headers/helper.h"
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


void cpu_dotVectorToMatrix(float *ans, float *vector, float *matrix, int x_col_dim, int w_row_dim, int w_col_dim) {

	assert(x_col_dim == w_row_dim);

	for (int row = 0; row < w_row_dim; row++) {

		float x_elem = vector[row];

		for (int col = 0; col < w_col_dim; col++) {
			ans[col] += x_elem * matrix[row * w_col_dim + col];
		}
	}
}

void cpu_addVectors(float *ans, float *arr, int size) {

	for (int i = 0; i < size; i++) {
		ans[i] += arr[i];
	}
}

void cpu_sigmoid(float *ans, int size) {

	for (int i = 0; i < size; i++) {
		ans[i] = 1.0 / (1.0 + exp(-ans[i])); 
	}
}