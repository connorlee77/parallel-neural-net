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


void delta(float *ans, float *predicted, float label, int size) {

	float *temp = (float *) malloc(sizeof(float) * size);

	gpuErrChk( cudaMemcpy(temp, predicted, sizeof(float) * size, cudaMemcpyDeviceToHost) );
	temp[(int)label] -= 1.0;
	gpuErrChk( cudaMemcpy(ans, temp, sizeof(float) * size, cudaMemcpyHostToDevice) );

	free(temp);
}