#ifndef ANN_CUH
#define ANN_CUH


/* TODO: This is a CUDA header file.
If you have any functions in your .cu file that need to be
accessed from the outside, declare them here */
void dotVectorToMatrix(unsigned int maxBlocks, unsigned int threadsPerBlock, float *ans, float *vector, float *matrix, int x_col_dim, int w_row_dim, int w_col_dim, cudaStream_t stream);
void addVectors(unsigned int maxBlocks, unsigned int threadsPerBlock, float *output, float *input, float *arr, int size, cudaStream_t stream);
void sigmoid(unsigned int maxBlocks, unsigned int threadsPerBlock, float *output, float *input, int size, cudaStream_t stream);

void calculateDeltas(unsigned int maxBlocks, 
	unsigned int threadsPerBlock, 
	float *ans, float *vector, 
	float *matrix, float *input,
	int x_col_dim, int w_row_dim, 
	int w_col_dim, cudaStream_t stream);

void dotVectorTransposeToVector(unsigned int maxBlocks, 
	unsigned int threadsPerBlock,
	float *ans, float *vector1, 
	float *vector2, float *delta_b, 
	int col_dim1, int col_dim2, cudaStream_t stream);


void updateWeights(unsigned int maxBlocks, 
	unsigned int threadsPerBlock,
	float *weights, float gamma, 
	float *delta_weights, int row_dim, 
	int col_dim, float alpha, cudaStream_t stream);

void updateBias(unsigned int maxBlocks, 
	unsigned int threadsPerBlock,
	float *bias, float gamma, 
	float *delta_bias, int size, cudaStream_t stream);

void delta(unsigned int maxBlocks, 
	unsigned int threadsPerBlock,
	float *ans, float *predicted, 
	float label, int size, cudaStream_t stream);

#endif // ANN_CUH
