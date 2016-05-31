#ifndef ANN_CUH
#define ANN_CUH


/* TODO: This is a CUDA header file.
If you have any functions in your .cu file that need to be
accessed from the outside, declare them here */
void dotVectorToMatrix(unsigned int maxBlocks, unsigned int threadsPerBlock, float *ans, float *vector, float *matrix, int x_col_dim, int w_row_dim, int w_col_dim);
void addVectors(unsigned int maxBlocks, unsigned int threadsPerBlock, float *output, float *input, float *arr, int size);
void sigmoid(unsigned int maxBlocks, unsigned int threadsPerBlock, float *output, float *input, int size);

void calculateDeltas(unsigned int maxBlocks, 
	unsigned int threadsPerBlock, 
	float *ans, float *vector, 
	float *matrix, float *input,
	int x_col_dim, int w_row_dim, 
	int w_col_dim);

void dotVectorTransposeToVector(unsigned int maxBlocks, 
	unsigned int threadsPerBlock,
	float *ans, float *vector1, 
	float *vector2, int col_dim1, 
	int col_dim2);

void dotVectorTransposeToVector(unsigned int maxBlocks, 
	unsigned int threadsPerBlock,
	float *ans, float *vector1, 
	float *vector2, int col_dim1, 
	int col_dim2); 

void updateWeights(unsigned int maxBlocks, 
	unsigned int threadsPerBlock,
	float *weights, float gamma, 
	float *delta_weights, int row_dim, 
	int col_dim, float alpha);

void updateBias(unsigned int maxBlocks, 
	unsigned int threadsPerBlock,
	float *bias, float gamma, 
	float *delta_bias, int size);


#endif // ANN_CUH
