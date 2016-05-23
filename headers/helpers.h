#ifndef HELPERS_H
#define HELPERS_H

void dotVectorToMatrix(float *ans, float *vector, float **matrix, int x_col_dim, int w_row_dim, int w_col_dim);

void dotVectorToMatrixTranspose(float *ans, float *vector, float **matrix, int x_col_dim, int w_row_dim, int w_col_dim);

void dotVectorTransposeToVector(float **ans, float *vector1, float *vector2, int col_dim1, int col_dim2);

void hadamardVector(float *ans, float *vector, int size);

void addVectors(float *ans, float *arr, int size);

void sigmoid(float *ans, int size);

void sigmoid_dx(float *ans, int size);

void delta(float *ans, float *predicted, float label, int size);

void updateBias(float *bias, float gamma, float *delta_bias, int size);
void updateWeights(float **weights, float gamma, float **delta_weights, int row_dim, int col_dim, float alpha);
#endif // HELPERS_H