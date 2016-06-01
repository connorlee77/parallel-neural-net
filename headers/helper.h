#ifndef HELPER_H
#define HELPER_H
void cpu_dotVectorToMatrix(float *ans, float *vector, float *matrix, int x_col_dim, int w_row_dim, int w_col_dim);
void cpu_addVectors(float *ans, float *arr, int size);

void cpu_sigmoid(float *ans, int size);
#endif // HELPER_H