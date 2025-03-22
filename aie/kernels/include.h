#ifndef FUNCTION_INCLUDES_H
#define FUNCTION_INCLUDES_H


// define shift right for output values after matrix mult
#define SHIFT 0


// multiple AIE parameters (XxYxZ on manuscript)
#define mult_X 1
#define mult_Y 1 // Has to be 4 for pattern 1, 3 for pattern 2
#define mult_Z 1


// single kernel dimensions (MxKxN on manuscript)
#define single_M 8
#define single_K 16
#define single_N 8


// AI Engine API dimensions
#define M_API 4
#define K_API 8
#define N_API 4

#endif
