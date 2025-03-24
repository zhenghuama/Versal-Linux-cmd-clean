#include <adf.h>
#include "aie_api/aie.hpp"
#include "aie_api/aie_adf.hpp"
#include "kernels.h"
// #include "../data/matB0.h"

// INT8 sizes
// 4x8x4
// 4x16x4
// 8x8x4
// 2x8x8
// 4x8x8
// 2x16x8
// 4x16x8

// Autoencoder: 
// https://github.com/mlcommons/tiny/blob/master/benchmark/training/anomaly_detection/keras_model.py
// the model based on the simple dense auto encoder 
// (128*128*128*128*8*128*128*128*128)

// Batch size = 2
// (Batch, Input, Output)

#define N 2

const int8 matB0 [128*128] = {5};
void f0(input_window_int8 * __restrict matA,	output_window_int8 * __restrict matC) {
		gemm<2,16,8, N,128,128, 0>(matA,matC, matB0);
	}
const int8 matB1 [128*128] = {10};
  void f1(input_window_int8 * __restrict matA,	output_window_int8 * __restrict matC) {
		gemm<2,16,8, N,128,128, 0>(matA,matC, matB1);
	}