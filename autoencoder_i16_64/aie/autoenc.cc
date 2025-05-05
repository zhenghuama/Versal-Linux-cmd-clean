#include <adf.h>
#include "aie_api/aie.hpp"
#include "aie_api/aie_adf.hpp"
#include "kernels.h"
// #include "../data/matB0.h"

// int16 sizes
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
// (64*64*64*64*8*64*64*64*64)

// Batch size = 2
// (Batch, Input, Output)

// 0: 64,64
// 1: 64,64
// 2: 64,64
// 3: 64, 8
// 4: 8, 8
// 5: 8, 64
// 6: 64,64
// 7: 64,64
// 8: 64,64

#define N 2

#define DENSE_FN(IDX, API_M, API_K, API_N, S_M, S_K, S_N, WEIGHTS)      \
  const int16_t matB##IDX [S_K * S_N] = WEIGHTS;                         \
  void f##IDX(  input_window_int16  * __restrict matA,                   \
                output_window_int16 * __restrict matC) {                 \
    gemm<API_M, API_K, API_N, S_M, S_K, S_N, 0>(matA, matC, matB##IDX); \
  }

DENSE_FN(0, 2,4,8, N,64,64, {5})
DENSE_FN(1, 2,4,8, N,64,64, {5})
DENSE_FN(2, 2,4,8, N,64,64, {5})
DENSE_FN(3, 2,4,8, N,64,  8, {5})
DENSE_FN(4, 2,4,8, N,  8,  8, {5})
DENSE_FN(5, 2,4,8, N,  8,64, {5})
DENSE_FN(6, 2,4,8, N,64,64, {5})
DENSE_FN(7, 2,4,8, N,64,64, {5})
DENSE_FN(8, 2,4,8, N,64,64, {5})

#undef DENSE_FN