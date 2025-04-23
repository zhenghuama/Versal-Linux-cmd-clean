#include <adf.h>
#include "aie_api/aie.hpp"
#include "aie_api/aie_adf.hpp"
#include "kernels.h"
#include "const.h"
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

// 0: 128,128
// 1: 128,128
// 2: 128,128
// 3: 128, 8
// 4: 8, 8
// 5: 8, 128
// 6: 128,128
// 7: 128,128
// 8: 128,128

#define MAT_B(IDX, S_K, S_N, WEIGHTS)\
  const int8_t matB##IDX [S_K * S_N] = WEIGHTS;                         \

  MAT_B(0, 128, 128, {5})
  MAT_B(1, 128, 128, {5})
  MAT_B(2, 128, 128, {5})
  MAT_B(3, 128, 128, {5})
  MAT_B(4, 128, 128, {5})
  MAT_B(5, 128, 128, {5})
  MAT_B(6, 128, 128, {5})
  MAT_B(7, 128, 128, {5})
  MAT_B(8, 128, 128, {5})

#define DENSE_FN(IDX, API_M, API_K, API_N, S_M, S_K, S_N, WEIGHTS)      \
  void f##IDX(  input_window_int8  * __restrict matA,                   \
                output_window_int8 * __restrict matC) {                 \
    gemm<API_M, API_K, API_N, S_M, S_K, S_N, 0>(matA, matC, matB##IDX); \
  }

DENSE_FN(0, 2,16,8, N,128,128, {5})
DENSE_FN(1, 2,16,8, N,128,128, {5})
DENSE_FN(2, 2,16,8, N,128,128, {5})
DENSE_FN(3, 2,16,8, N,128,  8, {5})
DENSE_FN(4, 2, 8,8, N,  8,  8, {5})
DENSE_FN(5, 2, 8,8, N,  8,128, {5})
DENSE_FN(6, 2,16,8, N,128,128, {5})
DENSE_FN(7, 2,16,8, N,128,128, {5})
DENSE_FN(8, 2,16,8, N,128,128, {5})

#define PARALLEL_MMUL(IDX, API_M, API_K, API_N, S_M, S_K, S_N, WEIGHTS, PART, F)      \
  void p##IDX##PART(  input_window_int8  * __restrict matA,                   \
                output_window_int8 * __restrict matC) {                 \
    partial_gemm<API_M, API_K, API_N, S_M, S_K / F, S_N, 0, PART>(matA, matC, matB##IDX); \
  }

PARALLEL_MMUL(0, 2,16,8, N,128,128, {5}, 0, FACTOR)
PARALLEL_MMUL(1, 2,16,8, N,128,128, {5}, 0, FACTOR)
PARALLEL_MMUL(2, 2,16,8, N,128,128, {5}, 0, FACTOR)
PARALLEL_MMUL(3, 2,16,8, N,128,  8, {5}, 0, FACTOR)
PARALLEL_MMUL(4, 2, 8,8, N,  8,  8, {5}, 0, FACTOR)
PARALLEL_MMUL(5, 2, 8,8, N,  8,128, {5}, 0, FACTOR)
PARALLEL_MMUL(6, 2,16,8, N,128,128, {5}, 0, FACTOR)
PARALLEL_MMUL(7, 2,16,8, N,128,128, {5}, 0, FACTOR)
PARALLEL_MMUL(8, 2,16,8, N,128,128, {5}, 0, FACTOR)

PARALLEL_MMUL(0, 2,16,8, N,128,128, {5}, 1, FACTOR)
PARALLEL_MMUL(1, 2,16,8, N,128,128, {5}, 1, FACTOR)
PARALLEL_MMUL(2, 2,16,8, N,128,128, {5}, 1, FACTOR)
PARALLEL_MMUL(3, 2,16,8, N,128,  8, {5}, 1, FACTOR)
PARALLEL_MMUL(4, 2, 8,8, N,  8,  8, {5}, 1, FACTOR)
PARALLEL_MMUL(5, 2, 8,8, N,  8,128, {5}, 1, FACTOR)
PARALLEL_MMUL(6, 2,16,8, N,128,128, {5}, 1, FACTOR)
PARALLEL_MMUL(7, 2,16,8, N,128,128, {5}, 1, FACTOR)
PARALLEL_MMUL(8, 2,16,8, N,128,128, {5}, 1, FACTOR)

#define SUM(IDX, API_N, S_M, S_N)      \
  void s##IDX(  input_window_int8  * __restrict matA,                   \
                input_window_int8  * __restrict matB,                   \
                output_window_int8 * __restrict matC) {                 \
    sum<API_N, S_M, S_N, 0>(matA, matB, matC); \
  }

  SUM(0, 16, N,128)
  SUM(1, 16, N,128)
  SUM(2, 16, N,128)
  SUM(3, 16, N,128)
  SUM(4, 8, N,8)
  SUM(5, 8, N,8)
  SUM(6, 16, N,128)
  SUM(7, 16, N,128)
  SUM(8, 16, N,128)

  #define PARALLEL_SUM(IDX, API_N, S_M, S_N, PART, F)      \
  void s##IDX##PART(  input_window_int8  * __restrict matA,                   \
                input_window_int8  * __restrict matB,                   \
                output_window_int8 * __restrict matC) {                 \
    partial_sum<API_N, S_M, S_N / F, 0, PART>(matA, matB, matC); \
  }

  PARALLEL_SUM(0, 16, N,128, 0, FACTOR)
  PARALLEL_SUM(1, 16, N,128, 0, FACTOR)
  PARALLEL_SUM(2, 16, N,128, 0, FACTOR)
  PARALLEL_SUM(3, 16, N,128, 0, FACTOR)
  PARALLEL_SUM(4, 8, N,8, 0, FACTOR)
  PARALLEL_SUM(5, 8, N,8, 0, FACTOR)
  PARALLEL_SUM(6, 16, N,128, 0, FACTOR)
  PARALLEL_SUM(7, 16, N,128, 0, FACTOR)
  PARALLEL_SUM(8, 16, N,128, 0, FACTOR)

  PARALLEL_SUM(0, 16, N,128, 1, FACTOR)
  PARALLEL_SUM(1, 16, N,128, 1, FACTOR)
  PARALLEL_SUM(2, 16, N,128, 1, FACTOR)
  PARALLEL_SUM(3, 16, N,128, 1, FACTOR)
  PARALLEL_SUM(4, 8, N,8, 1, FACTOR)
  PARALLEL_SUM(5, 8, N,8, 1, FACTOR)
  PARALLEL_SUM(6, 16, N,128, 1, FACTOR)
  PARALLEL_SUM(7, 16, N,128, 1, FACTOR)
  PARALLEL_SUM(8, 16, N,128, 1, FACTOR)

#undef MAT_B
#undef DENSE_FN
#undef PARALLEL_MMUL
#undef SUM

