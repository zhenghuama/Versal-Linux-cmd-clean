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

// Jettagger: 
// https://github.com/fastmachinelearning/fastml-science/tree/main/jet-classify

// Batch size = 2
// (Batch, Input, Output)

// 0: 16,64
// 1: 64,32
// 2: 32,32
// 3: 32, 5

#define N 2

#define DENSE_FN(IDX, API_M, API_K, API_N, S_M, S_K, S_N, WEIGHTS)      \
  const int8_t matB##IDX [S_K * S_N] = WEIGHTS;                         \
  void f##IDX(  input_window_int8  * __restrict matA,                   \
                output_window_int8 * __restrict matC) {                 \
    gemm<API_M, API_K, API_N, S_M, S_K, S_N, 0>(matA, matC, matB##IDX); \
  }

DENSE_FN(0, 2,16,8, N, 16, 64, {5})
DENSE_FN(1, 2,16,8, N, 64, 32, {5})
DENSE_FN(2, 2,16,8, N, 32, 32, {5})
DENSE_FN(3, 2,16,8, N, 32,  8, {5})

#undef DENSE_FN