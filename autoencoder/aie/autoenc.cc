#include <adf.h>
#include "aie_api/aie.hpp"
#include "aie_api/aie_adf.hpp"
#include "kernels.h"

// INT8 sizes
// 4x8x4
// 4x16x4
// 8x8x4
// 2x8x8
// 4x8x8
// 2x16x8
// 4x16x8

void gemm_4x8x4_8x16x8_0(
	input_window_int8 * __restrict matA,
	output_window_int32 * __restrict matC) {
		gemm<4,8,4, 8,16,8, 0>(matA,matC);
	}