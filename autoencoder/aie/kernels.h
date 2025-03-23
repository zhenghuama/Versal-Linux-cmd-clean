
#ifndef FUNCTION_KERNELS_H
#define FUNCTION_KERNELS_H

	void gemm(
		input_window_int8 * __restrict matA,
		output_window_int32 * __restrict matC);
	
	
	void gemm_4x8x4_8x16x8_0(
		input_window_int8 * __restrict matA,
		output_window_int32 * __restrict matC);

	void vectorized_add(input_window_int32 * __restrict in_1, input_window_int32 * __restrict in_2,
							output_window_int32 * __restrict out);


#endif
