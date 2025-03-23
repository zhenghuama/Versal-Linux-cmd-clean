
#ifndef AUTOENC_H
#define AUTOENC_H
	
	void gemm_4x8x4_8x16x8_0(
		input_window_int8 * __restrict matA,
		output_window_int32 * __restrict matC);

#endif
