
#ifndef FUNCTION_KERNELS_H
#define FUNCTION_KERNELS_H

  template <int M_API, int K_API, int N_API, int single_M, int single_K, int single_N, int SHIFT>
	void gemm(input_window_int8 * __restrict matB, output_window_int32 * __restrict matC);

#endif
