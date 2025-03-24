
#ifndef FUNCTION_KERNELS_H
#define FUNCTION_KERNELS_H

#include <adf.h>
#include "aie_api/aie.hpp"
#include "aie_api/aie_adf.hpp"


template <int M_API, int K_API, int N_API, int single_M, int single_K, int single_N, int SHIFT>
void gemm(
	input_window_int8 * __restrict matA,
	output_window_int8 * __restrict matC, const int8 matB []) {

	using MMUL = aie::mmul<M_API, K_API, N_API, int8, int8>;

	// pointers of matrices
	const int8* __restrict pA = (int8*) matA->ptr;
	const int8* __restrict pB = (int8*) matB;
	int8* __restrict pC = (int8*) matC->ptr;

	// unroll the loops for more optimization
	for (unsigned i = 0; i < (single_M/M_API); i+=2)
//		chess_prepare_for_pipelining
		chess_flatten_loop
	{

		int8 * __restrict pC1 = pC + (i * (single_N/N_API)) * MMUL::size_C;
		int8 * __restrict pC2 = pC + ((i+1) * (single_N/N_API)) * MMUL::size_C;;

		for (unsigned j = 0; j < (single_N/N_API); j+=2)
		chess_flatten_loop
//		chess_prepare_for_pipelining
//		Just write it this way, don't question it, or it won't scheudle at every clk.
		{
			const int8 * __restrict pA1 = pA + ( i * (single_K/K_API) + 0) * MMUL::size_A;
			const int8 * __restrict pA2 = pA + ( (i+1) * (single_K/K_API) + 0) * MMUL::size_A;

			const int8 * __restrict pB1 = pB + ( 0 * (single_N/N_API) + j) * MMUL::size_B;
			const int8 * __restrict pB2 = pB + ( 0 * (single_N/N_API) + (j+1)) * MMUL::size_B;


			aie::vector<int8, MMUL::size_A> A0 = aie::load_v<MMUL::size_A>(pA1); pA1 += MMUL::size_A;
			aie::vector<int8, MMUL::size_A> A1 = aie::load_v<MMUL::size_A>(pA2); pA2 += MMUL::size_A;

			aie::vector<int8, MMUL::size_B> B0 = aie::load_v<MMUL::size_B>(pB1); pB1 += MMUL::size_B * (single_N/N_API);
			aie::vector<int8, MMUL::size_B> B1 = aie::load_v<MMUL::size_B>(pB2); pB2 += MMUL::size_B * (single_N/N_API);

			MMUL C00;
			MMUL C01;
			MMUL C10;
			MMUL C11;

			// matrix multiply by initializing to 0
			C00.mul(A0, B0);
			C01.mul(A0, B1);
			C10.mul(A1, B0);
			C11.mul(A1, B1);

			for (unsigned k = 0; k < (single_K/K_API)-1; k++)
//			chess_prepare_for_pipelining
			chess_flatten_loop
			{
				A0 = aie::load_v<MMUL::size_A>(pA1); pA1 += MMUL::size_A;
				A1 = aie::load_v<MMUL::size_A>(pA2); pA2 += MMUL::size_A;

				B0 = aie::load_v<MMUL::size_B>(pB1); pB1 += MMUL::size_B * (single_N/N_API);
				B1 = aie::load_v<MMUL::size_B>(pB2); pB2 += MMUL::size_B * (single_N/N_API);

				// matrix multiply and adding partial blocks
				C00.mac(A0, B0);
				C01.mac(A0, B1);
				C10.mac(A1, B0);
				C11.mac(A1, B1);
			}
			auto C00_i8 = C00.template to_vector<int8>(SHIFT);
      auto C01_i8 = C01.template to_vector<int8>(SHIFT);
      auto C10_i8 = C10.template to_vector<int8>(SHIFT);
      auto C11_i8 = C11.template to_vector<int8>(SHIFT);

      auto C00_relu = aie::max(C00_i8, (int8)0); 
      auto C01_relu = aie::max(C01_i8, (int8)0); 
      auto C10_relu = aie::max(C10_i8, (int8)0); 
      auto C11_relu = aie::max(C11_i8, (int8)0); 

			aie::store_v(pC1, C00_relu); pC1 +=MMUL::size_C;
			aie::store_v(pC1, C01_relu); pC1 +=MMUL::size_C;
			aie::store_v(pC2, C10_relu); pC2 +=MMUL::size_C;
			aie::store_v(pC2, C11_relu); pC2 +=MMUL::size_C;

			// aie::store_v(pC1, C00.template to_vector<int8>(SHIFT)); pC1 +=MMUL::size_C;
			// aie::store_v(pC1, C01.template to_vector<int8>(SHIFT)); pC1 +=MMUL::size_C;
			// aie::store_v(pC2, C10.template to_vector<int8>(SHIFT)); pC2 +=MMUL::size_C;
			// aie::store_v(pC2, C11.template to_vector<int8>(SHIFT)); pC2 +=MMUL::size_C;
		}
	}
}

#endif
