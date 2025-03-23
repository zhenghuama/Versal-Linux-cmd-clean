
#include <adf.h>
#include "kernels.h"
#include <vector>

#include <adf.h>
#include "kernels.h"
#include "kernels/include.h"

void gemm_4_8_4__8_16_8_0 (input_window_int8 * __restrict matB,
	output_window_int32 * __restrict matC) {
		gemm<4,8,4, 8,16,8, 0>(matB, matC);
	}

using namespace adf;

class simpleGraph : public adf::graph {
private:
  kernel mat_mul_k[mult_Y * mult_X * mult_Z];

public:
  input_plio  B[mult_Y * mult_Z];
  output_plio C[mult_X * mult_Z];

  simpleGraph(){

	  for (int i = 0; i < mult_Y * mult_Z; i++){
		  B[i] = input_plio::create(plio_128_bits, "data/matB" + std::to_string(i) + ".txt");
	  }

	  for (int i = 0; i < mult_X * mult_Z; i++){
		  C[i] = output_plio::create(plio_128_bits, "data/matC" + std::to_string(i) + ".txt");
	  }

	  // kernels creation
	  for (int i = 0; i < mult_Y * mult_X * mult_Z; i++){
		  mat_mul_k[i] = kernel::create(gemm_4_8_4__8_16_8_0);
	  }

	  // Single kernel connections
	  connect< window<single_K*single_N*1> >  (B[0].out[0], mat_mul_k[0].in[0]);
	  connect< window<single_M*single_N*4> >  (mat_mul_k[0].out[0], C[0].in[0]);

	  // direct the source file of kernels
	  for (int i = 0; i < mult_Y * mult_X * mult_Z; i++){
		  source(mat_mul_k[i]) = "kernels/kernels.cc";
	  }

	  runtime<ratio>(mat_mul_k[0]) = 1.0;
  }
};


using namespace adf;

simpleGraph mygraph;

int main(void) {
  mygraph.init();
  mygraph.run(10);
  mygraph.end();
  return 0;
}
