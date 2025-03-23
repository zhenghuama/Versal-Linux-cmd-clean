
#include <adf.h>
#include "autoenc.h"
#include <vector>

#define mult_X 1
#define mult_Y 1 // Has to be 4 for pattern 1, 3 for pattern 2
#define mult_Z 1
#define single_M 8
#define single_K 16
#define single_N 8

using namespace adf;

class simpleGraph : public adf::graph {
private:
  kernel mat_mul_k[mult_Y * mult_X * mult_Z];

public:
  input_plio  A[mult_X * mult_Y];
  output_plio C[mult_X * mult_Z];

  simpleGraph(){

	  for (int i = 0; i < mult_X * mult_Y; i++){
		  A[i] = input_plio::create(plio_128_bits, "data/matA" + std::to_string(i) + ".txt");
	  }

	  for (int i = 0; i < mult_X * mult_Z; i++){
		  C[i] = output_plio::create(plio_128_bits, "data/matC" + std::to_string(i) + ".txt");
	  }

	  // kernels creation
	  for (int i = 0; i < mult_Y * mult_X * mult_Z; i++){
		  mat_mul_k[i] = kernel::create(gemm_4x8x4_8x16x8_0);
	  }

	  // Single kernel connections
	  connect< window<single_K*single_N*1> >  (A[0].out[0], mat_mul_k[0].in[0]);
	  connect< window<single_M*single_N*4> >  (mat_mul_k[0].out[0], C[0].in[0]);

	  // direct the source file of kernels
	  for (int i = 0; i < mult_Y * mult_X * mult_Z; i++){
		  source(mat_mul_k[i]) = "autoenc.cc";
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
