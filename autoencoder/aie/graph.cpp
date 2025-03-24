
#include <adf.h>
#include "autoenc.h"
#include <vector>

#define single_M 8
#define single_K 16
#define single_N 8

using namespace adf;

class simpleGraph : public adf::graph {
private:
  kernel layer0;

public:
  input_plio  A;
  output_plio C;

  simpleGraph(){

		A = input_plio::create(plio_128_bits, "data/matA0.txt");
		C = output_plio::create(plio_128_bits, "data/matC0.txt");

	  // kernels creation
		layer0 = kernel::create(gemm_4x8x4_8x16x8_0);

	  // Single kernel connections
	  connect< window<single_K*single_N*1> >  (A.out[0], layer0.in[0]);
	  connect< window<single_M*single_N*1> >  (layer0.out[0], C.in[0]);

	  // direct the source file of kernels
		source(layer0) = "autoenc.cc";

	  runtime<ratio>(layer0) = 1.0;
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
