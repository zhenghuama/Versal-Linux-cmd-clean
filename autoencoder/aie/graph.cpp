
#include <adf.h>
#include "autoenc.h"
#include <vector>

#define N_LAYERS 2

using namespace adf;

class simpleGraph : public adf::graph {
private:
  kernel layers [N_LAYERS];

public:
  input_plio  A;
  output_plio C;

  simpleGraph(){

		A = input_plio::create(plio_128_bits, "data/matA0.txt");
		C = output_plio::create(plio_128_bits, "data/matC0.txt");

	  // kernels creation
		layers[0] = kernel::create(gemm_4x8x4_8x16x8_0);
		layers[1] = kernel::create(gemm_4x8x4_8x8x16_0);

	  // Single kernel connections
	  connect< window<8*16*1> >  (A        .out[0], layers[0].in[0]);
	  connect< window<8*8 *1> >  (layers[0].out[0], layers[1].in[0]);
	  connect< window<8*16*1> >  (layers[1].out[0], C        .in[0]);

	  // direct the source file of kernels
		for (int i=0; i<N_LAYERS; i++) {
			source(layers[i]) = "autoenc.cc";
			runtime<ratio>(layers[i]) = 1.0;
		}
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
