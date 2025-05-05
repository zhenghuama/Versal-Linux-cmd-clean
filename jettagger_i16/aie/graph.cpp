
#include <adf.h>
#include "model.h"
#include <vector>

#define N_LAYERS 4

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
		layers[0] = kernel::create(f0);
		layers[1] = kernel::create(f1);
		layers[2] = kernel::create(f2);
		layers[3] = kernel::create(f3);

	  // Single kernel connections
	  connect< window<2* 16*2> >  (A        .out[0], layers[0].in[0]);
	  connect< window<2* 64*2> >  (layers[0].out[0], layers[1].in[0]);
	  connect< window<2* 32*2> >  (layers[1].out[0], layers[2].in[0]);
	  connect< window<2* 32*2> >  (layers[2].out[0], layers[3].in[0]);
	  connect< window<2*  8*2> >  (layers[3].out[0], C        .in[0]);

	  // direct the source file of kernels
		for (int i=0; i<N_LAYERS; i++) {
			source(layers[i]) = "model.cc";
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
