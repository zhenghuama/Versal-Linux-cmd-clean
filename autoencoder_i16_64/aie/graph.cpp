
#include <adf.h>
#include "autoenc.h"
#include <vector>

#define N_LAYERS 9

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
		layers[4] = kernel::create(f4);
		layers[5] = kernel::create(f5);
		layers[6] = kernel::create(f6);
		layers[7] = kernel::create(f7);
		layers[8] = kernel::create(f8);

	  // Single kernel connections
	  connect< window<2*128*1> >  (A        .out[0], layers[0].in[0]);
	  connect< window<2*128*1> >  (layers[0].out[0], layers[1].in[0]);
	  connect< window<2*128*1> >  (layers[1].out[0], layers[2].in[0]);
	  connect< window<2*128*1> >  (layers[2].out[0], layers[3].in[0]);
	  connect< window<2*  8*1> >  (layers[3].out[0], layers[4].in[0]);
	  connect< window<2*  8*1> >  (layers[4].out[0], layers[5].in[0]);
	  connect< window<2*128*1> >  (layers[5].out[0], layers[6].in[0]);
	  connect< window<2*128*1> >  (layers[6].out[0], layers[7].in[0]);
	  connect< window<2*128*1> >  (layers[7].out[0], layers[8].in[0]);
	  connect< window<2*128*1> >  (layers[8].out[0], C        .in[0]);

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
