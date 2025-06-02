
#include <adf.h>
#include "autoenc.h"
#include <vector>
#include <tuple>
#include "const.h"

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

class ParallelMMUlGraph : public adf::graph {
	private:
	  kernel mmuls [N_LAYERS][FACTOR];
	  kernel sums [N_LAYERS][FACTOR];
	
	public:
	  input_plio  A;
	  output_plio C;
	
	  ParallelMMUlGraph(){
			A = input_plio::create(plio_128_bits, "data/matA0.txt");
			C = output_plio::create(plio_128_bits, "data/matC0.txt");

			// kernels creation
			mmuls[0][0] = kernel::create(p00);
			location<kernel>(mmuls[0][0]) = tile(10, 0);
			mmuls[1][0] = kernel::create(p10);
			location<kernel>(mmuls[1][0]) = tile(10, 2);
			mmuls[2][0] = kernel::create(p20);
			location<kernel>(mmuls[2][0]) = tile(11, 1);
			mmuls[3][0] = kernel::create(f3); // kernel::create(p30);
			location<kernel>(mmuls[3][0]) = tile(12, 0);
			mmuls[4][0] = kernel::create(f4); // kernel::create(p40);
			location<kernel>(mmuls[4][0]) = tile(11, 0);
			mmuls[5][0] = kernel::create(f5); // kernel::create(p50);
			location<kernel>(mmuls[5][0]) = tile(11, 0);
			mmuls[6][0] = kernel::create(p60);
			location<kernel>(mmuls[6][0]) = tile(11, 2);
			mmuls[7][0] = kernel::create(p70);
			location<kernel>(mmuls[7][0]) = tile(10, 5);
			mmuls[8][0] = kernel::create(p80);
			location<kernel>(mmuls[8][0]) = tile(11, 4);

			mmuls[0][1] = kernel::create(p01);
			location<kernel>(mmuls[0][1]) = tile(9, 0);
			mmuls[1][1] = kernel::create(p11);
			location<kernel>(mmuls[1][1]) = tile(9, 1);
			mmuls[2][1] = kernel::create(p21);
			location<kernel>(mmuls[2][1]) = tile(10, 1);
			// mmuls[3][1] = kernel::create(p31);
			// mmuls[4][1] = kernel::create(p41);
			// mmuls[5][1] = kernel::create(p51);
			mmuls[6][1] = kernel::create(p61);
			location<kernel>(mmuls[6][1]) = tile(10, 3);
			mmuls[7][1] = kernel::create(p71);
			location<kernel>(mmuls[7][1]) = tile(9, 4);
			mmuls[8][1] = kernel::create(p81);
			location<kernel>(mmuls[8][1]) = tile(10, 4);

			sums[0][0] = kernel::create(s00);
			location<kernel>(sums[0][0]) = tile(10, 1);
			sums[1][0] = kernel::create(s10);
			location<kernel>(sums[1][0]) = tile(10, 1);
			sums[2][0] = kernel::create(s20);
			location<kernel>(sums[2][0]) = tile(11, 0);
			// sums[3][0] = kernel::create(s30);
			// sums[4][0] = kernel::create(s40);
			// sums[5][0] = kernel::create(s50);
			sums[6][0] = kernel::create(s60);
			location<kernel>(sums[6][0]) = tile(10, 4);
			sums[7][0] = kernel::create(s70);
			location<kernel>(sums[7][0]) = tile(10, 4);
			sums[8][0] = kernel::create(sum8); // kernel::create(s80);
			location<kernel>(sums[8][0]) = tile(11, 3);

			// sums[0][1] = kernel::create(s01);
			// sums[1][1] = kernel::create(s11);
			// sums[2][1] = kernel::create(s21); // UNUSED
			// sums[3][1] = kernel::create(s31);
			// sums[4][1] = kernel::create(s41);
			// sums[5][1] = kernel::create(s51);
			// sums[6][1] = kernel::create(s61);
			// sums[7][1] = kernel::create(s71);
			// sums[8][1] = kernel::create(s81); // UNUSED

			auto mmulUnusedKernels = std::vector<std::tuple<int,int>>{
				{3,1}, {4,1}, {5,1}
			};
			auto sumUnusedKernels = std::vector<std::tuple<int,int>>{
				{1,0}, {2,0}, {3,0}, {4,0}, {5,0}, {6,0}, {7,0},
				{1,1}, {2,1}, {3,1}, {4,1}, {5,1}, {6,1}, {7,1}, {8,1}
			};

	
			// Kernel connections
			connect< window<2*128*1> >  (A        .out[0], mmuls[0][0].in[0]);
			connect< window<2*128*1> >  (A        .out[0], mmuls[0][1].in[0]);
			for(int i = 0; i < 2; i++){
				connect< window<2*128*1> >  (mmuls[i][0].out[0], sums[i][0].in[0]);
				connect< window<2*128*1> >  (mmuls[i][0].out[0], sums[i][1].in[1]);

				connect< window<2*128*1> >  (mmuls[i][1].out[0], sums[i][0].in[1]);
				connect< window<2*128*1> >  (mmuls[i][1].out[0], sums[i][1].in[0]);

				connect< window<2*128*1> >  (sums[i][0].out[0], mmuls[i+1][0].in[0]);
				// connect< window<2*128*1> >  (sums[i][1].out[0], mmuls[i+1][1].in[0]);
			}
			connect< window<2*128*1> >  (mmuls[2][0].out[0], sums[2][0].in[0]);
			connect< window<2*128*1> >  (mmuls[2][1].out[0], sums[2][0].in[1]);
			connect< window<2*128*1> >  (sums[2][0].out[0], mmuls[3][0].in[0]);

			/*
			for(int i = 3; i < 5; i++){
				connect< window<2*  8*1> >  (mmuls[i][0].out[0], sums[i][0].in[0]);
				connect< window<2*  8*1> >  (mmuls[i][0].out[0], sums[i][1].in[1]);

				connect< window<2*  8*1> >  (mmuls[i][1].out[0], sums[i][0].in[1]);
				connect< window<2*  8*1> >  (mmuls[i][1].out[0], sums[i][1].in[0]);

				connect< window<2*  8*1> >  (sums[i][0].out[0], mmuls[i+1][0].in[0]);
				connect< window<2*  8*1> >  (sums[i][1].out[0], mmuls[i+1][1].in[0]);
			}
			*/
			connect< window<2*  8*1> >  (mmuls[3][0].out[0], mmuls[4][0].in[0]);
			connect< window<2*  8*1> >  (mmuls[4][0].out[0], mmuls[5][0].in[0]);
			connect< window<2*  128*1> >  (mmuls[5][0].out[0], mmuls[6][0].in[0]);
			connect< window<2*  128*1> >  (mmuls[5][0].out[0], mmuls[6][1].in[0]);
			
			for(int i = 6; i < 8; i++){
				connect< window<2*128*1> >  (mmuls[i][0].out[0], sums[i][0].in[0]);
				connect< window<2*128*1> >  (mmuls[i][0].out[0], sums[i][1].in[1]);

				connect< window<2*128*1> >  (mmuls[i][1].out[0], sums[i][0].in[1]);
				connect< window<2*128*1> >  (mmuls[i][1].out[0], sums[i][1].in[0]);

				connect< window<2*128*1> >  (sums[i][0].out[0], mmuls[i+1][0].in[0]);
				// connect< window<2*128*1> >  (sums[i][1].out[0], mmuls[i+1][1].in[0]);
			}
			connect< window<2*128*1> >  (mmuls[8][0].out[0], sums[8][0].in[0]);
			connect< window<2*128*1> >  (mmuls[8][1].out[0], sums[8][0].in[1]);
			connect< window<2*128*1> >  (sums[8][0].out[0], C        .in[0]);

			// direct the source file of kernels
			for (int i=0; i<N_LAYERS; i++) {
				for (int j=0; j<FACTOR; j++){
					auto t = std::tuple<int, int>{i, j};
					if(std::find(mmulUnusedKernels.begin(), mmulUnusedKernels.end(), t) == mmulUnusedKernels.end()){
						source(mmuls[i][j]) = "autoenc.cc";
						runtime<ratio>(mmuls[i][j]) = 0.8;
					}
					if(std::find(sumUnusedKernels.begin(), sumUnusedKernels.end(), t) == sumUnusedKernels.end()){
						source(sums[i][j]) = "autoenc.cc";
						runtime<ratio>(sums[i][j]) = 0.2;
					}
				}
			}
	  }
	};


using namespace adf;

// simpleGraph mygraph;
ParallelMMUlGraph myParallelGraph;

int main(void) {
	/*	
  	mygraph.init();
  	mygraph.run(10);
  	mygraph.end();
	*/
	
	myParallelGraph.init();
  	myParallelGraph.run(10);
  	myParallelGraph.end();
	
	
  	return 0;
}
