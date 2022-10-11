#include "rtml.h"

// disabled because of symbol collision
#if 1
void forward_pass (struct mlp* m) {
	// start with second layer, since the first layer is the input layer and there's
	// nothing to do there
	struct layer *current_layer=&(m->layers[1]);

	while (current_layer=current_layer->next) {
		// matrix-vector multiply
		for (int i=0;i<current_layer->neurons;i++) {
			double sum=0.f;
			for (int j=0;j<current_layer->prev->neurons;j++)
				sum+=current_layer->prev->outputs[j] * current_layer->weights[i*current_layer->prev->neurons+j];
			current_layer->outputs[i]=sum+current_layer->biases[i];
		}
	}
}
#endif

void backward_pass (struct mlp *m,double *y) {
	// skip to last layer
	struct layer *current_layer=&(m->layers[m->layerc-1]);

	// handle last layer separately
	for (int i=0;i<current_layer->neurons;i++) current_layer->deltas[i]=current_layer->outputs[i]-y[i];
	current_layer=current_layer->prev;

	while (current_layer->prev) {

		for (int i=0;i<current_layer->neurons;i++) {
			double sum=0.f;
			for (int j=0;j<current_layer->next->neurons;j++)
				sum+=current_layer->next->deltas[j]*current_layer->next->weights[j];
			current_layer->deltas[i]=current_layer->outputs[i]*sum;
		}

		current_layer=current_layer->prev;
	}
}

void update_weights (struct mlp *m) {
	struct layer *current_layer = &(m->layers[1]);
	double alpha = m->alpha;

	while (current_layer) {
		for (int i=0;i<current_layer->neurons;i++) {
			double sum=0.f;
			for (int j=0;j<current_layer->prev->neurons;j++)
				current_layer->weights[i*current_layer->prev->neurons+j] -=
					alpha * current_layer->deltas[i] * current_layer->prev->outputs[j];

			current_layer->biases[i] -= alpha * current_layer->deltas[i];
		}
		current_layer = current_layer->next;
	}
}

#if 0
double mac(double prev_outputs[1024],double current_weights[1024],int i,int prev_neurons) {
	//#pragma HLS INTERFACE ap_memory port=prev_outputs
	//#pragma HLS INTERFACE ap_memory port=current_weights
	#pragma HLS array_partition variable=prev_outputs cyclic factor=32
	#pragma HLS array_partition variable=current_weights cyclic factor=32

	double sum=0.f,prods[32],sums[32];
	#pragma HLS array_partition variable=prods complete
	#pragma HLS array_partition variable=sums complete

	int k;

	loop_1: for (int j=0;j<prev_neurons;j+=32) {
	#pragma HLS PIPELINE II=1

		for (k=0;k<32;k++) {
			#pragma HLS UNROLL factor=32
			prods[k]=prev_outputs[j+k] * current_weights[i*prev_neurons+j+k];
		}

		for (k=0;k<32;k++) {
			#pragma HLS UNROLL factor=32
			sums[j>>5]+=prods[k];
		}
	}

	loop_2: for (k=0;k<32;k++) {
		#pragma HLS UNROLL factor=32
		sum += sums[k];
	}

	return sum;
}

void forward_pass (struct mlp *m) {
	// start with second layer, since the first layer is the input layer and there's
	// nothing to do there
	struct layer *current_layer=&(m->layers[0]);
	double sum;

	while (current_layer=current_layer->next) {
		// matrix-vector multiply
		for (int i=0;i<current_layer->neurons;i++) {
			sum = mac(current_layer->prev->outputs,current_layer->weights,i,current_layer->prev->neurons);
			current_layer->outputs[i]=sum+current_layer->biases[i];
		}
	}
}
#endif
