struct layer;

struct layer {
	int isinput;
	int neurons;
	float *weights;
	float *biases;
	float *outputs;
	float *deltas;
	struct layer *prev;
	struct layer *next;
};

float mac(float prev_outputs[1024],float current_weights[1024],int i,int prev_neurons) {
	//#pragma HLS INTERFACE ap_memory port=prev_outputs
	//#pragma HLS INTERFACE ap_memory port=current_weights
	#pragma HLS array_partition variable=prev_outputs cyclic factor=32
	#pragma HLS array_partition variable=current_weights cyclic factor=32

	float sum=0.f,prods[32],sums[32];
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

void forward_pass (struct layer *mlp) {
	// start with second layer, since the first layer is the input layer and there's
	// nothing to do there
	struct layer *current_layer=mlp;
	float sum;

	while (current_layer=current_layer->next) {
		// matrix-vector multiply
		for (int i=0;i<current_layer->neurons;i++) {
			sum = mac(current_layer->prev->outputs,current_layer->weights,i,current_layer->prev->neurons);
			current_layer->outputs[i]=sum+current_layer->biases[i];
		}
	}
}
