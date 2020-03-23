#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define	HISTORY_LENGTH	10
#define HIDDEN_SIZE		10
#define TRAINING_WINDOW	40
#define SAMPLE_RATE		5000.f
#define SUBSAMPLE		200.f

struct params {
	float sample_rate;
	float time;
	float *freqs;
	float *phases;
};

typedef struct params * PARAMS;

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

void generate_synthetic_data (float **signal,PARAMS myparams) {
	// generate baseline time array
	int points = (int)ceilf(myparams->sample_rate * myparams->time);
	float sample_period = myparams->time / myparams->sample_rate;
	
	*signal = (float *)malloc(sizeof(float) * points);

	for (int i=0;i<points;i++) {
		(*signal)[i]=0.f;
		for (int j=0;myparams->freqs[j]!=0.f;j++)
			(*signal)[i] += sinf(2.f*M_PI*myparams->freqs[j]+myparams->phases[j]*sample_period * (float)i);
	}

		
}

void forward_pass (struct layer *mlp) {
	// start with second layer, since the first layer is the input layer and there's
	// nothing to do there
	struct layer *current_layer;

	while (current_layer=mlp->next) {
		// matrix-vector multiply
		for (int i=0;i<current_layer->neurons;i++) {
			float sum=0.f;
			for (int j=0;j<current_layer->prev->neurons;j++)
				sum+=current_layer->prev->outputs[j] * current_layer->weights[i*current_layer->prev->neurons+j];
			current_layer->outputs[i]=sum+current_layer->biases[i];
		}
	}
}

void backward_pass (struct layer *mlp,float *y) {
	// skip to last layer
	struct layer *current_layer=mlp;
	while (current_layer->next) current_layer=current_layer->next;

	// handle last layer separately
	for (int i=0;i<current_layer->neurons;i++) current_layer->deltas[i]=current_layer->outputs[i]-y[i];
	current_layer=current_layer->prev;

	while (current_layer->prev) {

		for (int i=0;i<current_layer->neurons;i++) {
			float sum=0.f;
			for (int j=0;j<current_layer->next->neurons;j++)
				sum+=current_layer->next->deltas[j]*current_layer->next->weights[j];
			current_layer->deltas[i]=current_layer->outputs[i]*sum;
		}

		current_layer=current_layer->prev;
	}
}

void update_weights (struct layer *mlp,float alpha) {
	struct layer *current_layer = mlp->next;
	
	while (current_layer) {
		for (int i=0;i<current_layer->neurons;i++) {
			float sum=0.f;
			for (int j=0;j<current_layer->prev->neurons;j++)
				current_layer->weights[i*current_layer->prev->neurons+j] -=
					alpha * current_layer->deltas[i] * current_layer->prev->outputs[j];

			current_layer->biases[i] -= alpha * current_layer->deltas[i];
		}
		current_layer = current_layer->next;
	}
}

void subsample (float *out,float *in,int len,float subsample_rate) {
	for (int i=0;i<(int)ceilf(len/subsample_rate);i++) {
		float position = (float)i * subsample_rate;
		float position_frac = position - floorf(position);
		int position_int = (int)floorf(position);
		out[i] = (1.f - position_frac)*in[position_int] + position_frac*in[position_int+1];
	}
}

int main () {
	// set up signal parameters
	PARAMS myparams = (PARAMS)malloc(sizeof(struct params));
	myparams->freqs = (float*)malloc(sizeof(float)*4);
	myparams->freqs[0]=10;
	myparams->freqs[1]=37;
	myparams->freqs[2]=78;
	myparams->freqs[3]=0;
	myparams->phases = (float*)malloc(sizeof(float)*4);
	myparams->phases[0]=0;
	myparams->phases[1]=1;
	myparams->phases[2]=2;
	myparams->phases[3]=0;
	myparams->time = 2.f;
	myparams->sample_rate=SAMPLE_RATE;

	float *signal;
	generate_synthetic_data (&signal,myparams);

	// set up MLP
	struct layer layer1,layer2,layer3;
	layer1.isinput=1;
	layer1.neurons=HISTORY_LENGTH;
	layer1.prev=0;
	layer2.outputs=(float*)malloc(sizeof(float)*HISTORY_LENGTH);
	layer1.next=&layer2;

	layer2.isinput=0;
	layer2.neurons=HIDDEN_SIZE;
	layer2.prev=&layer1;
	layer2.next=&layer3;
	layer2.weights=(float*)malloc(sizeof(float)*HIDDEN_SIZE*HISTORY_LENGTH);
	for (int i=0;i<HIDDEN_SIZE*HISTORY_LENGTH;i++) layer2.weights[i]=(float)rand()/RAND_MAX;
	layer2.biases=(float*)malloc(sizeof(float)*HIDDEN_SIZE);
	for (int i=0;i<HIDDEN_SIZE;i++) layer2.biases[i]=0.f;
	layer2.outputs=(float*)malloc(sizeof(float)*HIDDEN_SIZE);
	layer2.deltas=(float*)malloc(sizeof(float)*HIDDEN_SIZE);

	layer3.isinput=1;
	layer3.neurons=1;
	layer3.prev=&layer2;
	layer3.next=0;
	layer3.weights=(float*)malloc(sizeof(float)*HIDDEN_SIZE);
	layer3.outputs=(float*)malloc(sizeof(float));
	layer3.deltas=(float *)malloc(sizeof(float));
	layer3.biases=(float *)malloc(sizeof(float));
	layer3.biases[0]=0.f;

	

	// clean up
	free(myparams->freqs);
	free(myparams->phases);
	free(myparams);
	free(signal);
}

