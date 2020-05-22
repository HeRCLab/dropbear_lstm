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
	float *amps;
};

typedef struct params * PARAMS;

struct signal {
	float *t;
	float *s;
	int points;
	float sample_rate;
};

typedef struct signal * SIGNAL;

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

void generate_synthetic_data (PARAMS myparams,SIGNAL mysignal) {
	// generate baseline time array
	int points = (int)ceilf(myparams->sample_rate * myparams->time);
	float sample_period = 1.f / myparams->sample_rate;
	
	// allocate time and signal arrays
	mysignal->s = (float *)malloc(sizeof(float) * points);
	mysignal->t = (float *)malloc(sizeof(float) * points);

	// synthesize
	for (int i=0;i<points;i++) {
		mysignal->s[i]=0.f;
		mysignal->t[i]=sample_period * (float)i;
		for (int j=0;myparams->freqs[j]!=0.f;j++) {
			mysignal->s[i] += myparams->amps[j]*sinf(2.f*M_PI*myparams->freqs[j]*mysignal->t[i] + myparams->phases[j]);
		}
	}

	mysignal->points = points;
	
	mysignal->sample_rate = myparams->sample_rate;
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

void subsample (SIGNAL in_signal,SIGNAL out_signal,float subsample_rate) {
	int len = in_signal->points;
	int len_new = out_signal->points = ceil(len / subsample_rate);
	
	out_signal->sample_rate = in_signal->sample_rate/subsample_rate;
	
	for (int i=0;i<len_new;i++) {
		float position = (float)i * subsample_rate;
		float position_frac = position - floorf(position);
		int position_int = (int)floorf(position);
		out_signal->s[i] = (1.f - position_frac)*
					in_signal->s[position_int] +
					position_frac*
					in_signal->s[position_int+1];
		out_signal->t[i] = out_signal->sample_rate * (float)i;
	}
}

void initialize_signal_parameters (PARAMS myparams) {
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
	myparams->amps = (float*)malloc(sizeof(float)*4);
	myparams->amps[0]=1;
	myparams->amps[1]=2;
	myparams->amps[2]=3;
	myparams->amps[3]=4;
	myparams->time = 2.f;
	myparams->sample_rate=SAMPLE_RATE;
}

void initialize_mlp (struct layer *layers) {
	layers[0].isinput=1;
	layers[0].neurons=HISTORY_LENGTH;
	layers[0].prev=0;
	layers[0].next=&layers[1];

	layers[1].outputs=(float*)malloc(sizeof(float)*HISTORY_LENGTH);
	layers[1].isinput=0;
	layers[1].neurons=HIDDEN_SIZE;
	layers[1].prev=&layers[0];
	layers[1].next=&layers[2];
	layers[1].weights=(float*)malloc(sizeof(float)*HIDDEN_SIZE*HISTORY_LENGTH);
	
	for (int i=0;i<HIDDEN_SIZE*HISTORY_LENGTH;i++) layers[1].weights[i]=(float)rand()/RAND_MAX;
	layers[1].biases=(float*)malloc(sizeof(float)*HIDDEN_SIZE);
	for (int i=0;i<HIDDEN_SIZE;i++) layers[1].biases[i]=0.f;
	layers[1].outputs=(float*)malloc(sizeof(float)*HIDDEN_SIZE);
	layers[1].deltas=(float*)malloc(sizeof(float)*HIDDEN_SIZE);

	layers[2].isinput=1;
	layers[2].neurons=1;
	layers[2].prev=&layers[1];
	layers[2].next=0;
	layers[2].weights=(float*)malloc(sizeof(float)*HIDDEN_SIZE);
	layers[2].outputs=(float*)malloc(sizeof(float));
	layers[2].deltas=(float *)malloc(sizeof(float));
	layers[2].biases=(float *)malloc(sizeof(float));
	layers[2].biases[0]=0.f;
}

void plot (SIGNAL mysignal,char *title) {
	// dump signal
	char str[4096];
	
	sprintf(str,"/usr/bin/gnuplot -p -e \""
			"set title '%s';"
			"set xlabel 'time (s)';"
			"set ylabel 'accel';"
			"set key off;"
			"plot '<cat' with lines;"
			"\"",title);
			
	FILE *myplot = popen(str,"w");
	
	if (!myplot) {
		perror("Error opening gnuplot");
		exit(1);
	}
	
	for (int i=0;i<mysignal->points;i++) {
		fprintf (myplot,"%0.4f %0.4f\n",mysignal->t[i],mysignal->s[i]);
	}
	
	fclose(myplot);
}

int main () {
	// set up signal
	PARAMS myparams = (PARAMS)malloc(sizeof(struct params));
	initialize_signal_parameters(myparams);
	
	// synthesize and plot signal
	SIGNAL mysignal = (SIGNAL)malloc(sizeof(struct signal));
	generate_synthetic_data (myparams,mysignal);
	plot(mysignal,"original signal");
	
	SIGNAL mysignal_subsampled = (SIGNAL)malloc(sizeof(struct signal));
	subsample(mysignal,mysignal_subsampled,0.25f);
	plot(mysignal_subsampled,"original signal subsampled");
	
	return 0;

	// set up MLP
	struct layer layers[3];
	initialize_mlp(layers);
	
	//plot(mysignal);	

	// clean up
	free(myparams->freqs);
	free(myparams->phases);
	free(myparams);
	free(mysignal);
	
	return 0;
}

