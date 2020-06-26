#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <utils.h>
#include <onnx.pb-c.h>

#define	HISTORY_LENGTH		10
#define HIDDEN_SIZE		10
#define TRAINING_WINDOW		40
#define SAMPLE_RATE		5000.f
#define SUBSAMPLE_RATE		1250.f
#define PREDICTION_TIME		10

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
	struct layer *current_layer=mlp->next;

	while (current_layer=current_layer->next) {
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
	for (int i=0;i<current_layer->neurons;i++)
		current_layer->deltas[i]=current_layer->outputs[i]-y[i];

	// move back one layer
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
			for (int j=0;j<current_layer->prev->neurons;j++)
				current_layer->weights[i*current_layer->prev->neurons+j] -=
					alpha * current_layer->deltas[i] * current_layer->prev->outputs[j];

			current_layer->biases[i] -= alpha * current_layer->deltas[i];
		}
		current_layer = current_layer->next;
	}
}

void subsample (SIGNAL in_signal,SIGNAL out_signal,float subsample_rate) {
	float subsample_ratio = subsample_rate / in_signal->sample_rate;
	int len = in_signal->points;
	int len_new = out_signal->points = ceilf(len * subsample_ratio);
	
	out_signal->sample_rate = subsample_rate;
	// allocate time and signal arrays
	out_signal->s = (float *)malloc(sizeof(float) * len_new);
	out_signal->t = (float *)malloc(sizeof(float) * len_new);
	
	for (int i=0;i<len_new;i++) {
		float position = (float)i / subsample_ratio;
		float position_frac = position - floorf(position);
		int position_int = (int)floorf(position);
		out_signal->s[i] = (1.f - position_frac)*
					in_signal->s[position_int] +
					position_frac*
					in_signal->s[position_int+1];
		out_signal->t[i] = (float)i / out_signal->sample_rate;
	}
}

void initialize_signal_parameters (PARAMS myparams) {
	myparams->freqs = (float*)malloc(sizeof(float)*7);
	myparams->freqs[0]=2;
	myparams->freqs[1]=3.7;
	myparams->freqs[2]=7.8;
	myparams->freqs[3]=0.12;
	myparams->freqs[4]=0.54;
	myparams->freqs[5]=1.3;
	myparams->freqs[6]=0;
	myparams->phases = (float*)malloc(sizeof(float)*7);
	myparams->phases[0]=0;
	myparams->phases[1]=1;
	myparams->phases[2]=2;
	myparams->phases[3]=3;
	myparams->phases[4]=5;
	myparams->phases[5]=1;
	myparams->phases[6]=0;
	myparams->amps = (float*)malloc(sizeof(float)*7);
	myparams->amps[0]=1;
	myparams->amps[1]=2;
	myparams->amps[2]=3;
	myparams->amps[3]=0.7;
	myparams->amps[4]=2.3;
	myparams->amps[5]=1;
	myparams->amps[6]=0;
	myparams->time = 60.f;
	myparams->sample_rate=SAMPLE_RATE;
}

void initialize_mlp (struct layer *layers,Onnx__ModelProto *model) {
	// input layer
	layers[0].isinput=1;
	layers[0].neurons=HISTORY_LENGTH;
	layers[0].prev=0;
	layers[0].next=&layers[1];

	// hidden layer
	layers[1].outputs=(float*)malloc(sizeof(float)*HISTORY_LENGTH);
	layers[1].isinput=0;
	layers[1].neurons=HIDDEN_SIZE;
	layers[1].prev=&layers[0];
	layers[1].next=&layers[2];
	layers[1].weights=(float*)malloc(sizeof(float)*HIDDEN_SIZE*HISTORY_LENGTH);

	Onnx__GraphProto *mygraph = model->graph;
	Onnx__TensorProto **mytensor = mygraph->initializer;
	
	for (int i=0;i<HIDDEN_SIZE*HISTORY_LENGTH;i++) {
		if (model) {
			layers[1].weights[i]=*((float *)mytensor[0]->raw_data.data+i);
		} else {
			layers[1].weights[i]=(float)rand()/RAND_MAX - 0.5f;
		}
	}
	layers[1].biases=(float*)malloc(sizeof(float)*HIDDEN_SIZE);
	for (int i=0;i<HIDDEN_SIZE;i++) layers[1].biases[i]=0.f;
	layers[1].outputs=(float*)malloc(sizeof(float)*HIDDEN_SIZE);
	layers[1].deltas=(float*)malloc(sizeof(float)*HIDDEN_SIZE);

	// output layer
	layers[2].isinput=1;
	layers[2].neurons=1;
	layers[2].prev=&layers[1];
	layers[2].next=0;
	layers[2].weights=(float*)malloc(sizeof(float)*HIDDEN_SIZE);
	for (int i=0;i<HIDDEN_SIZE;i++) {
		if (model) {
			1;
			//layers[2].weights[i]=*((float *)model->graph.initializer[2]->raw_data+i);
			layers[2].weights[i]=*((float *)mytensor[2]->raw_data.data+i);
		} else {
			layers[2].weights[i]=(float)rand()/RAND_MAX - 0.5f;
		}
	}
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

void free_signal (SIGNAL mysignal) {
	free(mysignal->t);
	free(mysignal->s);
}

int main (int argc,char **argv) {
	if (argc != 2) {
		fprintf(stderr,"usage:\n%s <onnx file>\n",argv[0]);
		return 1;
	}

	// open ONNX file
	Onnx__ModelProto *model = openOnnxFile(argv[1]);
	if (!model) {
		char str[1024];
		sprintf(str,"Error opening \"%s\"\n",argv[1]);
		perror(str);
		return 1;
	}
	
	// set up signal
	PARAMS myparams = (PARAMS)malloc(sizeof(struct params));
	initialize_signal_parameters(myparams);
	
	// synthesize and plot signal
	SIGNAL mysignal = (SIGNAL)malloc(sizeof(struct signal));
	generate_synthetic_data (myparams,mysignal);
	plot(mysignal,"original signal");
	
	SIGNAL mysignal_subsampled = (SIGNAL)malloc(sizeof(struct signal));
	subsample(mysignal,mysignal_subsampled,SUBSAMPLE_RATE);
	plot(mysignal_subsampled,"original signal subsampled");
	
	// set up MLP
	struct layer layers[3];
	initialize_mlp(layers,model);

	// set up predicted signal
	SIGNAL mysignal_predicted=(SIGNAL)malloc(sizeof(struct signal));
	// set number of points to that of subsampled signal
	mysignal_predicted->points = mysignal_subsampled->points;
	// allocate time axis
	mysignal_predicted->t = (float *)malloc(mysignal_subsampled->points * sizeof(float));
	// copy time axis from subssampled signal
	memcpy(mysignal_predicted->t,mysignal_subsampled->t,mysignal_subsampled->points);
	// allocate y axis
	mysignal_predicted->s = (float *)malloc(mysignal_subsampled->points * sizeof(float));

	// zero-pad on the left side
	for (int i=0;i<mysignal_subsampled->points - PREDICTION_TIME;i++) {
		int start_of_input = i - HISTORY_LENGTH - PREDICTION_TIME;

		if (start_of_input >= 0) {
			// compute outputs
			layers[0].outputs = &mysignal_subsampled->s[start_of_input];
			forward_pass(layers);

			// compute deltas
			backward_pass(layers,&mysignal_subsampled->s[i]);
			// update weights
			update_weights(layers,0.01);
		}

		// make prediction
		start_of_input = i - HISTORY_LENGTH;
		if (start_of_input < 0) {
			mysignal_predicted->s[i] = 0.f;
		} else {	
			// make a prediction
			layers[0].outputs = &mysignal_subsampled->s[i-HISTORY_LENGTH];
			forward_pass(layers);
			mysignal_predicted->s[i] = layers[2].outputs[0];
		}
	}
	
	plot(mysignal_predicted,"predicted signal");	

	// clean up
	free(mysignal);
	free(mysignal_subsampled);
	free(myparams->freqs);
	free(myparams->phases);
	free(myparams);
	
	return 0;
}

