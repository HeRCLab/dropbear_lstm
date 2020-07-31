#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <mlpx.h>

#define	HISTORY_LENGTH	10
#define HIDDEN_SIZE		10
#define TRAINING_WINDOW	40
#define SAMPLE_RATE		5000.f
#define SUBSAMPLE		200.f
#define PREDICTION_TIME	10

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
	int weightc;
	float *weights;
	float *biases;
	float *outputs;
	float *deltas;
	struct layer *prev;
	struct layer *next;
};

struct mlp {
	int layerc;
	struct layer* layers;
	float alpha;
	int mlpxhandle;
};

// Use if a fatal error has occurred relating to MLPX, this will print the
// error to standard error, and then exit nonzero.
#define mlpx_fatal() do { fprintf(stderr, "%s:%d:%s(): MLPX Error: %s\n", __FILE__, __LINE__, __func__, MLPXGetError()); exit(1); } while(0)

// Execute the given statement. If it returns a nonzero value, assume a fatal
// MLPX error has occurred.
#define mlpx_must(_stmt) do { if ( (_stmt) != 0) { mlpx_fatal(); } } while(0)

// load the specified MLPX file, or return nil if an error occurred
//
// howinit can be one of several values:
//
// 0 -- initial weight and bias values in the MLPX are copied, and any
// remaining values are initialized to 0.
//
// 1 -- all initial values available in the MLPX are copied, and any remaining
// values are initialized to 0.
//
// NOTE: we don't support directly initializing to random values, since `mlpx
// new` can already do this for us anyway.
//
// NOTE: the value of alpha for the initialize snapshot is used as the MLP's
// alpha value.
//
// NOTE: because the C implementation does not track activations and outputs
// separately, the outputs field is ignored and the activations field is used
// as intended.
struct mlp* load_mlpx(char* path, int howinit) {
	int handle;
	struct mlp* m;
	double alpha;

	m = malloc(sizeof(struct mlp));
	if (m == NULL) {
		fprintf(stderr, "could not allocate memory!\n");
		exit(1);
	}

	// obtain a handle on the MLPX object
	mlpx_must(MLPXOpen(path, &handle));

	// Get layer count -- snapshot index 0 refers to the snapshot with
	// the earliest position in canonical sort order, which should be
	// the initializer. If there are no snapshots, this will error.
	mlpx_must(MLPXSnapshotGetNumLayers(handle, 0, &(m->layerc)));

	// allocate space for said layers
	m->layers = malloc(sizeof(struct layer) * m->layerc);
	if (m->layers == NULL) {
		fprintf(stderr, "could not allocate memory!\n");
		exit(1);
	}

	// Create a duplicate of the source MLPX which will be topologically
	// identical. We will set it's initializer layers to have the same
	// values as those of the input MLPX.
	mlpx_must(MLPXIsomorphicDuplicate(handle, &(m->mlpxhandle), "initializer"));
	mlpx_must(MLPXSnapshotGetAlpha(handle, 0, &alpha));
	mlpx_must(MLPXSnapshotSetAlpha(m->mlpxhandle, 0, alpha));

	// the "weights" for the input layer are nonsense away, but it makes it
	// easier to interop with MLPX, which assume there are neurons many of
	// them (MLPX also knows they are nonsense, but puts *something* there
	// for convenience).
	int lastlayersize = 1;

	for (int layerindex = 0 ; layerindex < m->layerc ; layerindex++ ) {
		int neurons, nweights, nbiases, noutputs, ndeltas;
		struct layer* layer;

		layer = &(m->layers[layerindex]);

		layer->isinput = (layerindex == 0);

		mlpx_must(MLPXLayerGetNeurons(handle, 0, layerindex, &neurons));
		layer->neurons = neurons;

		// NOTE: we assume the MLPX has weight and bias values and
		// always load from those. This code is intentionally designed
		// so it will error out if those arrays are missing.

		// read in weights from MLPX
		nweights = lastlayersize * neurons;
		layer->weightc = nweights;
		layer->weights = malloc(sizeof(float) * nweights);
		if (layer->weights == NULL) {
			fprintf(stderr, "could not allocate memory!\n");
			exit(1);
		}
		for (int weightindex = 0 ; weightindex < nweights ; weightindex++) {
			double weight;
			mlpx_must(MLPXLayerGetWeight(handle, 0, layerindex, weightindex, &weight));
			layer->weights[weightindex] = (float) weight;
			mlpx_must(MLPXLayerSetWeight(m->mlpxhandle, 0, layerindex, weightindex, weight));
		}

		// read in biases from MLPX
		nbiases = neurons;
		layer->biases = malloc(sizeof(float) * nbiases);
		if (layer->biases == NULL) {
			fprintf(stderr, "could not allocate memory!\n");
			exit(1);
		}
		for (int biasindex = 0 ; biasindex < nbiases ; biasindex++) {
			double bias;
			mlpx_must(MLPXLayerGetBias(handle, 0, layerindex, biasindex, &bias));
			layer->biases[biasindex] = (float) bias;
			mlpx_must(MLPXLayerSetBias(m->mlpxhandle, 0, layerindex, biasindex, bias));
		}

		// initialize outputs
		noutputs = neurons;
		layer->outputs = malloc(sizeof(float) * noutputs);
		if (layer->outputs == NULL) {
			fprintf(stderr, "could not allocate memory!\n");
			exit(1);
		}
		for (int outputindex = 0 ; outputindex < noutputs ; outputindex++) {
			double output;
			if (howinit == 1) {
				mlpx_must(MLPXLayerGetActivation(handle, 0, layerindex, outputindex, &output));
			} else {
				output = 0;
			}

			layer->outputs[outputindex] = (float) output;
			mlpx_must(MLPXLayerSetActivation(m->mlpxhandle, 0, layerindex, outputindex, output));
		}

		// initialize deltas
		ndeltas = neurons;
		layer->deltas = malloc(sizeof(float) * ndeltas);
		if (layer->deltas == NULL) {
			fprintf(stderr, "could not allocate memory!\n");
			exit(1);
		}
		for (int deltaindex = 0 ; deltaindex < ndeltas ; deltaindex++) {
			double delta;
			if (howinit == 1) {
				mlpx_must(MLPXLayerGetDelta(handle, 0, layerindex, deltaindex, &delta));
			} else {
				delta = 0;
			}

			layer->deltas[deltaindex] = (float) delta;
			mlpx_must(MLPXLayerSetDelta(m->mlpxhandle, 0, layerindex, deltaindex, delta));
		}

		lastlayersize = neurons;
	}


	// Release the handle on the original MLPX we used to retrieve our
	// topology
	mlpx_must(MLPXClose(handle));

	return m;
}

// Create a new MLPX snapshot representing the current state of the MLP.
void take_mlpx_snapshot(struct mlp* m) {
	char* nextid;
	int snapc;

	// create the new snapshot
	mlpx_must(MLPXNextSnapshotID(m->mlpxhandle, &nextid));
	mlpx_must(MLPXGetNumSnapshots(m->mlpxhandle, &snapc));
	mlpx_must(MLPXMakeIsomorphicSnapshot(m->mlpxhandle, nextid, snapc-1));
	free(nextid);
	// new snapshot now has index snapc

	for (int layerindex = 0 ; layerindex < m->layerc ; layerindex ++ ) {
		struct layer* layer = &(m->layers[layerindex]);

		char* layerid;
		mlpx_must(MLPXLayerGetIDByIndex(m->mlpxhandle, snapc, layerindex, &layerid));

		for (int i = 0 ; i < layer->weightc ; i++) {
			mlpx_must(MLPXLayerSetWeight(m->mlpxhandle, snapc, layerindex, i, layer->weights[i]));
		}

		for (int i = 0 ; i < layer->neurons ; i++) {
			mlpx_must(MLPXLayerSetDelta(m->mlpxhandle, snapc, layerindex, i, layer->deltas[i]));
			mlpx_must(MLPXLayerSetBias(m->mlpxhandle, snapc, layerindex, i, layer->biases[i]));
			mlpx_must(MLPXLayerSetActivation(m->mlpxhandle, snapc, layerindex, i, layer->outputs[i]));
		}
	}
}

// Save out the current MLPX object to disk
void save_mlpx(struct mlp* m, char* path) {
	mlpx_must(MLPXSave(m->mlpxhandle, path));
}

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

// disabled because of symbol collision
#if 0
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
#endif

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
	// allocate time and signal arrays
	out_signal->s = (float *)malloc(sizeof(float) * len_new);
	out_signal->t = (float *)malloc(sizeof(float) * len_new);

	for (int i=0;i<len_new;i++) {
		float position = (float)i * subsample_rate;
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

void free_signal (SIGNAL mysignal) {
	free(mysignal->t);
	free(mysignal->s);
}

int main () {
	struct mlp* m;

	m = load_mlpx("input.mlpx", 0);
	save_mlpx(m, "temp1.mlpx");

	take_mlpx_snapshot(m);

	save_mlpx(m, "saved.mlpx");

	return;

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

	// set up MLP
	struct layer layers[3];
	initialize_mlp(layers);

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
	for (int i=0;i<HISTORY_LENGTH+PREDICTION_TIME;i++) {
		mysignal_predicted->s[i] = 0.f;
	}
	for (int i=HISTORY_LENGTH+PREDICTION_TIME;i<mysignal_subsampled->points;i++) {
		// make prediction based on current weights
		layers[0].outputs = &mysignal_subsampled->s[i-HISTORY_LENGTH-PREDICTION_TIME];
		forward_pass(layers);
		mysignal_predicted->s[i] = layers[2].outputs[0];

		// update weights
		backward_pass(layers,&mysignal_subsampled->s[i]);
		update_weights(layers,0.01);
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

