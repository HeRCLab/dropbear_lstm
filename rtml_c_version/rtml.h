#ifndef RTML_H
#define RTML_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <mlpx.h>
#include <wavegen.h>
#include <argp.h>

#define	HISTORY_LENGTH	10
#define HIDDEN_SIZE		10
#define TRAINING_WINDOW	40
#define SAMPLE_RATE		5000.f
#define SUBSAMPLE		200.f
#define PREDICTION_TIME	1

struct params {
	double sample_rate;
	double time;
	double *freqs;
	double *phases;
	double *amps;
};

typedef struct params * PARAMS;

struct signal {
	double *t;
	double *s;
	int points;
	double sample_rate;
};

typedef struct signal * SIGNAL;

struct layer {
	int isinput;
	int neurons;
	int weightc;
	double *weights;
	double *biases;
	double *outputs;
	double *deltas;
	struct layer *prev;
	struct layer *next;
};

struct mlp {
	int layerc;
	struct layer* layers;
	double alpha;
	int mlpxhandle;
};

// Use if a fatal error has occurred relating to MLPX, this will print the
// error to standard error, and then exit nonzero.
#define mlpx_fatal() do { fprintf(stderr, "%s:%d:%s(): MLPX Error: %s\n", __FILE__, __LINE__, __func__, MLPXGetError()); exit(1); } while(0)

// Execute the given statement. If it returns a nonzero value, assume a fatal
// MLPX error has occurred.
#define mlpx_must(_stmt) do { if ( (_stmt) != 0) { mlpx_fatal(); } } while(0)

// Use if a fatal error has occurred relating to wavegen, this will print the
// error to standard error, and then exit nonzero.
#define wavegen_fatal() do { fprintf(stderr, "%s:%d:%s(): wavegen Error: %s\n", __FILE__, __LINE__, __func__, WGGetError()); exit(1); } while(0)

// Execute the given statement. If it returns a nonzero value, assume a fatal
// wavegen error has occurred.
#define wavegen_must(_stmt) do { if ( (_stmt) != 0) { wavegen_fatal(); } } while(0)


struct mlp* load_mlpx(char* path, int howinit);
void take_mlpx_snapshot(struct mlp* m);
void save_mlpx(struct mlp* m, char* path);
void generate_synthetic_data (PARAMS myparams,SIGNAL mysignal);
void backward_pass (struct mlp *m,double *y);
void update_weights (struct mlp *m);
double mac(double prev_outputs[1024],double current_weights[1024],int i,int prev_neurons);
void forward_pass (struct mlp *m);
struct signal* load_wavegen(char* path);

#endif /* RTML_H */
