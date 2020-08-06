#include "rtml.h"

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

const char *argp_program_version = "online_training 0.0.1";
const char *argp_program_bug_address = "<3D22>";
static char doc[] = "RTML -- C implementation";
static char args_doc[] = "INPUT_WAVEGEN INPUT_MLPX OUTPUT_MLPX";
static struct argp_option options[] = {
    { "snapshotinterval", 's', "INTERVAL", OPTION_ARG_OPTIONAL, "Every snapshotinterval many passes, an MLPX snapshot will be generated. (default: 250)"},
    { 0 }
};

struct arguments {
	int snapshotinterval;
	char* inputmlpx;
	char* inputwavegen;
	char* outputmlpx;
};

static error_t parse_opt(int key, char* arg, struct argp_state *state) {
	struct arguments *arguments = state->input;

	switch(key) {
		case 's':
			if (arg == NULL) {
				fprintf(stderr, "Must provide a parameter for --snapshotinterval\n");
				return ARGP_ERR_UNKNOWN;
			}

			// this works around that in the short form, -s=X, the
			// = is included in the arg, but not in the long form
			// case.
			arguments->snapshotinterval = atoi((*arg == '=') ? &(arg[1]) : arg);
			break;

		case ARGP_KEY_ARG:
			if (state->arg_num == 0) {
				arguments->inputwavegen = arg;
			} else if (state->arg_num == 1) {
				arguments->inputmlpx = arg;
			} else if (state->arg_num == 2) {
				arguments->outputmlpx = arg;
			} else {
				// too many args
				argp_usage(state);
			}
			break;

		case ARGP_KEY_END:
			if (state->arg_num < 3) {
				// not enough args
				argp_usage(state);
			}
			break;

		default:
			return ARGP_ERR_UNKNOWN;
	}

	return 0;
}

static struct argp argp = { options, parse_opt, args_doc, doc };

int main(int argc, char *argv[]) {
	struct arguments arguments;

	arguments.snapshotinterval = 250;
	arguments.inputmlpx = NULL;
	arguments.outputmlpx = NULL;
	arguments.inputwavegen = NULL;

	argp_parse(&argp, argc, argv, 0, 0, &arguments);

	printf("snapshotinterval=%i\n", arguments.snapshotinterval);
	printf("inputmlpx=%s\n", arguments.inputmlpx);
	printf("outputmlpx=%s\n", arguments.outputmlpx);
	printf("inputwavegen=%s\n", arguments.inputwavegen);

	if (arguments.snapshotinterval < 1) {
		fprintf(stderr, "--snapshotinterval must be at least 1\n");
		exit(1);
	}

	struct mlp* m;

	m = load_mlpx(arguments.inputmlpx, 0);

	// load in signal from input wavegen file
	SIGNAL mysignal_subsampled = load_wavegen(arguments.inputwavegen);
	plot(mysignal_subsampled,"wavegen signal");

	// set up predicted signal
	SIGNAL mysignal_predicted=(SIGNAL)malloc(sizeof(struct signal));
	// set number of points to that of subsampled signal
	mysignal_predicted->points = mysignal_subsampled->points;
	// allocate time axis
	mysignal_predicted->t = (double *)malloc(mysignal_subsampled->points * sizeof(double));
	// copy time axis from subssampled signal
	memcpy(mysignal_predicted->t,mysignal_subsampled->t,mysignal_subsampled->points);
	// allocate y axis
	mysignal_predicted->s = (double *)malloc(mysignal_subsampled->points * sizeof(double));

	// make sure that the size of the input layer is large enough
	if (m->layers[0].neurons < HISTORY_LENGTH) {
		fprintf(stderr, "FATAL: input layer size %d neurons too small, must be at least %d\n",
				m->layers[0].neurons, HISTORY_LENGTH);
		exit(1);
	}

	// sanity-check number of snapshots
	if ( ((1.0f * mysignal_subsampled->points) / arguments.snapshotinterval) > 5000) {
		fprintf(stderr, "FATAL: proposed snapshotinterval results in more than 5k snapshots, you almost certainly don't want to do this\n");
		exit(1);
	}

	fprintf(stderr, "\n");

	// zero-pad on the left side
	for (int i=0;i<HISTORY_LENGTH+PREDICTION_TIME;i++) {
		mysignal_predicted->s[i] = 0.f;
	}
	for (int i=HISTORY_LENGTH+PREDICTION_TIME;i<mysignal_subsampled->points;i++) {
		// make prediction based on current weights
		m->layers[0].outputs = &mysignal_subsampled->s[i-HISTORY_LENGTH-PREDICTION_TIME];
		forward_pass(m);
		mysignal_predicted->s[i] = m->layers[m->layerc-1].outputs[0];

		// update weights
		backward_pass(m, &mysignal_subsampled->s[i]);
		update_weights(m);

		if (i % arguments.snapshotinterval == 0) {
			fprintf(stderr, "\rprogress: %f%%", (i * 100.0f) / mysignal_subsampled->points);
			take_mlpx_snapshot(m);
		}
	}
	fprintf(stderr, "\n");

	plot(mysignal_predicted,"predicted signal");

	save_mlpx(m, arguments.outputmlpx);

	return 0;
}

