#include "rtml.h"

struct signal* load_wavegen(char* path) {
	int handle;
	wavegen_must(WGOpen(path, &handle));

	struct signal* sig = malloc(sizeof(struct signal));
	
	wavegen_must(WGSize(handle, &(sig->points)));

	sig->t = malloc(sizeof(double) * sig->points);
	sig->s = malloc(sizeof(double) * sig->points);
	wavegen_must(WGCopyS(handle, sig->s));
	wavegen_must(WGCopyT(handle, sig->t));

	wavegen_must(WGSampleRate(handle, &(sig->sample_rate)));

	wavegen_must(WGClose(handle));

	return sig;
}
