#include "rtml.h"

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
		layer->weights = malloc(sizeof(double) * nweights);
		if (layer->weights == NULL) {
			fprintf(stderr, "could not allocate memory!\n");
			exit(1);
		}
		for (int weightindex = 0 ; weightindex < nweights ; weightindex++) {
			double weight;
			mlpx_must(MLPXLayerGetWeight(handle, 0, layerindex, weightindex, &weight));
			layer->weights[weightindex] = (double) weight;
			mlpx_must(MLPXLayerSetWeight(m->mlpxhandle, 0, layerindex, weightindex, weight));
		}

		// read in biases from MLPX
		nbiases = neurons;
		layer->biases = malloc(sizeof(double) * nbiases);
		if (layer->biases == NULL) {
			fprintf(stderr, "could not allocate memory!\n");
			exit(1);
		}
		for (int biasindex = 0 ; biasindex < nbiases ; biasindex++) {
			double bias;
			mlpx_must(MLPXLayerGetBias(handle, 0, layerindex, biasindex, &bias));
			layer->biases[biasindex] = (double) bias;
			mlpx_must(MLPXLayerSetBias(m->mlpxhandle, 0, layerindex, biasindex, bias));
		}

		// initialize outputs
		noutputs = neurons;
		layer->outputs = malloc(sizeof(double) * noutputs);
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

			layer->outputs[outputindex] = (double) output;
			mlpx_must(MLPXLayerSetActivation(m->mlpxhandle, 0, layerindex, outputindex, output));
		}

		// initialize deltas
		ndeltas = neurons;
		layer->deltas = malloc(sizeof(double) * ndeltas);
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

			layer->deltas[deltaindex] = (double) delta;
			mlpx_must(MLPXLayerSetDelta(m->mlpxhandle, 0, layerindex, deltaindex, delta));
		}

		lastlayersize = neurons;
	}

	// Now we establish the links between the layers...
	for (int layerindex = 0 ; layerindex < m->layerc ; layerindex++ ) {
		struct layer* layer;
		layer = &(m->layers[layerindex]);

		layer->prev = (layerindex == 0            ) ? NULL : &(m->layers[layerindex-1]);
		layer->next = (layerindex == (m->layerc-1)) ? NULL : &(m->layers[layerindex+1]);
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

