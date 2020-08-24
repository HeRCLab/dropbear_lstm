package parameters

const (
	HISTORY_LENGTH  int     = 10
	HIDDEN_SIZE     int     = 10
	SAMPLE_RATE     float64 = 5000
	SUBSAMPLE       float64 = 0.25
	PREDICTION_TIME int     = 1
	CHANSIZE        int     = 256
	ALPHA           float64 = 0.00001
	DATASET_SIZE    float64 = 60 // number of seconds to run the model for
)
