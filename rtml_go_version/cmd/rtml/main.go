package main

import (
	"fmt"
	"os"

	"github.com/herclab/dropbear_lstm/rtml_go_version/mlp"

	"github.com/herclab/herc-file-formats/mlpx/go/mlpx"

	"github.com/akamensky/argparse"

	"github.com/cheggaaa/pb/v3"

	"math"

	"github.com/herclab/wavegen/pkg/wavegen"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
)

type RTMLResult struct {
	GroundTruth []wavegen.Sample
	Prediction  []wavegen.Sample
	Bias        []wavegen.Sample
	RMSE        []wavegen.Sample
	NN          *mlp.MLP
}

const epsilon float64 = 0.00001

type RTMLParameters struct {
	// Frequency is the sample rate used by other parameters. For example
	// HistoryLength * Frequency should be a number of samples.
	Frequency float64

	// HistoryLength is the history length L_h in seconds
	HistoryLength float64

	// PredictionOffset is the prediction offset Δp in seconds
	PredictionOffset float64

	//PredictionLength is the prediction length Lp in seconds
	PredictionLength float64

	// TrainingWindow is the training window Wt in seconds
	TrainingWindow float64
}

// TrainingWindowValues returns an array of signal values comprising the
// training window. The samples will be evenly spaced with a frequency of
// p.Frequency.
func (p RTMLParameters) TrainingWindowValues(sig *wavegen.Signal) []float64 {
	w := []float64{}
	for t := 0.0; t < p.TrainingWindow; t += 1.0 / p.Frequency {
		w = append(w, sig.Interpolate(t)[0].S)
	}

	return w
}

// HistoryWindowValues returns an array of signal values comprising the
// history window relative to the current time tp.
func (p RTMLParameters) HistoryWindowValues(sig *wavegen.Signal, tp float64) []float64 {
	w := []float64{}
	for t := tp - p.HistoryLength; t+epsilon < tp; t += 1.0 / p.Frequency {
		w = append(w, sig.Interpolate(t)[0].S)
	}

	return w
}

// PredictionWindowValues returns an array of signal values comprising the
// prediction window relative to the current time tp.
func (p RTMLParameters) PredictionWindowValues(sig *wavegen.Signal, tp float64) []float64 {
	w := []float64{}
	for t := tp + p.PredictionOffset; t+epsilon <= tp+p.PredictionOffset+p.PredictionLength; t += 1.0 / p.Frequency {
		w = append(w, sig.Interpolate(t)[0].S)
	}

	return w
}

func (p RTMLParameters) RunRTML(nn *mlp.MLP, sig *wavegen.Signal, saveevery int, showprogress bool) RTMLResult {

	res := RTMLResult{
		GroundTruth: make([]wavegen.Sample, 0),
		Prediction:  make([]wavegen.Sample, 0),
		Bias:        make([]wavegen.Sample, 0),
		RMSE:        make([]wavegen.Sample, 0),
		NN:          nn,
	}

	for i, t := range sig.T {
		res.GroundTruth = append(res.GroundTruth, wavegen.Sample{t, sig.S[i]})
	}

	// Create a set of empty values within the training window.
	//
	// NOTE: we want t to be strictly less than the training window because
	// we start tp at p.TrainingWindow later.
	for t := 0.0; t < p.TrainingWindow; t += 1.0 / p.Frequency {
		res.Prediction = append(res.Prediction, wavegen.Sample{t, 0})
	}

	var bar *pb.ProgressBar
	if showprogress {
		bar = pb.StartNew(int((sig.Duration() - p.PredictionLength - p.PredictionOffset) * p.Frequency))
	}

	// Now we actually run the neural network. Note that the upper bound on
	// tp has to accommodate for the fact that we need data in the
	// prediction window for training during each step.
	for tp := 0.0; tp < (sig.Duration() - p.PredictionLength - p.PredictionOffset); tp += 1.0 / p.Frequency {

		if showprogress {
			bar.Increment()
		}

		// Get the input, the history window.
		input := p.HistoryWindowValues(sig, tp)

		// Make a prediction.
		err := nn.ForwardPass(input)
		if err != nil {
			panic(err)
		}

		// We only save the results if we are outside of the training
		// window.
		if tp >= p.TrainingWindow {
			res.Prediction = append(res.Prediction, wavegen.Sample{tp, nn.OutputLayer().Activation[0]})
			res.Bias = append(res.Bias, wavegen.Sample{tp, nn.OutputLayer().Bias[0]})
		}

		// Get data for the backwards pass.
		expectedOutput := p.PredictionWindowValues(sig, tp)

		// Perform the backwards pass.
		err = nn.BackwardPass(expectedOutput)
		if err != nil {
			panic(err)
		}

		// and perform a weight update
		nn.UpdateWeights()

		if int(tp*p.Frequency)%saveevery == 0 {
			nn.Snapshot()
		}
	}

	if showprogress {
		bar.Finish()
	}

	return res
}

func rmse(theta1, theta2 float32) float32 {
	return float32(math.Sqrt(math.Pow(float64(theta1-theta2), 2)))
}

func main() {
	params := RTMLParameters{
		Frequency:        5000.0,
		HistoryLength:    10.0 / 5000.0,
		PredictionOffset: 0,
		PredictionLength: 1.0 / 5000.0,
		// TrainingWindow:   100.0 / 5000.0,
		TrainingWindow: 0,
	}

	parser := argparse.NewParser("rtml", "real time machine learning")

	inputfile := parser.String("i", "input", &argparse.Options{Required: true, Help: "Input Wavegen file."})

	loadmlpx := parser.String("l", "loadmlpx", &argparse.Options{Required: true, Help: "Use the specified MLPX file to initialize the network. The initializer snapshot is used to restore weights and biases (other values are ignored)."})

	savemlpx := parser.String("m", "savemlpx", &argparse.Options{Help: "Save snapshots in MLPX format to this file."})

	saveplot := parser.String("p", "saveplot", &argparse.Options{Help: "Save results plot in this file."})

	showprog := parser.Flag("P", "progress", &argparse.Options{Help: "Show progress bar during execution."})

	saveevery := parser.Int("e", "saveevery", &argparse.Options{
		Help:    "Save an MLPX snapshot every saveevery iterations",
		Default: 10,
	})

	if err := parser.Parse(os.Args); err != nil {
		fmt.Print(parser.Usage(err))
		panic("Error parsing arguments")
	}

	m, err := mlpx.ReadJSON(*loadmlpx)
	if err != nil {
		panic(err)
	}

	nn, err := mlp.NewMLPFromMLPX(m)

	if err != nil {
		panic(err)
	}

	wf, err := wavegen.ReadJSON(*inputfile)
	if err != nil {
		panic(err)
	}

	res := params.RunRTML(nn, wf.Signal, *saveevery, *showprog)

	if *savemlpx != "" {
		err := res.NN.SaveSnapshot(*savemlpx)
		if err != nil {
			panic(err)
		}
	}

	if *saveplot != "" {

		p, err := plot.New()
		if err != nil {
			panic(err)
		}

		p.X.Label.Text = "Time"

		err = plotutil.AddLinePoints(p,
			"Ground Truth", wavegen.SampleList(res.GroundTruth),
			"Prediction", wavegen.SampleList(res.Prediction),
			"Bias", wavegen.SampleList(res.Bias))
		if err != nil {
			panic(err)
		}

		err = p.Save(10*vg.Inch, 6*vg.Inch, *saveplot)
		if err != nil {
			panic(err)
		}

	}
}
