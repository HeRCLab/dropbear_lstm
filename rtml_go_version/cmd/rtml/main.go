package main

import (
	"fmt"
	"os"

	"github.com/herclab/dropbear_lstm/rtml_go_version/mlp"
	"github.com/herclab/dropbear_lstm/rtml_go_version/mlp/parameters"

	"github.com/akamensky/argparse"

	"math"

	"github.com/herclab/wavegen/pkg/wavegen"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
)

type RTMLResult struct {
	GroundTruth []wavegen.Sample
	Subsampled  []wavegen.Sample
	Prediction  []wavegen.Sample
	Bias        []wavegen.Sample
	RMSE        []wavegen.Sample
	NN          *mlp.MLP
}

func RunRTML(sig *wavegen.Signal) RTMLResult {

	res := RTMLResult{
		GroundTruth: make([]wavegen.Sample, 0),
		Subsampled:  make([]wavegen.Sample, 0),
		Prediction:  make([]wavegen.Sample, 0),
		Bias:        make([]wavegen.Sample, 0),
		RMSE:        make([]wavegen.Sample, 0),
	}

	for i, t := range sig.T {
		res.GroundTruth = append(res.GroundTruth, wavegen.Sample{t, sig.S[i]})
	}

	for i, t := range sig.T {
		// XXX: should actually subsample???
		res.Subsampled = append(res.Subsampled, wavegen.Sample{t, sig.S[i]})
	}

	// nn := mlp.NewMLP(parameters.ALPHA, mlp.ReLU, mlp.ReLUDeriv, parameters.HISTORY_LENGTH, parameters.HIDDEN_SIZE, 1)
	nn := mlp.NewMLP(parameters.ALPHA, mlp.Identity, mlp.Unit, parameters.HISTORY_LENGTH, parameters.HIDDEN_SIZE, 1)
	fmt.Printf("pre-training weights for layer 1 %v\n", nn.Layer[1].Weight)
	fmt.Printf("pre-training biases for layer 1 %v\n", nn.Layer[1].Bias)
	res.NN = nn

	for i := 2*parameters.HISTORY_LENGTH + 2*parameters.PREDICTION_TIME + 2; i < len(sig.T); i++ {

		if i%10 == 0 {
			nn.Snapshot()
		}

		// first train with the available data...
		err := nn.ForwardPass(sig.S[i-2*parameters.HISTORY_LENGTH-2*parameters.PREDICTION_TIME-2 : i-parameters.HISTORY_LENGTH-2*parameters.PREDICTION_TIME-2])
		if err != nil {
			panic(err)
		}

		err = nn.BackwardPass(sig.S[i-parameters.HISTORY_LENGTH-parameters.PREDICTION_TIME-1 : i-parameters.HISTORY_LENGTH-parameters.PREDICTION_TIME])
		if err != nil {
			panic(err)
		}

		nn.UpdateWeights()

		// now make a prediction
		nn.ForwardPass(sig.S[i-parameters.HISTORY_LENGTH-parameters.PREDICTION_TIME : i-parameters.PREDICTION_TIME])

		// t := sub.T[i-parameters.PREDICTION_TIME] // time of Activation[0]
		// t := sub.T[i-parameters.PREDICTION_TIME] - float64(parameters.HISTORY_LENGTH+1)/sub.SampleRate
		// t := sub.T[i] // time of Activation[9]
		t := sig.T[i] - float64(parameters.PREDICTION_TIME)/sig.SampleRate

		res.Prediction = append(res.Prediction, wavegen.Sample{t, nn.OutputLayer().Output[0]})
		res.Bias = append(res.Bias, wavegen.Sample{t, nn.OutputLayer().Bias[0]})

		// TODO: should probably do RMSE also

		// XXX: where does this magic number come from?!
		// predchan <- Point{t + float64(20)/sub.SampleRate, nn.OutputLayer().Activation[0]}

	}

	return res
}

func rmse(theta1, theta2 float32) float32 {
	return float32(math.Sqrt(math.Pow(float64(theta1-theta2), 2)))
}

func main() {
	parser := argparse.NewParser("rtml", "real time machine learning")

	inputfile := parser.String("i", "input", &argparse.Options{Required: true, Help: "Input Wavegen file."})

	savemlpx := parser.String("m", "savemlpx", &argparse.Options{Help: "Save snapshots in MLPX format to this file."})

	saveplot := parser.String("p", "saveplot", &argparse.Options{Help: "Save results plot in this file."})

	if err := parser.Parse(os.Args); err != nil {
		fmt.Print(parser.Usage(err))
		panic("Error parsing arguments")
	}

	wf, err := wavegen.ReadJSON(*inputfile)
	if err != nil {
		panic(err)
	}

	res := RunRTML(wf.Signal)

	if *savemlpx != "" {
		res.NN.SaveSnapshot(*savemlpx)
	}

	if *saveplot != "" {

		p, err := plot.New()
		if err != nil {
			panic(err)
		}

		p.X.Label.Text = "Time"

		err = plotutil.AddLinePoints(p,
			"Ground Truth", wavegen.SampleList(res.GroundTruth),
			"Subsampled", wavegen.SampleList(res.Subsampled),
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
