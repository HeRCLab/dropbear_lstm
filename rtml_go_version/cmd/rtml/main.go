package main

import (
	"fmt"
	"os"

	"github.com/herclab/dropbear_lstm/rtml_go_version/mlp"
	"github.com/herclab/dropbear_lstm/rtml_go_version/mlp/parameters"

	"github.com/herclab/herc-file-formats/mlpx/go/mlpx"

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

func RunRTML(nn *mlp.MLP, sig *wavegen.Signal, saveevery int) RTMLResult {

	res := RTMLResult{
		GroundTruth: make([]wavegen.Sample, 0),
		Subsampled:  make([]wavegen.Sample, 0),
		Prediction:  make([]wavegen.Sample, 0),
		Bias:        make([]wavegen.Sample, 0),
		RMSE:        make([]wavegen.Sample, 0),
		NN:          nn,
	}

	for i, t := range sig.T {
		res.GroundTruth = append(res.GroundTruth, wavegen.Sample{t, sig.S[i]})
	}

	for i, t := range sig.T {
		// XXX: should actually subsample???
		res.Subsampled = append(res.Subsampled, wavegen.Sample{t, sig.S[i]})
	}

	t := 0.0
	for i := 0; i < parameters.HISTORY_LENGTH+parameters.PREDICTION_TIME; i++ {
		res.Prediction = append(res.Prediction, wavegen.Sample{t, 0})

		t += 1 / sig.SampleRate
	}

	for i := parameters.HISTORY_LENGTH + parameters.PREDICTION_TIME; i < sig.Size(); i++ {

		// copy input into outputs of input layer
		for j := range nn.Layer[0].Output {
			s := sig.MustIndex(i - parameters.HISTORY_LENGTH - parameters.PREDICTION_TIME)
			nn.Layer[0].Activation[j] = s.S
		}

		// make a prediction
		nn.ForwardPass(nn.Layer[0].Activation)
		res.Prediction = append(res.Prediction, wavegen.Sample{t, nn.OutputLayer().Activation[0]})
		res.Bias = append(res.Bias, wavegen.Sample{t, nn.OutputLayer().Bias[0]})

		// propagate actual results
		output := []float64{}
		for j := 0; j < nn.OutputLayer().TotalNeurons(); j++ {
			output = append(output, sig.MustIndex(i+j).S)
		}
		nn.BackwardPass(output)

		nn.UpdateWeights()

		if i%saveevery == 0 {
			nn.Snapshot()
		}

		t += 1 / sig.SampleRate
	}

	return res
}

func rmse(theta1, theta2 float32) float32 {
	return float32(math.Sqrt(math.Pow(float64(theta1-theta2), 2)))
}

func main() {
	parser := argparse.NewParser("rtml", "real time machine learning")

	inputfile := parser.String("i", "input", &argparse.Options{Required: true, Help: "Input Wavegen file."})

	loadmlpx := parser.String("l", "loadmlpx", &argparse.Options{Required: true, Help: "Use the specified MLPX file to initialize the network. The initializer snapshot is used to restore weights and biases (other values are ignored)."})

	savemlpx := parser.String("m", "savemlpx", &argparse.Options{Help: "Save snapshots in MLPX format to this file."})

	saveplot := parser.String("p", "saveplot", &argparse.Options{Help: "Save results plot in this file."})

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

	res := RunRTML(nn, wf.Signal, *saveevery)

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
