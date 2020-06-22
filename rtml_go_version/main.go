package main

// XXX: you need to use the fork of GIU https://github.com/charlesdaniels/giu
// because this program depends on experimental, un-merged implot bindings.

import (
	"fmt"
	"os"

	g "github.com/AllenDang/giu"
	"github.com/AllenDang/giu/imgui"

	"github.com/HeRCLab/dropbear_lstm/rtml_go_version/mlp"

	"github.com/akamensky/argparse"

	"math"

	"github.com/herclab/wavegen/pkg/wavegen"
)

const (
	HISTORY_LENGTH  int     = 10
	HIDDEN_SIZE     int     = 10
	SAMPLE_RATE     float64 = 5000
	SUBSAMPLE       float64 = 0.25
	PREDICTION_TIME int     = 10
	CHANSIZE        int     = 256
	ALPHA           float64 = 0.0001
	DATASET_SIZE    float64 = 60 // number of seconds to run the model for
)

func RunRTML(groundtruth, subchan, predchan, biaschan chan wavegen.Sample, sig *wavegen.Signal) {
	count := 0 // used to know when we need to force a GUI update

	for i, t := range sig.T {
		groundtruth <- wavegen.Sample{t, sig.S[i]}
		count++
		if count >= CHANSIZE {
			g.Update()
			count = 0
		}
	}

	for i, t := range sig.T {
		subchan <- wavegen.Sample{t, sig.S[i]}
		count++
		if count >= CHANSIZE {
			g.Update()
			count = 0
		}
	}

	// nn := mlp.NewMLP(ALPHA, mlp.ReLU, mlp.ReLUDeriv, HISTORY_LENGTH, HIDDEN_SIZE, 1)
	nn := mlp.NewMLP(ALPHA, mlp.Identity, mlp.Unit, HISTORY_LENGTH, HIDDEN_SIZE, 1)
	fmt.Printf("pre-training weights for layer 1 %v\n", nn.Layer[1].Weight)
	fmt.Printf("pre-training biases for layer 1 %v\n", nn.Layer[1].Bias)

	for i := 2*HISTORY_LENGTH + 2*PREDICTION_TIME + 2; i < len(sig.T); i++ {

		// first train with the available data...
		err := nn.ForwardPass(sig.S[i-2*HISTORY_LENGTH-2*PREDICTION_TIME-2 : i-HISTORY_LENGTH-2*PREDICTION_TIME-2])
		if err != nil {
			panic(err)
		}

		err = nn.BackwardPass(sig.S[i-HISTORY_LENGTH-PREDICTION_TIME-1 : i-HISTORY_LENGTH-PREDICTION_TIME])
		if err != nil {
			panic(err)
		}

		nn.UpdateWeights()

		// now make a prediction
		nn.ForwardPass(sig.S[i-HISTORY_LENGTH-PREDICTION_TIME : i-PREDICTION_TIME])

		// t := sub.T[i-PREDICTION_TIME] // time of Activation[0]
		// t := sub.T[i-PREDICTION_TIME] - float64(HISTORY_LENGTH+1)/sub.SampleRate
		// t := sub.T[i] // time of Activation[9]
		t := sig.T[i] - float64(PREDICTION_TIME)/sig.SampleRate

		predchan <- wavegen.Sample{t, nn.OutputLayer().Output[0]}
		biaschan <- wavegen.Sample{t, nn.OutputLayer().Bias[0]}

		// XXX: where does this magic number come from?!
		// predchan <- Point{t + float64(20)/sub.SampleRate, nn.OutputLayer().Activation[0]}

		g.Update()
		count++
		if count >= CHANSIZE {
			g.Update()
			count = 0
		}

	}

	fmt.Printf("post-training weights for layer 1 %v\n", nn.Layer[1].Weight)
	fmt.Printf("psot-training biases for layer 1 %v\n", nn.Layer[1].Bias)

}

func rmse(theta1, theta2 float32) float32 {
	return float32(math.Sqrt(math.Pow(float64(theta1-theta2), 2)))
}

var groundchannel chan wavegen.Sample
var groundX []float32
var groundY []float32
var subchannel chan wavegen.Sample
var subX []float32
var subY []float32
var predchannel chan wavegen.Sample
var predX []float32
var predY []float32
var rmseX []float32
var rmseY []float32
var biaschannel chan wavegen.Sample
var biasX []float32
var biasY []float32

func loop() {
	// first, check if we got any new asynchronous data and flush
	// the channels...
	for {
		select {
		case p := <-groundchannel:
			groundX = append(groundX, float32(p.T))
			groundY = append(groundY, float32(p.S))
		default:
			// we have exhausted all of the data available
			goto flushedground
		}
	}

flushedground:

	for {
		select {
		case p := <-subchannel:
			subX = append(subX, float32(p.T))
			subY = append(subY, float32(p.S))
		default:
			// we have exhausted all of the data available
			goto flushedsub
		}
	}

flushedsub:

	for {
		select {
		case p := <-predchannel:
			predX = append(predX, float32(p.T))
			predY = append(predY, float32(p.S))
		default:
			// we have exhausted all of the data available
			goto flushedpred
		}
	}

flushedpred:

	for {
		select {
		case p := <-biaschannel:
			biasX = append(biasX, float32(p.T))
			biasY = append(biasY, float32(p.S))
		default:
			// we have exhausted all of the data available
			goto flushedbias
		}
	}

flushedbias:

	if len(predX) > 0 {
		// subX[i] should be at right around the start of the
		// predictions
		i := 0
		for subX[i] < predX[0] {
			i++
		}

		// first time, since we'll key of of rmseX later, this has to
		// be handled separately
		if len(rmseX) == 0 {
			rmseX = append(rmseX, predX[0])
			rmseY = append(rmseY, rmse(predY[0], subY[i]))
		}

		// This is more complicated than it strictly needs to be,
		// because I want to account for pred and sub having
		// potentially different X values. The right way to do this
		// would be to interpolate both to have exactly aligned X
		// values.
		for len(rmseX) < len(predX) {
			x := rmseX[len(rmseX)-1]
			i := 0
			for predX[i] < x {
				i++
			}
			i++

			j := 0
			for subX[j] < predX[i] {
				j++
				if j >= len(subX) {
					goto rmsedone
				}
			}

			rmseX = append(rmseX, predX[i])
			rmseY = append(rmseY, rmse(predY[i], subY[j]))
		}
	}

rmsedone:

	g.SingleWindow("RTML", g.Layout{
		g.SplitLayout("split1", g.DirectionVertical, false, 300,
			g.Wrapper(func() {

				// no data to plot yet
				if len(groundX) == 0 || len(subX) == 0 {
					return
				}

				if (imgui.BeginPlot("Ground Truth", "t", "s", imgui.Vec2{-1, -1}, int(imgui.ImPlotFlags_Default), int(imgui.ImPlotAxisFlags_Default), int(imgui.ImPlotAxisFlags_Default), int(imgui.ImPlotAxisFlags_Auxiliary), int(imgui.ImPlotAxisFlags_Auxiliary))) {
					imgui.PlotLinePoints("baseline", groundX, groundY, 0)
					imgui.PlotLinePoints("subsampled", subX, subY, 0)
					imgui.EndPlot()
				}

			}),
			g.Wrapper(func() {

				// no data to plot yet
				if len(predX) == 0 || len(subX) == 0 || len(rmseX) == 0 {
					return
				}

				if (imgui.BeginPlot("Predicted Signal", "t", "s", imgui.Vec2{-1, -1}, int(imgui.ImPlotFlags_Default), int(imgui.ImPlotAxisFlags_Default), int(imgui.ImPlotAxisFlags_Default), int(imgui.ImPlotAxisFlags_Auxiliary), int(imgui.ImPlotAxisFlags_Auxiliary))) {
					imgui.PlotLinePoints("predicted", predX, predY, 0)
					imgui.PlotLinePoints("subsampled", subX, subY, 0)
					imgui.PlotLinePoints("RMSE", rmseX, rmseY, 0)
					imgui.PlotLinePoints("output bias", biasX, biasY, 0)
					imgui.EndPlot()
				}

			}),
		),
	})
}

func main() {
	parser := argparse.NewParser("rtml", "real time machine learning")

	inputfile := parser.String("i", "input", &argparse.Options{Required: true, Help: "Input Wavegen file"})

	if err := parser.Parse(os.Args); err != nil {
		fmt.Print(parser.Usage(err))
		panic("Error parsing arguments")
	}

	wf, err := wavegen.ReadJSON(*inputfile)
	if err != nil {
		panic(err)
	}

	groundchannel = make(chan wavegen.Sample, CHANSIZE)
	groundX = make([]float32, 0)
	groundY = make([]float32, 0)

	subchannel = make(chan wavegen.Sample, CHANSIZE)
	subX = make([]float32, 0)
	subY = make([]float32, 0)

	predchannel = make(chan wavegen.Sample, CHANSIZE)
	predX = make([]float32, 0)
	predY = make([]float32, 0)

	biaschannel = make(chan wavegen.Sample, CHANSIZE)
	biasX = make([]float32, 0)
	biasY = make([]float32, 0)

	rmseX = make([]float32, 0)
	rmseY = make([]float32, 0)

	go RunRTML(groundchannel, subchannel, predchannel, biaschannel, wf.Signal)

	wnd := g.NewMasterWindow("Hello world", 700, 800, 0, nil)
	wnd.Main(loop)
}
