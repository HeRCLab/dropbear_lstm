package main

// XXX: you need to use the fork of GIU https://github.com/charlesdaniels/giu
// because this program depends on experimental, un-merged implot bindings.

import (
	"fmt"
	g "github.com/AllenDang/giu"
	"github.com/AllenDang/giu/imgui"
	"github.com/HeRCLab/dropbear_lstm/rtml_go_version/mlp"
	"math"
)

const (
	HISTORY_LENGTH  int     = 10
	HIDDEN_SIZE     int     = 10
	TRAINING_WINDOW int     = 40
	SAMPLE_RATE     float64 = 5000
	SUBSAMPLE       float64 = 0.25
	PREDICTION_TIME int     = 10
	CHANSIZE        int     = 256
	ALPHA           float64 = 0.1
)

type Signal struct {
	T          []float64
	S          []float64
	SampleRate float64
}

type Point struct {
	X float64
	Y float64
}

func GenerateSyntheticData(sample_rate, time float64, freqs, phases, amps []float64) (*Signal, error) {
	// number of points to generate
	points := int(math.Ceil(sample_rate * time))
	sample_period := 1.0 / sample_rate

	if (len(freqs) != len(phases)) || (len(freqs) != len(amps)) || (len(phases) != len(amps)) {
		return nil, fmt.Errorf("freqs, phases, amps must be the same length")
	}

	sig := &Signal{
		T:          make([]float64, points),
		S:          make([]float64, points),
		SampleRate: sample_rate,
	}

	for i := 0; i < points; i++ {
		sig.S[i] = 0
		sig.T[i] = sample_period * float64(i)
		for j, freq := range freqs {
			sig.S[i] += amps[j] * math.Sin(2*math.Pi*freq*sig.T[i]+phases[j])
		}
	}

	return sig, nil
}

func (sig *Signal) Subsample(rate float64) *Signal {
	subsize := int(math.Ceil(float64(len(sig.T)) * rate))
	sub := &Signal{
		T:          make([]float64, subsize),
		S:          make([]float64, subsize),
		SampleRate: sig.SampleRate * rate,
	}

	for i := 0; i < subsize; i++ {
		position := float64(i) / rate
		position_frac := position - math.Floor(position)

		sub.S[i] = (1.0-position_frac)*sig.S[int(position)] + position_frac*sig.S[int(position)+1]
		sub.T[i] = float64(i) / sub.SampleRate
	}

	return sub
}

func RunRTML(groundtruth, subchan, predchan chan Point) {
	sig, err := GenerateSyntheticData(
		SAMPLE_RATE,           // sample_rate
		2.0,                   // time
		[]float64{10, 37, 78}, // freqs
		[]float64{0, 1, 2},    // phases
		[]float64{1, 2, 3},    // amplitudes
	)

	count := 0 // used to know when we need to force a GUI update

	if err != nil {
		panic(err)
	}

	for i, t := range sig.T {
		groundtruth <- Point{t, sig.S[i]}
		count++
		if count >= CHANSIZE {
			g.Update()
			count = 0
		}
	}

	sub := sig.Subsample(SUBSAMPLE)

	for i, t := range sub.T {
		subchan <- Point{t, sub.S[i]}
		count++
		if count >= CHANSIZE {
			g.Update()
			count = 0
		}
	}

	nn := mlp.NewMLP(ALPHA, mlp.Identity, mlp.Unit, HISTORY_LENGTH, HIDDEN_SIZE, PREDICTION_TIME)

	for i := HISTORY_LENGTH + PREDICTION_TIME + 1; i < len(sub.T); i++ {
		t := sub.T[i]
		err := nn.ForwardPass(sub.S[i-HISTORY_LENGTH-PREDICTION_TIME-1 : i-PREDICTION_TIME-1])
		if err != nil {
			panic(err)
		}

		predchan <- Point{t, nn.OutputLayer().Activation[0]}
		g.Update()
		count++
		if count >= CHANSIZE {
			g.Update()
			count = 0
		}

		err = nn.BackwardPass(sub.S[i-PREDICTION_TIME : i])
		if err != nil {
			panic(err)
		}

		nn.UpdateWeights()

	}

}

var groundchannel chan Point
var groundX []float32
var groundY []float32
var subchannel chan Point
var subX []float32
var subY []float32
var predchannel chan Point
var predX []float32
var predY []float32

func loop() {
	// first, check if we got any new asynchronous data and flush
	// the channels...
	for {
		select {
		case p := <-groundchannel:
			groundX = append(groundX, float32(p.X))
			groundY = append(groundY, float32(p.Y))
		default:
			// we have exhausted all of the data available
			goto flushedground
		}
	}

flushedground:

	for {
		select {
		case p := <-subchannel:
			subX = append(subX, float32(p.X))
			subY = append(subY, float32(p.Y))
		default:
			// we have exhausted all of the data available
			goto flushedsub
		}
	}

flushedsub:

	for {
		select {
		case p := <-predchannel:
			predX = append(predX, float32(p.X))
			predY = append(predY, float32(p.Y))
		default:
			// we have exhausted all of the data available
			goto flushedpred
		}
	}

flushedpred:

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
				if len(predX) == 0 {
					return
				}

				if (imgui.BeginPlot("Predicted Signal", "t", "s", imgui.Vec2{-1, -1}, int(imgui.ImPlotFlags_Default), int(imgui.ImPlotAxisFlags_Default), int(imgui.ImPlotAxisFlags_Default), int(imgui.ImPlotAxisFlags_Auxiliary), int(imgui.ImPlotAxisFlags_Auxiliary))) {
					imgui.PlotLinePoints("predicted", predX, predY, 0)
					imgui.PlotLinePoints("subsampled", subX, subY, 0)
					imgui.EndPlot()
				}

			}),
		),
	})
}

func main() {
	groundchannel = make(chan Point, CHANSIZE)
	groundX = make([]float32, 0)
	groundY = make([]float32, 0)

	subchannel = make(chan Point, CHANSIZE)
	subX = make([]float32, 0)
	subY = make([]float32, 0)

	predchannel = make(chan Point, CHANSIZE)
	predX = make([]float32, 0)
	predY = make([]float32, 0)

	go RunRTML(groundchannel, subchannel, predchannel)

	wnd := g.NewMasterWindow("Hello world", 700, 800, 0, nil)
	wnd.Main(loop)
}
