package wavegen

import (
	"math"
	"testing"

	"github.com/montanaflynn/stats"
)

func TestValidateIndex(t *testing.T) {
	sig := &Signal{
		S:          []float64{1, 2, 3},
		T:          []float64{1, 2, 3},
		SampleRate: 1,
	}

	var err error

	err = sig.ValidateIndex(-1)
	if err == nil {
		t.Errorf("ValidateIndex(-1) should have errored but did not")
	}

	err = sig.ValidateIndex(3)
	if err == nil {
		t.Errorf("ValidateIndwx(3) should have errored but did not")
	}

	for _, v := range []int{0, 1, 2} {
		err = sig.ValidateIndex(v)
		if err != nil {
			t.Errorf("ValidateIndex(%d) should NOT have errored but did", v)
		}
	}

	sig.S = append(sig.S, 4)
	err = sig.ValidateIndex(0)
	if err == nil {
		t.Errorf("ValidateIndex failed to detect corrupted signal")
	}

}

func TestNearestIndex(t *testing.T) {
	sig := &Signal{
		S:          []float64{1, 2, 3},
		T:          []float64{1, 2, 3},
		SampleRate: 1,
	}

	cases := []struct {
		time      float64
		overshoot bool
		expect    int
	}{
		{0, true, 0},
		{0, false, 0},
		{0.5, true, 0},
		{0.5, false, 0},
		{1, true, 0},
		{1, false, 0},
		{1.5, true, 1},
		{1.5, false, 0},
		{2.5, true, 2},
		{2.5, false, 1},
		{3, false, 2},
		{3, true, 2},
		{3.0, false, 2},
		{3.0, true, 2},
		{3.5, false, 2},
		{3.5, true, 2},
		{4.5, false, 2},
		{4.5, true, 2},
	}

	for i, c := range cases {
		res := sig.NearestIndex(c.time, c.overshoot)
		if res != c.expect {
			t.Errorf("Test case %d failed: NearestIndex(%f, %v)=%d, but expected %d",
				i, c.time, c.overshoot, res, c.expect)
		}
	}

	sig = &Signal{}
	res := sig.NearestIndex(0, true)
	if res != 0 {
		t.Errorf("NearestIndex does not correctly handle zero-length case.")
	}
	res = sig.NearestIndex(0, false)
	if res != 0 {
		t.Errorf("NearestIndex does not correctly handle zero-length case.")
	}
}

func recoveryWrapper(sig *Signal, index int) (res Sample, paniced bool) {
	defer func() {
		if r := recover(); r != nil {
			paniced = true
		}

	}()

	res = sig.MustIndex(index)

	return res, false
}

func TestIndexing(t *testing.T) {
	sig := &Signal{
		S:          []float64{4, 5, 6},
		T:          []float64{1, 2, 3},
		SampleRate: 1,
	}

	cases := []struct {
		index       int
		shouldpanic bool
		expect      Sample
	}{
		{0, false, Sample{1, 4}},
		{1, false, Sample{2, 5}},
		{2, false, Sample{3, 6}},
		{-1, true, Sample{0, 0}},
		{3, true, Sample{0, 0}},
	}

	for i, c := range cases {
		res, paniced := recoveryWrapper(sig, c.index)
		if c.shouldpanic != paniced {
			t.Errorf("Test case %d: MustIndex(%d) should have caused a panic, but did not",
				i, c.index)
		}

		if res != c.expect {
			t.Errorf("Test case %d: MustIndex(%d)=%v, should have been %v",
				i, c.index, res, c.expect)
		}
	}

}

func TestInterpolation(t *testing.T) {
	sig := &Signal{
		S:          []float64{4, 5, 6},
		T:          []float64{1, 2, 3},
		SampleRate: 1,
	}

	eta := 0.0000001

	cases := []struct {
		time   float64
		expect Sample
	}{
		{0, Sample{0, 4}},
		{1, Sample{1, 4}},
		{1.25, Sample{1.25, 4.25}},
		{1.5, Sample{1.5, 4.5}},
		{1.75, Sample{1.75, 4.75}},
		{2, Sample{2, 5}},
		{2.1, Sample{2.1, 5.1}},
		{3, Sample{3, 6}},
		{4, Sample{4, 6}},
	}

	for i, c := range cases {
		res := sig.Interpolate(c.time)
		if math.Abs(res[0].T-c.expect.T) > eta || math.Abs(res[0].S-c.expect.S) > eta {
			t.Errorf("Test case %d: Interpolate(%f)=%v, should have been %v",
				i, c.time, res[0], c.expect)
		}
	}
}

func TestValidateParameters(t *testing.T) {
	// make sure filling in the noises and noise magnitudes works
	w := &WaveParameters{
		Frequencies: []float64{1, 2},
	}
	err := w.ValidateParameters()

	if err == nil {
		t.Errorf("failed to detect invalid parameters")
	}

	if len(w.Noises) != 2 {
		t.Errorf("didn't fill in Noises")
	}

	if len(w.NoiseMagnitudes) != 2 {
		t.Errorf("didn't fill in Noises")
	}

	// check mismatched noises and frequencies
	w.Noises = append(w.Noises, "")
	err = w.ValidateParameters()
	if err == nil {
		t.Errorf("failed to detect mismatched noises and frequencies")
	}

	w.Noises = make([]string, 0)
	w.Phases = []float64{3, 4}
	err = w.ValidateParameters()
	if err == nil {
		t.Errorf("failed to detect mismatched amplitudes and frequencies")
	}

	w.Amplitudes = []float64{5, 6}
	w.NoiseMagnitudes = []float64{1, 2, 3}
	err = w.ValidateParameters()
	if err == nil {
		t.Errorf("failed to detect mismatched noise magnitudes and frequencies")
	}

	w.NoiseMagnitudes = []float64{}
	err = w.ValidateParameters()
	if err != nil {
		t.Errorf("correct parameters incorrectly errored")
	}
}

func TestNoise(t *testing.T) {
	runcount := 10000
	eta := 0.1

	w := &WaveParameters{
		SampleRate:           50,
		Offset:               0,
		Duration:             10,
		Frequencies:          []float64{1, 2},
		Phases:               []float64{0, 2},
		Amplitudes:           []float64{1, 2},
		Noises:               []string{"none", "pseudo"},
		NoiseMagnitudes:      []float64{1.0, 2.0},
		GlobalNoise:          "pseudo",
		GlobalNoiseMagnitude: 3.0,
	}

	cases := []struct {
		index        int
		expectMedian float64
		expectMin    float64
		expectMax    float64
	}{
		{0, 0, 0, 0},
		{1, 1.0, 0.0, 2.0},
		{-1, 1.5, 0.0, 3.0},
	}

	for i, c := range cases {
		values := make([]float64, runcount)
		for j, _ := range values {
			var err error
			values[j], err = w.Noise(c.index)
			if err != nil {
				t.Error(err)
			}
			if (values[j] < c.expectMin) || (values[j] > c.expectMax) {
				t.Errorf("Test case %d: Value %f out of bounds %f...%f",
					i, values[j], c.expectMin, c.expectMax)
			}
		}
		median, err := stats.Median(values)
		if err != nil {
			t.Error(err)
		}
		if math.Abs(median-c.expectMedian) > eta {
			t.Errorf("Test case %d: expected median value %f, but got %f",
				i, c.expectMedian, median)
		}
	}

	_, err := w.Noise(-2)
	if err == nil {
		t.Errorf("Noise failed to detect out of bounds index")
	}

	_, err = w.Noise(2)
	if err == nil {
		t.Errorf("Noise failed to detect out of bounds index")
	}

}

func TestWavegen(t *testing.T) {
	eta := 0.0000001

	w := &WaveParameters{
		SampleRate:           10000,
		Offset:               0,
		Duration:             10,
		Frequencies:          []float64{1},
		Phases:               []float64{0},
		Amplitudes:           []float64{1},
		Noises:               []string{"none"},
		NoiseMagnitudes:      []float64{1.0},
		GlobalNoise:          "none",
		GlobalNoiseMagnitude: 0.0,
	}

	sig, err := w.GenerateSyntheticData()

	if err != nil {
		t.Error(err)
	}

	res := sig.Interpolate(0, 0.25, 0.5, 0.75, 1)
	expect := []float64{0, 1, 0, -1, 0}

	if len(res) != len(expect) {
		t.Fatalf("Wrong number of samples, Interpolate() is broken!")
	}

	for i, v := range res {
		if math.Abs(v.S-expect[i]) > eta {
			t.Errorf("Test case %d: expected signal value %f, got %f",
				i, expect[i], v.S)
		}
	}

}
