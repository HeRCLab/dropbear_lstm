package wavegen

import (
	"math"
	"testing"
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
