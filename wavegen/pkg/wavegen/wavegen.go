// package wavegen is used for generating synthetic test data for the Dropbear
// LSTM project.
package wavegen

import (
	"fmt"
	"math"
)

// WaveParameters is used to store the parameters that generate a particular
// wave. See GenerateSyntheticData().
type WaveParameters struct {
	// SampleRate is the sample rate at which the wave should be generated,
	// in Hz
	SampleRate float64

	// Offset is the time at which samples should begin being collected, in
	// seconds
	Offset float64

	// Duration is the number of of samples that should be generated
	Duration float64

	// Frequencies is the list of frequencies of Sin wave that should be
	// generated
	Frequencies []float64

	// Phases is the list of phases of the Sin waves that should be
	// generated
	Phases []float64

	// Amplitudes is the list of amplitudes of Sin that should be generated
	Amplitudes []float64

	// Noises stores a list of noise functions which are applied on a
	// per-signal basis.  if this field is left empty, then no noise will
	// be generated for the given signal. These noise functions are
	// understood:
	//
	// * "normal" (rand.NormFloat64)
	// * "" (no noise)
	Noises []string

	// NoiseMagnitues is a list of coefficients to the given noise function
	// for a particular signal.  If empty, it is assumed that all
	// magnitudes are 1.0.
	NoiseMagnitudes []float64

	// GlobalNoise accepts the same values as Noises, but are applied
	// globally rather than to a specific signal.
	GlobalNoise string

	// GlobalNoiseMagnitude works similarly to NoiseMagnitudes, but applies
	// to the global noise.
	GlobalNoiseMagnitude float64
}

// Signal represents a time-series signal
//
// When generating or modifying a signal, you must guarantee that the T and S
// values are sorted by T (e.g. earlier points in time have lower indices), and
// that T ans S are the same size.
type Signal struct {
	// signal values
	s []float64

	// sample rate in Hz
	sampleRate float64

	// offset in seconds of the first signal value from t=0
	offset float64
}

// Sample represents a single sample from a Signal
type Sample struct {

	// The time component of the sample
	T float64

	// The value component of the sample
	S float64
}

// SampleRate returns the signal's sample rate
func (s *Signal) SampleRate() float64 {
	return s.sampleRate
}

// Returns the number of samples which a Signal contains.
func (s *Signal) Size() int {
	return len(s.s)
}

// Index retrieves the ith sample value.
func (s *Signal) Index(i int) (Sample, error) {
	if i < 0 || i >= len(s.s) {
		return Sample{0,0}, fmt.Errorf("Index out of bounds %d for signal of length %d", i, len(s.s))
	}

	t := s.offset + float64(i) * (1 / s.sampleRate)
	s := s.s[i]
	return Sample{T: t, S:s}
}

// MustIndex works identically to Index(), but calls panic() if an error occurs
func (s *Signal) MustIndex(i int) Sample {
	sample, err := s.Index(i)
	if err != nil {
		panic(err)
	}
	return sample
}

// NearestIndex will return the index within a signal which has a time value
// as close as possible to the specified time argument. It will return
// an index with a greater or equal value if overshoot is true, and a lesser
// or equal value if overshoot is false. It will return 0 if the time value
// is before the beginning of the signal, and s.Size()-1 if the time value
// is after the end of the signal.
//
// Note that time includes the offset, so time=0 when the offset is 1 second
// will return 0, since it is before the beginning of the signal data.
func (s *Signal) NearestIndex(time float64, overshoot bool) int {
	if time < s.offset {
		return 0
	}

	t := s.offset
	i := 0
	for {
		if overshoot && (t >= time) {
			return i
		} else if (t == time) || ((t + 1/s.sampleRate) >= time) {
			return i
		}

		i++
		t += 1 / s.sampleRate

		if i >= s.Size() {
			return s.Size() - 1
		}

	}
}

// Interpolate can be used to perform linear interpolation. It will compute a
// separate linear interpolation for each point in times, and return an
// appropriate sample for each.
func (s *Signal) Interpolate(times ...float64) []Sample {
	interpolated := make([]Sample, len(times))
	for t, i := times {
		i0 := s.NearestIndex(t, false) // undershoot
		i1 := s.NearestIndex(t, true) // overshoot

		t0, s0 := s.MustIndex(i0)
		t1, s1 := s.MustIndex(i1)

		// In the case where the requested value is exactly equal to a
		// known value, we can omit the interpolation and return it
		// directly. This would work equivalently with t1 and s1.
		if t0 == t {
			return Signal{T: t, S: s0}
		}

		d0 := math.Abs(t0 - t)
		d1 := math.Abs(t1 - t)

		interpolated[i] = Signal{T: t, S: (d0 * s0) / (d0 + d1) + (d1 * s1) / (d0 + d1)}
	}
}

// GenerateSyntheticData generates a signal which is a composition of several
// Sin functions of the given frequencies, phases, and amplitudes, with noise
// optionally applied to each signal, and optionally applied to the data
// overall.
func (w *WaveParameters) GenerateSyntheticData(*Signal, error) {
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
