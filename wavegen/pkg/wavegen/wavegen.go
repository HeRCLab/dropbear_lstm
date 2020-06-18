// Package wavegen is used for generating synthetic test data for the Dropbear
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
	S []float64

	T []float64

	// sample rate in Hz, note that this is set by the caller, this library
	// cannot guarantee the accuracy of this value
	SampleRate float64
}

// Sample represents a single sample from a Signal
type Sample struct {

	// The time component of the sample
	T float64

	// The value component of the sample
	S float64
}

// ValidateIndex ensures that the specified index can be retrieved from the
// signal, and generates an error if not.
func (s *Signal) ValidateIndex(i int) error {
	if len(s.S) != len(s.T) {
		return fmt.Errorf("Signal is corrupt, S array is length %d, but T array is length %T",
			len(s.S), len(s.T))
	}

	if i < 0 || i >= s.Size() {
		return fmt.Errorf("Index %d out of bounds for signal of length %d",
			i, s.Size())
	}

	return nil
}

// Size returns the number of samples which a Signal contains.
func (s *Signal) Size() int {
	return len(s.S)
}

// Index retrieves the ith sample value.
func (s *Signal) Index(i int) (Sample, error) {
	err := s.ValidateIndex(i)
	if err != nil {
		return Sample{}, err
	}

	return Sample{T: s.T[i], S: s.S[i]}, nil
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
	if s.Size() == 0 {
		return 0
	}

	if time < s.T[0] {
		return 0
	}

	i := 0
	for {
		t := s.T[i]

		if t == time {
			return i
		} else if overshoot && (t >= time) {
			return i
		} else if (t != time) && (i+1 >= s.Size()) {
			// prevent the next case from going out of bounds
			return s.Size() - 1
		} else if (!overshoot) && ((s.T[i+1]) > time) {
			return i
		}

		i++
	}
}

// Interpolate can be used to perform linear interpolation. It will compute a
// separate linear interpolation for each point in times, and return an
// appropriate sample for each.
func (s *Signal) Interpolate(times ...float64) []Sample {
	interpolated := make([]Sample, len(times))
	for i, t := range times {
		i0 := s.NearestIndex(t, false) // undershoot
		i1 := s.NearestIndex(t, true)  // overshoot

		s0 := s.MustIndex(i0)
		s1 := s.MustIndex(i1)

		// In the case where the requested value is exactly equal to a
		// known value, we can omit the interpolation and return it
		// directly. This would work equivalently with t1 and s1.
		if s0.T == t && s1.T == t {
			interpolated[i] = Sample{T: t, S: s0.S}
			continue
		}

		dist := math.Abs(s0.T - s1.T)
		d0 := math.Abs(s0.T - t)
		d1 := math.Abs(s1.T - t)

		if dist == 0 {
			interpolated[i] = Sample{T: t, S: s0.S}
			continue
		}

		interpolated[i] = Sample{T: t, S: (d0*s1.S)/(d0+d1) + (d1*s0.S)/(d0+d1)}
	}

	return interpolated
}

// GenerateSyntheticData generates a signal which is a composition of several
// Sin functions of the given frequencies, phases, and amplitudes, with noise
// optionally applied to each signal, and optionally applied to the data
// overall.
// func (w *WaveParameters) GenerateSyntheticData(*Signal, error) {
//         // number of points to generate
//         points := int(math.Ceil(sample_rate * time))
//         sample_period := 1.0 / sample_rate
//
//         if (len(freqs) != len(phases)) || (len(freqs) != len(amps)) || (len(phases) != len(amps)) {
//                 return nil, fmt.Errorf("freqs, phases, amps must be the same length")
//         }
//
//         sig := &Signal{
//                 T:          make([]float64, points),
//                 S:          make([]float64, points),
//                 SampleRate: sample_rate,
//         }
//
//         for i := 0; i < points; i++ {
//                 sig.S[i] = 0
//                 sig.T[i] = sample_period * float64(i)
//                 for j, freq := range freqs {
//                         sig.S[i] += amps[j] * math.Sin(2*math.Pi*freq*sig.T[i]+phases[j])
//                 }
//         }
//
//         return sig, nil
// }
