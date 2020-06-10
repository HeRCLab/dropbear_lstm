// mlp implements a simple multilayer perceptron based on this one:
// https://github.com/charlesdaniels/teaching-learning/tree/master/algorithms/backprop
package mlp

import (
	"fmt"
	"math"
	"math/rand"
)

type Layer struct {

	// Previous layer, nil for input layer
	Prev *Layer

	// Next layer, nil for output layer
	Next *Layer

	// Weight[j * Prev.TotalNeurons() + i] = weight for neuron j in THIS
	// layer coming in from the output of neuron i in the PREVIOUS layer.
	Weight []float64

	// Outputs for this layer
	Output []float64

	// The output after activation -- this is separate because we need the
	// pre-activation outputs for certain computations
	Activation []float64

	Delta []float64

	Bias []float64
}

func (l *Layer) TotalNeurons() int {
	return len(l.Output)
}

// Retrieve the weight associated with the link from prevNeuron in the previous
// layer, to thisNeuron in this layer.
func (l *Layer) GetWeight(thisNeuron, prevNeuron int) float64 {

	// input layer
	if l.Prev == nil {
		return 1.0
	} else {
		return l.Weight[thisNeuron*l.Prev.TotalNeurons()+prevNeuron]
	}
}

func (l *Layer) SetWeight(thisNeuron, prevNeuron int, newWeight float64) {
	if l.Prev == nil {
		return
	}
	l.Weight[thisNeuron*l.Prev.TotalNeurons()+prevNeuron] = newWeight
}

func NewLayer(size int, prev, next *Layer) *Layer {
	l := &Layer{
		Prev:       prev,
		Next:       next,
		Delta:      make([]float64, size),
		Output:     make([]float64, size),
		Activation: make([]float64, size),
		Bias:       make([]float64, size),
	}

	if prev != nil {
		l.Weight = make([]float64, size*(prev.TotalNeurons()))
		for i, _ := range l.Weight {
			l.Weight[i] = rand.Float64()
		}
	} else {
		// This is the inputlayer, so there are no weights
		l.Weight = nil
	}

	for i, _ := range l.Bias {
		// l.Bias[i] = rand.Float64()
		l.Bias[i] = 0
	}

	return l
}

type MLP struct {
	Layer []*Layer

	// Learning rate
	Alpha float64

	ActivationFunction func(float64) float64

	DerivActivationFunction func(float64) float64
}

func (nn *MLP) InputLayer() *Layer {
	return nn.Layer[0]
}

func (nn *MLP) OutputLayer() *Layer {
	return nn.Layer[len(nn.Layer)-1]
}

func NewMLP(alpha float64, g, gprime func(float64) float64, layerSizes ...int) *MLP {
	nn := &MLP{
		Layer:                   make([]*Layer, len(layerSizes)),
		Alpha:                   alpha,
		ActivationFunction:      g,
		DerivActivationFunction: gprime,
	}

	// generate the layers and their links back to the previous layers
	for i, v := range layerSizes {
		if i == 0 {
			nn.Layer[i] = NewLayer(v, nil, nil)
		} else {
			nn.Layer[i] = NewLayer(v, nn.Layer[i-1], nil)
		}

	}

	for i, _ := range nn.OutputLayer().Weight {
		nn.OutputLayer().Weight[i] = 0
	}

	// generate links to next layers
	for i, _ := range layerSizes {
		if i < len(layerSizes)-1 {
			nn.Layer[i].Next = nn.Layer[i+1]
		}
	}

	return nn

}

func Sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-1.0*x))
}

func SigmoidDeriv(x float64) float64 {
	return Sigmoid(x) * (1 - Sigmoid(x))
}

func Logit(x float64) float64 {
	return math.Log(x / (1 - x))
}

func Identity(x float64) float64 {
	return x
}

func Unit(x float64) float64 {
	return 1.0
}

func (nn *MLP) ForwardPass(input []float64) error {
	// The input must be the same size as the input layer, for obvious
	// reasons.
	if len(input) != nn.InputLayer().TotalNeurons() {
		return fmt.Errorf("Input vector size %d =/= input layer size %d",
			len(input), nn.Layer[0].TotalNeurons())
	}

	// copy input data into input layer outputs
	for i := 0; i < nn.InputLayer().TotalNeurons(); i++ {
		nn.InputLayer().Activation[i] = input[i]

		// in_j is not used for the input layer
		nn.InputLayer().Output[i] = 0
	}

	// consider remaining layers
	for l := 1; l < len(nn.Layer); l++ {
		layer := nn.Layer[l]
		for j := 0; j < layer.TotalNeurons(); j++ {
			// in_j ← ∑i w_i,j ai
			sum := 0.0 // sum is in_j
			for i := 0; i < layer.Prev.TotalNeurons(); i++ {
				// layer.Prev.Activation[j] is a_i
				// layer.GetWeight(j, i) w_i,j
				sum += layer.Prev.Activation[i] * layer.GetWeight(j, i)
			}

			// explicitly account for the bias, which is handled
			// implicitly in the Russel & Norvig text as weight
			sum += layer.Bias[j]

			// a_j ← g(in_j)
			layer.Activation[j] = nn.ActivationFunction(sum)

			// We also save in_j because we need it later for
			// computing the Δ values
			layer.Output[j] = sum
		}
	}

	return nil
}

// "output" refers to the expected output data
func (nn *MLP) BackwardPass(output []float64) error {
	if len(output) != nn.OutputLayer().TotalNeurons() {
		return fmt.Errorf("Output vector size %d =/= output layer size %d",
			len(output), nn.OutputLayer().TotalNeurons())
	}

	// calculate deltas for output layer
	for j := 0; j < nn.OutputLayer().TotalNeurons(); j++ {
		// Δ[j] ← g'(in_j) × (y_j - a_j)
		//
		// nn.OutputLayer().Delta[j] is Δ[j]
		//
		// g' is nn.DerivActivationFunction()
		//
		// in_j is layer.Output[j]
		//
		// a_j is layer.activation[j]
		//
		// y_j is output[j]
		//
		//    Notice that we have already asserted that the output
		//    layer and the output vector are the same size, so it is
		//    safe to assume we won't go out of bounds in output
		nn.OutputLayer().Delta[j] =
			nn.DerivActivationFunction(nn.OutputLayer().Output[j]) * (output[j] - nn.OutputLayer().Activation[j])
	}

	// and for remaining layers
	for l := len(nn.Layer) - 2; l >= 0; l-- {
		layer := nn.Layer[l]
		for i := 0; i < layer.TotalNeurons(); i++ {
			// Δ[i] ← g'(in_i) ∑j w_i,j Δ[j]
			//
			// layer.Delta[i] is Δ[i]
			//
			// layer.Next.GetWeight(j, i) w_i,j
			//
			//    The indices appear reversed because of how
			//    GetWeight() is written, j is the destination
			//    neuron on layer.Next, i is the neuron in this
			//    current layer.
			//
			// layer.Next.Delta[j] is Δ[j]
			//
			// layer.Output[i] is in_i
			layer.Delta[i] = 0
			for j := 0; j < layer.Next.TotalNeurons(); j++ {
				layer.Delta[i] += layer.Next.GetWeight(j, i) * layer.Next.Delta[j]
			}
			layer.Delta[i] *= nn.DerivActivationFunction(layer.Output[i])
		}
	}

	return nil
}

func (nn *MLP) UpdateWeights() {
	for l, layer := range nn.Layer {
		for i := 0; i < layer.TotalNeurons(); i++ {
			if l == 0 {
				// skip input layer, weights are always 1
				continue
			}
			for j := 0; j < layer.Prev.TotalNeurons(); j++ {
				// w_i,j ← w_i,j + α × a_i × Δ[j]
				//
				// layer.Activation[i] is a_i
				//
				// layer.Prev.Delta[j] is Δ[j]
				//
				// nn.Alpha is α
				layer.SetWeight(i, j,
					layer.GetWeight(i, j)+nn.Alpha*layer.Activation[i]*layer.Prev.Delta[j])

				// we also update the bias at this point,
				// keeping in mind that the a_i for the bias is
				// implied to be 1
				layer.Bias[i] += nn.Alpha * layer.Delta[i]
			}
		}
	}
}

func (nn *MLP) Train(input, output []float64) error {
	err := nn.ForwardPass(input)
	if err != nil {
		return err
	}

	err = nn.BackwardPass(output)
	if err != nil {
		return err
	}

	nn.UpdateWeights()

	return nil
}

func (nn *MLP) Predict(input []float64) ([]float64, error) {
	err := nn.ForwardPass(input)
	if err != nil {
		return nil, err
	}

	return nn.OutputLayer().Activation, nil
}
