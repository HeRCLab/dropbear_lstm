// mlp implements a simple multilayer perceptron based on this one:
// https://github.com/charlesdaniels/teaching-learning/tree/master/algorithms/backprop
package mlp

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/herclab/herc-file-formats/mlpx/go/mlpx"
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

	ActivationFunction      func(float64) float64
	DerivActivationFunction func(float64) float64

	mlpxActivationString string
	mlpxID               string
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
	target := thisNeuron*l.Prev.TotalNeurons() + prevNeuron
	l.Weight[target] = newWeight
}

func NewLayer(size int, prev, next *Layer, g, gprime func(float64) float64) *Layer {
	l := &Layer{
		Prev:                    prev,
		Next:                    next,
		Delta:                   make([]float64, size),
		Output:                  make([]float64, size),
		Activation:              make([]float64, size),
		Bias:                    make([]float64, size),
		ActivationFunction:      g,
		DerivActivationFunction: gprime,
	}

	if prev != nil {
		l.Weight = make([]float64, size*(prev.TotalNeurons()))
		for i, _ := range l.Weight {
			l.Weight[i] = rand.Float64()
		}
	} else {
		// This is the input layer, so we create a dummy list for
		// compatibility with MLPX
		l.Weight = make([]float64, size)
	}

	for i, _ := range l.Bias {
		l.Bias[i] = rand.Float64()
	}

	return l
}

type MLP struct {
	Layer []*Layer

	// Learning rate
	Alpha float64

	mlpx *mlpx.MLPX
}

// Returns appropriate functions based on an MLP activation string
func resolveMLPXActivation(a string) (func(float64) float64, func(float64) float64) {
	if a == "relu" {
		return ReLU, ReLUDeriv
	}

	if a == "sigmoid" {
		return Sigmoid, SigmoidDeriv
	}

	return Identity, Unit
}

// NewMLPFromMLPX instantiates a new MLP object from a given MLPX. The MLP
// will be isomorphic to the MLPX. This routine does not initialize the
// MLPX values at all.
//
// The value of Alpha is taken from the MLPX initialize snapshot.
//
// Also note that this does not attach the specified MLPX object to the created
// MLP, instead a new MLP is created.
func NewMLPFromMLPX(m *mlpx.MLPX) (*MLP, error) {
	err := m.Validate()
	if err != nil {
		return nil, err
	}

	// because we validated, we can pick any snapshot to extract the
	// topology
	layerSizes := make([]int, 0)
	gs := make([]func(float64) float64, 0)
	gprimes := make([]func(float64) float64, 0)
	snap, err := m.Initializer()
	if err != nil {
		return nil, err
	}

	// extract all the layer sizes and activation functions
	for _, layerID := range snap.SortedLayerIDs() {
		layer := snap.Layers[layerID]
		g, gp := resolveMLPXActivation(layer.ActivationFunction)

		layerSizes = append(layerSizes, layer.Neurons)
		gs = append(gs, g)
		gprimes = append(gprimes, gp)
	}

	if len(layerSizes) == 0 {
		return nil, fmt.Errorf("No layers in MLPX")
	}

	// note that we will overwrite g and gprime shortly
	nn := NewMLP(snap.Alpha, gs[0], gprimes[0], layerSizes...)

	// insert the activation functions
	for i, layerID := range snap.SortedLayerIDs() {
		layer := snap.Layers[layerID]

		nn.Layer[i].ActivationFunction = gs[i]
		nn.Layer[i].DerivActivationFunction = gprimes[i]
		nn.Layer[i].mlpxActivationString = layer.ActivationFunction
	}

	// get the initializer from the MLPX we just created inside of the MLP
	initializer, err := nn.mlpx.Initializer()
	if err != nil {
		return nil, err
	}

	// now copy over the initializer data
	for layerno, layerID := range snap.SortedLayerIDs() {
		layer := snap.Layers[layerID]

		if layer.Weights != nil {
			temp := make([]float64, len(*layer.Weights))
			initializer.Layers[layerID].Weights = &temp
			for i, v := range *layer.Weights {
				nn.Layer[layerno].Weight[i] = v
				(*initializer.Layers[layerID].Weights)[i] = v
			}
		}

		if layer.Outputs != nil {
			temp := make([]float64, len(*layer.Outputs))
			initializer.Layers[layerID].Outputs = &temp
			for i, v := range *layer.Outputs {
				nn.Layer[layerno].Output[i] = v
				(*initializer.Layers[layerID].Outputs)[i] = v
			}
		}

		if layer.Activations != nil {
			temp := make([]float64, len(*layer.Activations))
			initializer.Layers[layerID].Activations = &temp
			for i, v := range *layer.Activations {
				nn.Layer[layerno].Activation[i] = v
				(*initializer.Layers[layerID].Activations)[i] = v
			}
		}

		if layer.Deltas != nil {
			temp := make([]float64, len(*layer.Deltas))
			initializer.Layers[layerID].Deltas = &temp
			for i, v := range *layer.Deltas {
				nn.Layer[layerno].Delta[i] = v
				(*initializer.Layers[layerID].Deltas)[i] = v
			}
		}

		if layer.Biases != nil {
			temp := make([]float64, len(*layer.Biases))
			initializer.Layers[layerID].Biases = &temp
			for i, v := range *layer.Biases {
				nn.Layer[layerno].Bias[i] = v
				(*initializer.Layers[layerID].Biases)[i] = v
			}
		}

	}

	return nn, nil
}

func (nn *MLP) InputLayer() *Layer {
	return nn.Layer[0]
}

func (nn *MLP) OutputLayer() *Layer {
	return nn.Layer[len(nn.Layer)-1]
}

func NewMLP(alpha float64, g, gprime func(float64) float64, layerSizes ...int) *MLP {
	nn := &MLP{
		Layer: make([]*Layer, len(layerSizes)),
		Alpha: alpha,
		mlpx:  mlpx.MakeMLPX(),
	}

	snapid := nn.mlpx.NextSnapshotID()
	nn.mlpx.MustMakeSnapshot(snapid, alpha)

	// generate the layers and their links back to the previous layers
	for i, v := range layerSizes {
		if i == 0 {
			nn.Layer[i] = NewLayer(v, nil, nil, g, gprime)
			nn.Layer[i].mlpxID = "input"
			nn.mlpx.Snapshots[snapid].MustMakeLayer(
				"input",                // ID
				v,                      // neurons
				"",                     // pred
				fmt.Sprintf("%d", i+1)) // succ

		} else if i == len(layerSizes)-1 {
			// the output layer doesn't share our activation
			// function, since we want it to have full freedom of
			// range.
			nn.Layer[i] = NewLayer(v, nn.Layer[i-1], nil, Identity, Unit)
			nn.Layer[i].mlpxID = "output"

			nn.mlpx.Snapshots[snapid].MustMakeLayer(
				"output",               // ID
				v,                      // neurons
				fmt.Sprintf("%d", i-1), // pred
				"")                     // succ
		} else {
			pred := fmt.Sprintf("%d", i-1)
			succ := fmt.Sprintf("%d", i+1)
			if i-1 == 0 {
				pred = "input"
			}
			if i+1 == len(layerSizes)-1 {
				succ = "output"
			}

			nn.Layer[i] = NewLayer(v, nn.Layer[i-1], nil, g, gprime)
			nn.Layer[i].mlpxID = fmt.Sprintf("%d", i)
			nn.mlpx.Snapshots[snapid].MustMakeLayer(
				fmt.Sprintf("%d", i), // ID
				v,                    // neurons
				pred,                 // pred
				succ)                 // succ
		}
	}

	// generate links to next layers
	for i, _ := range layerSizes {
		if i < len(layerSizes)-1 {
			nn.Layer[i].Next = nn.Layer[i+1]
		}
	}

	return nn
}

// Snapshot takes a new snapshot using the embedded mlpx object
func (nn *MLP) Snapshot() error {

	// determine the pervious snapshot, we'll make ourself isomorphic to it
	snapids := nn.mlpx.SortedSnapshotIDs()
	prev := snapids[len(snapids)-1]

	// create the snapshot
	snapid := nn.mlpx.NextSnapshotID()
	err := nn.mlpx.MakeIsomorphicSnapshot(snapid, prev)
	if err != nil {
		return err
	}
	snapshot := nn.mlpx.Snapshots[snapid]

	for i, l := range nn.Layer {
		layer, ok := snapshot.Layers[l.mlpxID]
		if !ok {
			return fmt.Errorf("Layer %d mlpxID '%s' does not map to layer in attached MLPX", i, l.mlpxID)
		}

		layer.Neurons = l.TotalNeurons()

		// note we make deep copies, so they won't later change from
		// under us

		weights := make([]float64, len(l.Weight))
		copy(weights, l.Weight)
		layer.Weights = &weights

		outputs := make([]float64, len(l.Output))
		copy(weights, l.Output)
		layer.Outputs = &outputs

		activations := make([]float64, len(l.Activation))
		copy(activations, l.Activation)
		layer.Activations = &activations

		deltas := make([]float64, len(l.Delta))
		copy(deltas, l.Delta)
		layer.Deltas = &deltas

		biases := make([]float64, len(l.Bias))
		copy(biases, l.Bias)
		layer.Biases = &biases
	}

	return nil
}

func (nn *MLP) SaveSnapshot(path string) error {
	return nn.mlpx.WriteJSON(path)
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

func ReLU(x float64) float64 {
	if x > 0 {
		return x
	} else {
		return 0
	}
}

func ReLUDeriv(x float64) float64 {
	if x > 0 {
		return 1
	} else {
		return 0
	}
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
			layer.Activation[j] = layer.ActivationFunction(sum)

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
			nn.OutputLayer().DerivActivationFunction(nn.OutputLayer().Output[j]) * (output[j] - nn.OutputLayer().Activation[j])
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
			layer.Delta[i] *= layer.DerivActivationFunction(layer.Output[i])
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
				prevWeight := layer.GetWeight(i, j)
				newWeight := prevWeight + nn.Alpha*layer.Activation[i]*layer.Prev.Delta[j]
				layer.SetWeight(i, j, newWeight)

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
