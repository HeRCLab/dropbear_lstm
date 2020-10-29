package rtml

import spinal.core._
import spinal.lib._
import mlpx._
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.Map

/*class Layer(incoming_units: Int,
            outgoing_units: Int,
            activation_func: Unit) extends Component {
}*/

// Systematically generates a pipelined neural network out of vector-multipliers.
// Assumes 'input' and 'output' layers exist.
class VectorMultMLP(mlpx_layers: mlpx.Snapshot) extends Component {
  val input_layer  = mlpx_layers.inputLayer
  val output_layer = mlpx_layers.outputLayer
  val io = new Bundle {
    val inputs  = in Vec(SFix(peak=7 exp, width=32 bits), input_layer.neurons)
    val outputs = out Vec(SFix(peak=7 exp, width=32 bits), output_layer.neurons)
  }

  var pipeline_layers = Map[String, Vec[SFix]]()
  var weights_layers = Map[String, Vec[SFix]]()

  // Create pipeline register banks as Maps of Reg[Vec]'s.
  for (k <- mlpx_layers.sortedLayerIDs) {
    val layer = mlpx_layers.layers(k)
    val weights = layer.weights.get
    pipeline_layers += (k -> Vec(Reg(SFix(peak=7 exp, width=32 bits)), layer.neurons))
    weights_layers  += (k -> Vec(Reg(SFix(peak=7 exp, width=32 bits)), weights.size))
    // Populate weights array.
    for (i <- 0 to weights.size - 1) {
      weights_layers(k)(i) := weights(i)
    }
  }

  def MAC(inputs: Vec[SFix], weights: Vec[SFix]): SFix = {
    val prods = Vec(SFix(peak=7 exp, width=32 bits), inputs.size)
    // The "all-in-one-shot" approach.
    for (i <- 0 to inputs.size - 1) {
      prods(i) := (inputs(i) * weights(i)).truncated
    }
    prods.reduceBalancedTree(_ + _)
  }

  // Skip input layer, iterate over remaining layers in sorted order.
  var prev_k: String = ""
  for (k <- mlpx_layers.sortedLayerIDs drop 1) {
    val layer = mlpx_layers.layers(k)
    val weights_values = layer.weights.get
    val biases = layer.biases

    //val weights_ram = Mem(SFix(peak=7 exp, width=32 bits), initialContent=weights_values)
    //val layer_outputs = Vec(SFix(peak=7 exp, width=32 bits), layer.neurons)

    // Pre-load weights into weights vector.

    for (i <- 0 to layer.neurons - 1) {
      if (prev_k == "") {
        pipeline_layers(k)(i) := MAC(io.inputs, weights_layers(k))
      } else {
        pipeline_layers(k)(i) := MAC(pipeline_layers(prev_k), weights_layers(k))
      }
    }
    prev_k = k
  }

  for (i <- 0 to output_layer.neurons - 1) {
    io.outputs(i) := pipeline_layers("output")(i)
  }
}

object VectorMultMLPTopVerilog {
  def main(args: Array[String], mlpx_layers: mlpx.Snapshot) {
    SpinalVerilog(new VectorMultMLP(mlpx_layers=mlpx_layers))
  }
}

object VectorMultMLPTopVhdl {
  def main(args: Array[String], mlpx_layers: mlpx.Snapshot) {
    SpinalVhdl(new VectorMultMLP(mlpx_layers=mlpx_layers))
  }
}
