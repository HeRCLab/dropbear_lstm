package rtml

import spinal.core._
import spinal.lib._

// We use a 2D list to allow some swaps to happen simultaneously.
object SortingNetwork {
  def default_swap_schedule(): List[List[(Int, Int)]] = {
    List(
      List((0, 2), (1, 3)),
      List((0, 1), (2, 3)),
      List((1, 2))
    )
  }
}

class SortingNetwork(swaps: List[List[(Int, Int)]]) extends Component {
  val io = new Bundle {
    val inputs  = in Vec(SInt(32 bits), 4)
    val outputs = out Vec(SInt(32 bits), 4)
  }

  var intermediates = List(io.inputs)

  for (layer <- swaps) {
    val next_layer = Vec(SInt(32 bits), 4)
    // Make initial assignments between "layers" of the network.
    for (i <- 0 to next_layer.size - 1) {
      next_layer(i) := intermediates.last(i)
    }
    // Swapping assignments "win" because they happen last for each layer.
    for ((a, b) <- layer) {
      // The "when" block will force some signals to become `reg` type in the Verilog.
      // This is okay, since they will still synthesize as combinational logic.
      when(intermediates.last(b) > intermediates.last(a)) {
        next_layer(a) := intermediates.last(b)
        next_layer(b) := intermediates.last(a)
      }
    }
    // Append the vector of signals for this layer.
    intermediates = intermediates ++ List(next_layer)
  }

  // Hook up the outputs.
  for (i <- 0 to io.outputs.size - 1) {
    io.outputs(i) := intermediates.last(i)
  }
}

object SortingNetworkTopVerilog {
  def main(args: Array[String]) {
    SpinalVerilog(new SortingNetwork(SortingNetwork.default_swap_schedule()))
  }
}

object SortingNetworkTopVhdl {
  def main(args: Array[String]) {
    SpinalVhdl(new SortingNetwork(SortingNetwork.default_swap_schedule()))
  }
}
