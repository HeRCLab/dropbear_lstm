// Spinal HDL translation of the Forward Pass logic in the Vivado HLS code.
package rtml

import spinal.core._
import spinal.lib._

// Hardware definition.
// Reads in values for 1024 cycles, then produces a MAC value.
class Accelerator extends Component {
  val io = new Bundle {
    val input =  in SInt(32 bits)
    val weight = in SInt(32 bits)
    val enable = in Bool
    val output = out SInt(64 bits)
  }
  // 1x Input BRAM
  

  // 7x Weights BRAMs



  val inputs = Vec(Reg(SInt(32 bits)), 1024)
  val weights = Vec(Reg(SInt(32 bits)), 1024)
  val counter = Counter(0 to 1023)
  val ready = False

  val final_value = Reg(SInt(64 bits)) init(0) // Register to cache final value.

  // Loading happening on each cycle.
  inputs(counter) := io.input
  weights(counter) := io.weight
  counter.increment()
  when (counter.willOverflow) { ready := True }

  when (ready) {
    val prods = Vec(SInt(64 bits), 1024)
    // The "all-in-one-shot" approach.
    for (i <- 0 to 1023) {
      prods(i) := inputs(i) * weights(i)
    }

    // This creates a lopsided binary tree "spine" of additions, which is bad.
    //io.output := prods.reduce(_ + _)
    // This performs a balanced tree-structured reduction.
    final_value := prods.reduceBalancedTree(_ + _)
  }
  io.output := final_value
}
/*
// Generate the RTMLTop's Verilog.
object MACTopVerilog {
  def main(args: Array[String]) {
    SpinalVerilog(new MAC)
  }
}

// Generate the RTMLTop's VHDL.
object MACTopVhdl {
  def main(args: Array[String]) {
    SpinalVhdl(new MAC)
  }
}


// Define a custom SpinalHDL configuration with synchronous reset instead of the default asynchronous one. This configuration can be resued everywhere
object MACSpinalConfig extends SpinalConfig(defaultConfigForClockDomains = ClockDomainConfig(resetKind = SYNC))

// Generate the MyTopLevel's Verilog using the above custom configuration.
object MACTopWithCustomConfig {
  def main(args: Array[String]) {
    MACSpinalConfig.generateVerilog(new MAC)
  }
}*/
