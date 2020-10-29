// Spinal HDL translation of the Forward Pass logic in the Vivado HLS code.
package rtml

import spinal.core._
import spinal.lib._
import mlpx._

// Hardware definition.
class RTMLTop(weights_init: Option[Array[Bits]] = None,
              inputs_init: Option[Array[Bits]] = None,
              mlpx_layers: mlpx.Snapshot) extends Component {
  val input_layer  = mlpx_layers.inputLayer
  val output_layer = mlpx_layers.outputLayer

  val io = new Bundle {
    val valid = out Bool
    //val output =  out SFix(peak=7 exp, width=32 bits)
    val inputs  = in Vec(SFix(peak=7 exp, width=32 bits), input_layer.neurons)
    val outputs = out Vec(SFix(peak=7 exp, width=32 bits), output_layer.neurons)
  }

  val output_value = Reg(SFix(peak=7 exp, width=32 bits)) init(0)
  val valid = Reg(Bool) init(false)

  //val mac_core = new MAC

  /*val inputs = new Area {
    val mem = inputs_init match {
      case Some(contents) => Mem(Bits(32 bits), initialContent = contents)
      case None => Mem(Bits(32 bits), wordCount = 1024)
    }
  }

  val weights = new Area {
    val mem = weights_init match {
      case Some(contents) => Mem(Bits(32 bits), initialContent = contents)
      case None => Mem(Bits(32 bits), wordCount = 1024)
    }
  }*/
  val mlp = new Area {
    val vmmlp = new VectorMultMLP(mlpx_layers)
    vmmlp.io.inputs := io.inputs
    io.outputs := vmmlp.io.outputs
  }

  // Area to handle feeding values into the MAC core.
  // Also handles marking outputs as valid.
  /*val mac_loader = new Area {
    val counter = Counter(0 to 1023)
    // Load a value from RAM into the MAC core every cycle.
    mac_core.io.input := S(inputs.mem.readSync(counter))
    mac_core.io.weight := S(weights.mem.readSync(counter))
    //}
    //for (i <- 0 to 1023) { // TODO
    //  mac_core.io.inputs(counter) := 0
    //  mac_core.io.weights(counter) := 0
    //}
    when(counter.willOverflowIfInc) {
      mac_core.io.enable := True
      io.valid := True
    }.otherwise {
      mac_core.io.enable := False
      io.valid := False
    }
    output_value := mac_core.io.output
    counter.increment()
  }*/

  // Generate vector multiplier layers that are connected by pipeline regs.
  //for (layer <- mlpx_layers) {
  //}

  // This output catches the end value coming out of the MAC core.
  //io.output := vmmlp.outputs
  io.valid := True
}

// Generate the RTMLTop's Verilog.
object RTMLTopVerilog {
  def main(args: Array[String], mlpx_layers: mlpx.Snapshot) {
    SpinalVerilog(new RTMLTop(mlpx_layers=mlpx_layers))
  }
}

// Generate the RTMLTop's VHDL.
object RTMLTopVhdl {
  def main(args: Array[String], mlpx_layers: mlpx.Snapshot) {
    SpinalVhdl(new RTMLTop(mlpx_layers=mlpx_layers))
  }
}


// Define a custom SpinalHDL configuration with synchronous reset instead of the default asynchronous one. This configuration can be resued everywhere
object RTMLSpinalConfig extends SpinalConfig(defaultConfigForClockDomains = ClockDomainConfig(resetKind = SYNC))

// Generate the MyTopLevel's Verilog using the above custom configuration.
object RTMLTopWithCustomConfig {
  def main(args: Array[String], mlpx_layers: mlpx.Snapshot) {
    RTMLSpinalConfig.generateVerilog(new RTMLTop(mlpx_layers=mlpx_layers))
  }
}
