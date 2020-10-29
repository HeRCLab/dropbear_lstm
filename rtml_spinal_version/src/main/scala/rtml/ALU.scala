package rtml

import spinal.core._
import spinal.lib._

class ALU extends Component {
  val io = new Bundle {
    val a      = in  UInt(32 bits)
    val b      = in  UInt(32 bits)
    val result = out UInt(32 bits)
  }

    val tA  = Reg(UInt(32 bits)) init(0)
    val tB  = Reg(UInt(32 bits)) init(0)
    val tC = Reg(UInt(32 bits)) init(0)
    tA := io.a
    tB := io.b
    tC := (tA * tB).resized
    io.result := tC

  // Rename clock signals to match Xilinx defaults for streaming AXI cores.
  /*private def renameClockDomain(): Unit = {
    ClockDomain.current.clock.setName("aclk")
    ClockDomain.current.reset.setName("aresetn")
    if (ClockDomain.current.hasClockEnableSignal) {
      ClockDomain.current.clockEnable.setName("aclken")
    }
  }

  // Execute the function renameBRAMIO after the creation of the component.
  addPrePopTask(() => renameClockDomain())*/
}


object ALUTopVerilog {
  def main(args: Array[String]) {
    SpinalVerilog(new ALU())
  }
}

// Define a custom SpinalHDL configuration with synchronous reset instead of the default asynchronous one. This configuration can be resued everywhere
object ALU_SpinalConfig extends SpinalConfig(defaultConfigForClockDomains = ClockDomainConfig(clockEdge = RISING, resetKind = ASYNC, resetActiveLevel = LOW, clockEnableActiveLevel = HIGH))

// Generate the MyTopLevel's Verilog using the above custom configuration.
object ALU_Verilog_TopWithCustomConfig {
  def main(args: Array[String]) {
    ALU_SpinalConfig.generateVerilog(new ALU())
  }
}

// Generate the MyTopLevel's Verilog using the above custom configuration.
object ALU_VHDL_TopWithCustomConfig {
  def main(args: Array[String]) {
    ALU_SpinalConfig.generateVhdl(new ALU())
  }
}
