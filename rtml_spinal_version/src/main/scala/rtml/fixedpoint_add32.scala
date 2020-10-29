// Spinal HDL translation of the Forward Pass logic in the Vivado HLS code.
package rtml

import spinal.core._
import spinal.lib._

class fixedpoint_add32(core_clock: ClockDomain = ClockDomain.current) extends Component {
  val io = new Bundle {
    //val aclk                = in  Bool
    //val aclken              = in  Bool
    //val aresetn             = in  Bool
    val a                   = slave  Stream(Bits(32 bits))
    val b                   = slave  Stream(Bits(32 bits))
    val result              = master Stream(Bits(32 bits))
    val m_axis_result_tuser = out Bits(2 bits)
  }

  noIoPrefix()

  // Configure the clock domain.
  /*val xilinxClockDomain = ClockDomain(
    clock  = io.aclk,
    reset  = io.aresetn,
    clockEnable = io.aclken,
    config = ClockDomainConfig(
      clockEdge        = RISING,
      resetKind        = ASYNC,
      resetActiveLevel = LOW
    )
  )*/
  /*val xilinxClockDomain = ClockDomain.external(
    name            = "xilinx",
    withReset       = true,
    withClockEnable = true,
    config = ClockDomainConfig(
      clockEdge        = RISING,
      resetKind        = ASYNC,
      resetActiveLevel = LOW
    )
  )*/

  val mainArea = new ClockingArea(core_clock) {
    //val bitsA  = io.a.toFlow.toReg()
    //val bitsB  = io.b.toFlow.toReg()
    val sfixA  = Reg(SFix(peak=7 exp, width=32 bits)) init(0)
    val sfixB  = Reg(SFix(peak=7 exp, width=32 bits)) init(0)
    val result = Reg(SFix(peak=7 exp, width=32 bits)) init(0)
    val ready_valid_tracker = Reg(Bits(3 bits)) init(0)

    io.a.ready := io.result.ready
    io.b.ready := io.result.ready
    ready_valid_tracker := (ready_valid_tracker |>> 1) | (B(io.result.ready & io.a.ready & io.b.ready) << 2)
    //io.result.payload := 0
    io.m_axis_result_tuser := 0

    // Load signed integers directly.
    sfixA.raw := io.a.payload.asSInt
    sfixB.raw := io.a.payload.asSInt

    // Set result every clock cycle.
    result := (sfixA + sfixB).truncated
    //io.result.payload := (io.a.payload.asSInt * io.b.payload.asSInt).asBits(31 downto 0)
    io.result.payload := result.asBits
    io.result.valid := ready_valid_tracker(2)
  }

  private def renameClockDomain(): Unit = {
    ClockDomain.current.clock.setName("aclk")
    ClockDomain.current.reset.setName("aresetn")
    if (ClockDomain.current.hasClockEnableSignal) {
      ClockDomain.current.clockEnable.setName("aclken")
    }
  }
  addPrePopTask(() => renameClockDomain())
}

object fixedpoint_add32_TopVerilog {
  def main(args: Array[String]) {
    SpinalVerilog(new fixedpoint_add32())
  }
}
