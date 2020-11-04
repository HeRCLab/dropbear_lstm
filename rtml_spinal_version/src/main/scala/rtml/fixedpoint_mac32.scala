// Spinal HDL translation of the Forward Pass logic in the Vivado HLS code.
package rtml

import spinal.core._
import spinal.lib._


class fixedpoint_mac32(core_clock: ClockDomain = ClockDomain.current, latency: Int = 1) extends Component {
  val io = new Bundle {
    //val aclk                = in  Bool
    //val aclken              = in  Bool
    //val aresetn             = in  Bool
    val a                   = slave (Stream(Bits(32 bits)))
    val b                   = slave (Stream(Bits(32 bits)))
    val c                   = slave (Stream(Bits(32 bits)))
    val result              = master(Stream(Bits(32 bits)))
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
    name            = "aclk",
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
    val mac_result = StreamFifo(Bits(32 bits), latency-1)
    val valid_tracker = Reg(Bits(latency bits)) init(0)

    // Load signed integers directly.
    io.a.ready := mac_result.io.push.ready
    io.b.ready := mac_result.io.push.ready
    io.c.ready := mac_result.io.push.ready
    mac_result.io.push.payload := (((io.a.payload.asSInt * io.b.payload.asSInt) >> 24).resized + io.c.payload.asSInt).asBits
    mac_result.io.push.valid := False

    //io.result.payload := 0
    io.m_axis_result_tuser := 0

    when (io.a.valid && io.b.valid && io.c.valid) {
      when (mac_result.io.push.ready) {
        mac_result.io.push.valid := True
      }
    }
    //io.result.payload := (io.a.payload.asSInt * io.b.payload.asSInt).asBits(31 downto 0)
    io.result.valid := mac_result.io.pop.valid
    io.result.payload := mac_result.io.pop.payload
    mac_result.io.pop.ready := io.result.ready
  }

  //xilinxClockDomain.clock.setName("aclk")
  //xilinxClockDomain.reset.setName("aresetn")
  //xilinxClockDomain.clockEnable.setName("aclken")

  private def renameClockDomain(): Unit = {
    ClockDomain.current.clock.setName("aclk")
    ClockDomain.current.reset.setName("aresetn")
    if (ClockDomain.current.hasClockEnableSignal) {
      ClockDomain.current.clockEnable.setName("aclken")
    }
  }
  addPrePopTask(() => renameClockDomain())
  /*private def renameClockDomain(): Unit = {
    xilinxClockDomain.clock.setName("aclk")
    xilinxClockDomain.reset.setName("aresetn")
    xilinxClockDomain.clockEnable.setName("aclken")
  }
  addPrePopTask(() => renameClockDomain())*/
}

object fixedpoint_mac32_TopVerilog {
  def main(args: Array[String]) {
    SpinalVerilog(new fixedpoint_mac32())
  }
}
