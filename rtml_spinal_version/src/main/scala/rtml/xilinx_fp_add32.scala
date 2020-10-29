// Spinal HDL translation of the Forward Pass logic in the Vivado HLS code.
package rtml

import spinal.core._
import spinal.lib._

class xilinx_fp_add32 extends BlackBox {
  val io = new Bundle {
    val aclk                = in  Bool
    val aclken              = in  Bool
    val aresetn             = in  Bool
    val a                   = slave  Stream(Bits(32 bits))
    val b                   = slave  Stream(Bits(32 bits))
    val result              = master Stream(Bits(32 bits))
    val m_axis_result_tuser = out Bits(2 bits)
  }
  // Rename streams to match Xilinx conventions.
  rename_stream_to_axi(io.a,      "s_axis_a_t")
  rename_stream_to_axi(io.b,      "s_axis_b_t")
  rename_stream_to_axi(io.result, "m_axis_result_t")

  // Remove the 'io_' prefix in the generated RTL.
  noIoPrefix()

  // Help SpinalHDL understand the clock/enable/reset mapping for the Blackbox.
  mapClockDomain(clock=io.aclk, enable=io.aclken, reset=io.aresetn)

  // Renaming function, to match Xilinx naming scheme.
  def rename_stream_to_axi[T <: Data](s: Stream[T], prefix: String) = {
    s.valid.setName(prefix + "valid")
    s.ready.setName(prefix + "ready")
    s.payload.setName(prefix + "data")
  }
}

