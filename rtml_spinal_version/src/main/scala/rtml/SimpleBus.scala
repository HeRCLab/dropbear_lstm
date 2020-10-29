// From: https://github.com/saahm/MemoryIssue

package rtml

import spinal.core._
import spinal.lib.IMasterSlave
/*
Simple Bus for memory mapping
1 Bus Master, N Bus Slaves
 */
case class SimpleBus(dataWidth:Int, addressWidth:Int) extends Bundle with IMasterSlave {
  val SBvalid = Bool
  val SBready = Bool
  val SBaddress = UInt(addressWidth bits)
  val SBwdata = Bits(dataWidth bits)
  val SBrdata = Bits(dataWidth bits)
  val SBwrite = Bool

  override def asMaster(): Unit = {
    out(SBvalid, SBaddress, SBwdata, SBwrite)
    in(SBready, SBrdata)
  }
}
