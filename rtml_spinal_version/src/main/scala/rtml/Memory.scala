// From: https://github.com/saahm/MemoryIssue

package rtml

import spinal.core._
import spinal.lib.slave


class Memory(memoryWidth : Int, wordCount : Int, initFile : String) extends Component {
  val io = new Bundle {
    val sb = slave(SimpleBus(32,32))
    val sel = in Bool
  }

  val mem = new Mem(Bits(memoryWidth bits), wordCount)
  val rdy = Reg(Bool) init(False)
  val read = io.sb.SBvalid && io.sel && !io.sb.SBwrite
  val write = io.sb.SBvalid && io.sel && io.sb.SBwrite
  val intDBG = Bits(memoryWidth bits)

  if(initFile.isEmpty){
    println("Init RAM with 0")
    mem.init(List.fill(wordCount)(B(0, memoryWidth bits)))
  }
  else {
    println("Init RAM with initFile")
    mem.init(Tools.readmemh(initFile))
    //mem.init(List.fill(wordCount)(B(0,memoryWidth bits)))
    //mem.init(Tools.readBytesFromTxt(initFile))
  }

  intDBG := mem(B(0,10 bits).asUInt).resized

//  io.sb.SBrdata := 0

  mem.write(
    enable = write,
    address = io.sb.SBaddress(log2Up(wordCount)-1 downto 0),
    data = io.sb.SBwdata
  )
  io.sb.SBrdata := mem.readSync(
    enable = read,
    address = io.sb.SBaddress(log2Up(wordCount)-1 downto 0)
  )
  rdy := False
  when((read | write) & io.sel){
    rdy := True
  }
  io.sb.SBready := rdy
}
