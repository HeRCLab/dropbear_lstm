// Spinal HDL design for the accelerator core.
//
// Design notes:
// - 8x external BRAMs total
//   - 1x BRAM  :: Inputs buffer.
//   - 7x BRAMs :: Per-neuron weights buffers.
// - Controlled via writable AXI registers.
// - Output FIFO via AXI.
// - OR: Output via AXI register range.
// - Each cycle, a new input can be read in, along with all of the weights that
//   correspond to it. This allows an embarassingly parallel MAC operation,
//   if desired. Current implementation is non-parallel, and pipelined.
// - 7x neurons can be processed by the accelerator at a time.
//   - Additional parallelism can be achieved by slapping down more
//     accelerators, SIMD-style.

package rtml

import scala.util.matching.Regex
import spinal.core._
import spinal.lib._
import spinal.lib.fsm._
import spinal.lib.bus.amba4.axi._
import spinal.lib.bus.bram._
import scala.collection.mutable.ArrayBuffer

// Enum used for logical states of the core.
// This helps with trigger conditions for streams, since Spinal doesn't
// support directly inspecting the FSM state.
object ForwardPassLogicalState extends SpinalEnum {
  val sIdle, sWork, sDone = newElement()
}


// Hardware definition.
//class forward_pass(config: ForwardPassConfig) extends Component {
class ForwardPass() extends Component {
  val io = new Bundle {
    val bram = Vec(master(BRAM(BRAMConfig(dataWidth=32, addressWidth=10))), 8)
    // [Xilinx] External interface for Stream[Data] FIFO type.
    val result = master Stream(Bits(32 bits))
    // [CUSTOM] Hand-rolled valid/ready/payload bus.
    val control = slave Stream(UInt(10 bits))
    val debug_state = out Bits(2 bits)
    val debug_input = out Bits(32 bits)
    val debug_weights = out Vec(Bits(32 bits), 7)
    val debug_accum = out Vec(Bits(32 bits), 7)
    val debug_accum_count = out UInt(10 bits)
    val debug_address_count = out UInt(10 bits)
  }

  def regToStream[T <: Data](source: T): Stream[T] = {
    val intermediate_stream = Stream(NoData)
    intermediate_stream.valid := True
    intermediate_stream.translateWith(source)
  }

  def regToFlow[T <: Data](source: T): Flow[T] = {
    val intermediate_flow = Flow(NoData)
    intermediate_flow.valid := True
    intermediate_flow.translateWith(source)
  }

  def all_inputs_ready(elements: Seq[fixedpoint_mac32]): Bool = {
    elements.map(_.io.a.ready).reduce(_ && _) &&
    elements.map(_.io.b.ready).reduce(_ && _) &&
    elements.map(_.io.c.ready).reduce(_ && _)
  }
  def all_results_ready(elements: Seq[fixedpoint_mac32]): Bool = elements.map(_.io.result.valid).reduce(_ && _)
  def all_results_fire(elements: Seq[fixedpoint_mac32]): Bool = elements.map(_.io.result.fire).reduce(_ && _)

  val logical_state = ForwardPassLogicalState()
  logical_state := ForwardPassLogicalState.sIdle // Default assignment. Manually overridden in state machine.

  // Generate MAC cores. Wiring comes later.
  val mac_cores = new ArrayBuffer[fixedpoint_mac32](7)
  for (i <- 0 to 6) {
    mac_cores += new fixedpoint_mac32(latency=1)
  }
  for (i <- 0 to 6) {
    mac_cores(i).io.result.ready := True
  }
  //----------------------------------------------------------------
  // BRAM fetch logic.
  // TODO: Add address incrementing logic.
  // TODO: Add logic to check that RAMs are ready for reading.
  // TODO: Add 1 cycle delay to ensure reads go through?
  val address_counter = Counter(0 to 1023)
  val input_counter = UInt(10 bits) // Counts from 0..nsum_inputs.
  val output_counter = Counter(0 to 6)
  val accumulator_counter = Counter(0 to 1023)

  val input_value = Bits(32 bits)
  val weights = Vec(Bits(32 bits), 7)
  //val accumulator = Vec(Reg(Bits(32 bits)) init(B"32'x0000"), 7)
  var mapped_accum_regs = new ArrayBuffer[Bits](7)
  for (i <- 0 to 6) {
    mapped_accum_regs += RegNextWhen(mac_cores(i).io.result.payload, mac_cores(i).io.result.fire, 0)
  }
  val accumulator = Vec(elements=mapped_accum_regs)

  // BRAM fetch.
  for (ram <- io.bram) {
    ram.wrdata := 0
    ram.we     := B(0)
    ram.addr   := address_counter.value
    ram.en     := True
  }

  //----------------------------------------------------------------
  // AXI registers and logic.
  io.control.ready := False  // Default assignment.
  val axi = new Area {
    val work_item_count = Reg(UInt(10 bits)) init(0)
    //val work_items_remaining = RegNext(work_item_count - address_counter)
  }

  // Stream from comb input value.
  input_value := io.bram(0).rddata
  val input_stream = regToStream(input_value)
  val input_streams = StreamFork(input_stream, portCount=7, synchronous=true)

  // Streams from comb weights values.
  for (i <- 1 to 7) {
    weights(i-1) := io.bram(i).rddata
  }
  val weights_streams = new ArrayBuffer[Stream[Bits]](7)
  for (i <- 0 to 6) {
    weights_streams += regToStream(weights(i))
  }

  // Streams for the accumulator reads.
  val accumulator_read_streams = new ArrayBuffer[Stream[Bits]](7)
  //val accumulator_write_streams = new ArrayBuffer[Stream[Bits]](7)
  for (i <- 0 to 6) {
    accumulator_read_streams += regToStream(accumulator(i))
    //accumulator_write_streams(i) += regToStream(accumulator(i)).takeWhen(all_results_ready(mac_cores))
  }

  // Wiring for MAC core inputs.
  for (i <- 0 to 6) {
    mac_cores(i).io.a << input_streams(i).haltWhen(logical_state === ForwardPassLogicalState.sDone)
    mac_cores(i).io.b << weights_streams(i).haltWhen(logical_state === ForwardPassLogicalState.sDone)
    mac_cores(i).io.c << accumulator_read_streams(i).haltWhen(logical_state === ForwardPassLogicalState.sDone)
    //accumulator_write_streams(i) << mac_cores(i).io.result
  }

  // Stream for the comb outputs selection mux.
  var output_fifo_stream = new StreamMux(Bits(32 bits), 7)
  output_fifo_stream.io.select := output_counter.value
  for (i <- 0 to 6) {
    output_fifo_stream.io.inputs(i) << accumulator_read_streams(i)
  }
  io.result << output_fifo_stream.io.output.continueWhen(logical_state === ForwardPassLogicalState.sDone)
  //io.result.payload := output_fifo_stream.io.output.payload // DEBUG
  //io.result.valid := False // DEBUG

  // DEBUG ---------------------------------------------------------
  io.debug_accum_count := accumulator_counter.value
  io.debug_address_count := address_counter.value
  io.debug_input := input_value
  io.debug_state := logical_state.asBits
  for (i <- 0 to 6) {
    io.debug_weights(i) := weights(i)
    io.debug_accum(i) := accumulator(i)
  }

  //----------------------------------------------------------------
  // State Machine
  val fsm = new StateMachine {
    val stateIdle = new State with EntryPoint
    val stateWork = new State
    val stateDone = new State
    stateIdle
      .onEntry {
        logical_state       := ForwardPassLogicalState.sIdle
        io.control.ready    := True
        axi.work_item_count := 0
        address_counter.clear()
        accumulator_counter.clear()
        output_counter.clear()
        for (i <- 0 to 6) { accumulator(i) := 0 }
      }
      .whenIsActive {
        logical_state    := ForwardPassLogicalState.sIdle
        io.control.ready := True
        when (io.control.valid) {
          axi.work_item_count := io.control.payload
          goto(stateWork)
        }
      }
    stateWork
      .onEntry {
        logical_state    := ForwardPassLogicalState.sWork
        io.control.ready := False
      }
      .whenIsActive {
        logical_state    := ForwardPassLogicalState.sWork
        when (all_results_ready(mac_cores))                  { address_counter.increment() }
        when (all_results_fire(mac_cores))                   { accumulator_counter.increment() }
        when (address_counter.value === axi.work_item_count) { goto(stateDone) }
      }
    stateDone
      .onEntry {
        logical_state    := ForwardPassLogicalState.sDone
        io.control.ready := False
        address_counter.clear()
        accumulator_counter.clear()
      }
      .whenIsActive {
        logical_state    := ForwardPassLogicalState.sDone
        when (io.result.ready)             { output_counter.increment() }
        when (output_counter.willOverflow) { goto(stateIdle) }
      }
  }


  //----------------------------------------------------------------
  // Rename signals

  // Remove io_ prefix from generated names.
  noIoPrefix()

  // Transform BRAM-related signals to look like Vivado's expected signals.
  private def renameBRAMIO(): Unit = {
    val vecname_regex: Regex = "[0-9a-zA-Z-]+_([0-9]+)_([0-9a-zA-Z- ]+)".r
    io.flatten.foreach(bt => {
      if(bt.getName().startsWith("bram")) {
        val original_name = bt.getName()
        // Use extractor pattern match for brevity.
        original_name match {
          case vecname_regex(n, signalName) => bt.setName(s"bram_${signalName}_$n")
        }
      }
    })
  }

  // Rename clock signals to match Xilinx defaults for streaming AXI cores.
  private def renameClockDomain(): Unit = {
    ClockDomain.current.clock.setName("aclk")
    ClockDomain.current.reset.setName("aresetn")
    if (ClockDomain.current.hasClockEnableSignal) {
      ClockDomain.current.clockEnable.setName("aclken")
    }
  }

  // Execute the renaming functions after the creation of the component.
  addPrePopTask(() => renameBRAMIO())
  addPrePopTask(() => renameClockDomain())
}

//----------------------------------------------------------------
// Generation targets

object ForwardPassTopVerilog {
  def main(args: Array[String]) {
    SpinalVerilog(new ForwardPass())
  }
}

object ForwardPassTopVhdl {
  def main(args: Array[String]) {
    SpinalVhdl(new ForwardPass())
  }
}

// Define a custom SpinalHDL configuration with synchronous reset instead of the default asynchronous one. This configuration can be resued everywhere
object ForwardPassSpinalConfig extends SpinalConfig(defaultConfigForClockDomains = ClockDomainConfig(clockEdge = RISING, resetKind = ASYNC, resetActiveLevel = LOW, clockEnableActiveLevel = HIGH))

// Generate the MyTopLevel's Verilog using the above custom configuration.
object ForwardPassTopWithCustomConfig {
  def main(args: Array[String]) {
    ForwardPassSpinalConfig.generateVerilog(new ForwardPass())
  }
}

// Generate the MyTopLevel's Verilog using the above custom configuration.
object ForwardPassTopWithCustomConfig2 {
  def main(args: Array[String]) {
    ForwardPassSpinalConfig.generateVerilog(ClockDomain.external("", withReset = true, withClockEnable = true) on new ForwardPass())
  }
}
