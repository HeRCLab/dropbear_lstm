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


// Hardware definition.
//class forward_pass(config: ForwardPassConfig) extends Component {
class ForwardPass() extends Component {
  val io = new Bundle {
    val bram = Vec(master(BRAM(BRAMConfig(dataWidth=32, addressWidth=10))), 8)
    // [Xilinx] External interface for Stream[Data] FIFO type.
    val result = master Stream(Bits(32 bits))
    // [CUSTOM] Hand-rolled valid/ready/payload bus.
    val control = slave Stream(UInt(10 bits))
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

  // Generate MAC cores. Wiring comes later.
  val mac_cores = new ArrayBuffer[fixedpoint_mac32](7)
  for (i <- 0 to 6) {
    mac_cores += new fixedpoint_mac32(latency=1)
  }

  //----------------------------------------------------------------
  // BRAM fetch logic.
  // TODO: Add address incrementing logic.
  // TODO: Add logic to check that RAMs are ready for reading.
  // TODO: Add 1 cycle delay to ensure reads go through?
  val address_counter = Counter(0 to 1023)
  val input_counter = UInt(10 bits) // Counts from 0..nsum_inputs.

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
    val work_items_remaining = RegNext(work_item_count - address_counter)
    val fsm_state = Reg(UInt(2 bits)) init(0)
  }

  val output_counter = Counter(0 to 6)
  val accumulator_counter = Counter(0 to 1023)
  val accumulator = Vec(Reg(Bits(32 bits)) init(B"32'x0000"), 7) //.keep() // TODO: Remove .keep()

  // Critical signals for knowing transition conditions.
  //val inputs_exhausted = axi.work_items_remaining === 0 // TODO: Might add a condition of not being in idle state.
  val inputs_exhausted = axi.work_items_remaining === 0 // TODO: Might add a condition of not being in idle state.
  val accumulator_finished = accumulator_counter.value === axi.work_item_count
  val in_done_state = False
  val inputs_exhausted_delayed = Delay(inputs_exhausted, 1) // 1 cycle after exhaustion of inputs.
  val pipeline_complete = inputs_exhausted && accumulator_finished

  val accumulator_out_streams = new ArrayBuffer[Stream[Bits]](7)
  val accumulator_in_streams = new ArrayBuffer[Stream[Bits]](7)
  var output_fifo_stream = new StreamMux(Bits(32 bits), 7)
  output_fifo_stream.io.select := output_counter.value
  for (i <- 0 to 6) {
    accumulator_out_streams += regToStream(accumulator(i)).takeWhen(in_done_state)
    accumulator_in_streams += regToStream(accumulator(i)).haltWhen(inputs_exhausted)
    output_fifo_stream.io.inputs(i) << accumulator_out_streams(i)
  }
  io.result << output_fifo_stream.io.output

  // Read bits and stuff directly into registers, with no conversion logic
  val input_value = Bits(32 bits)
  input_value := io.bram(0).rddata
  val input_stream = regToStream(input_value).haltWhen(inputs_exhausted)
  val input_streams = StreamFork(input_stream, portCount=7, synchronous=true)

  val weights = Vec(Bits(32 bits), 7)
  for (i <- 1 to 7) {
    weights(i-1) := io.bram(i).rddata
  }
  val weights_streams = new ArrayBuffer[Stream[Bits]](7)
  for (i <- 0 to 6) {
    weights_streams += regToStream(weights(i)).haltWhen(inputs_exhausted)
    // Alternately, maybe try takeWhen, or clearValidWhen?
  }

  // DEBUG
  io.debug_accum_count := accumulator_counter.value
  io.debug_address_count := address_counter.value
  io.debug_input := input_value
  for (i <- 0 to 6) {
    io.debug_weights(i) := weights(i)
    io.debug_accum(i) := accumulator(i)
  }

  // Streams -> MAC Core
  // MAC Core -> Direct Reg write
  for (i <- 0 to 6) {
    mac_cores(i).io.a << input_streams(i)
    mac_cores(i).io.b << weights_streams(i)
    mac_cores(i).io.c << accumulator_in_streams(i)
    mac_cores(i).io.result.ready := !in_done_state // TODO: Done state && reads available.
  }

  //----------------------------------------------------------------
  // State Machine

  val fsm = new StateMachine {
    val stateIdle = new State with EntryPoint
    val stateWork = new State
    val stateDone = new State

    stateIdle
      .whenIsActive {
        io.control.ready := True
        in_done_state := False

        output_counter.clear()
        accumulator_counter.clear()
        //axi.fsm_state := forward_pass_state.Idle
        // When AXI write to control register happens, transition: Idle -> Working
        when(io.control.valid) {
          axi.work_item_count := io.control.payload
          goto(stateWork)
        }
      }

    stateWork
      .whenIsActive {
        // Join on all inputs being ready before advancing.
        when (mac_cores.map(_.io.a.ready).reduce(_ && _) &&
              mac_cores.map(_.io.b.ready).reduce(_ && _) &&
              mac_cores.map(_.io.c.ready).reduce(_ && _)) {
          address_counter.increment()
        }
        when (mac_cores.map(_.io.result.valid).reduce(_ && _)) {
          for (i <- 0 to 6) {
            accumulator(i) := mac_cores(i).io.result.payload
          }
          accumulator_counter.increment()
        }

        // When inputs exhausted, and pipeline complete, transition: Working -> Done
        when(pipeline_complete) {
          goto(stateDone)
        }
      }

    stateDone
      .whenIsActive {
        //axi.fsm_state := forward_pass_state.Done
        axi.work_item_count := 0
        in_done_state := True
        address_counter.clear()

        when (io.result.ready && io.result.valid) {
          output_counter.increment()
        }

        // Go back to Idle state when FIFO is emptied.
        when (output_counter.willOverflow) {
          goto(stateIdle)
        }
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
