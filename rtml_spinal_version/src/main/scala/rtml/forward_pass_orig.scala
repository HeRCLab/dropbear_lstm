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
//
// --- Register Map ---
//
// Writeable AXI Registers:
// - 00 :: work_item_count
//         The "Work Item Count" tells the core how many items to fetch from
//         the BRAMs. Providing this parameter implicitly starts the core,
//         moving it from the "idle" state to the "working" state.
//
// Readable AXI Registers:
// - 00 :: work_item_count
// - 01 :: work_items_remaining
//         The number of iterations/fetches the core has remaining until it
//         enters the "done" state.
// - ?02 :: output_fifo_data
//         The FIFO of values once the core enters the "done" state.
// - 02 :: core_fsm_state
//         A register that tracks the state of the core, according to the
//         following FSM:
//          - 00: Idle
//          - 01: Working
//          - 02: Done
// - 03 :: outputs_valid
//         Bit-mapped register, showing 0/1 for not-done/done state across
//         the neurons. This register is mapped directly off the pipeline.
// - 04 .. 0B :: Output registers (contain the feedback register results)
package rtml

import scala.util.matching.Regex
import spinal.core._
import spinal.lib._
import spinal.lib.fsm._
import spinal.lib.bus.amba4.axi._
import spinal.lib.bus.bram._

// BRAM interface customized for use in this project.
// Note: Default BRAM from spinal.lib has different bit-width interfaces.
// Update: Default BRAM from spinal.lib only has differing bit-width on the write-enable.
/*case class BRAM(wordSize: Int, addrSize: Int) extends Bundle {
  val en   = out Bool
  val dout = in  Bits(wordSize bits)
  val din  = out Bits(wordSize bits)
  val we   = out Bool
  val addr = out UInt(addrSize bits)
  //val clk  = out Bool
  //val rst  = out Bool
}*/

object forward_pass_state2 extends Enumeration {
  type forward_pass_state = Value
  val Idle = Value(0)
  val Work = Value(1)
  val Done = Value(2)
}

case class ForwardPassConfig2(useFifo: Boolean, useFixPoint: Boolean)

// Hardware definition.
//class forward_pass(config: ForwardPassConfig) extends Component {
class forward_pass2() extends Component {
  val io = new Bundle {
    //val aclk                = in  Bool
    //val aclken              = in  Bool
    //val aresetn             = in  Bool
    // Vector of BRAM signal bundles. (Name patching happens later)
    //val bram = Vec(BRAM(wordSize=32, addrSize=10), 8)
    val bram = Vec(master(BRAM(BRAMConfig(dataWidth=32, addressWidth=10))), 8)
    // [Xilinx] FIFO control signals.
    //val fifo_full  = in  Bool
    //val fifo_out   = out Bits(32 bits)
    //val fifo_write = out Bool
    // [Xilinx] Accelerator Adapter handshake signals.
    //val ap_continue = in  Bool
    //val ap_done     = out Bool
    //val ap_idle     = out Bool
    //val ap_ready    = out Bool
    //val ap_start    = in  Bool
    // [Xilinx] External interface for Stream[Data] FIFO type.
    val result = master Stream(Bits(32 bits))
    // AXI R/W Interface.
    // We support reads/writes from several addresses.
    // 00
    //val control = Axi4(axiConfig)
    // [CUSTOM] Hand-rolled valid/ready/payload bus.
    val control = slave Stream(UInt(10 bits))
  }

  //----------------------------------------------------------------
  // BRAM fetch logic.
  // TODO: Add address incrementing logic.
  // TODO: Add logic to check that RAMs are ready for reading.
  // TODO: Add 1 cycle delay to ensure reads go through?
  val address_counter = Counter(0 to 1023)
  val input_counter = UInt(10 bits) // Counts from 0..num_inputs.

  //----------------------------------------------------------------
  // AXI registers and logic.
  val axi = new Area {
    val work_item_count = io.control.toFlow.toReg()
    val work_items_remaining = RegNext(work_item_count - address_counter)
    val fsm_state = Reg(UInt(2 bits)) init(0)
  }

  //val inputs_exhausted = num_inputs.willOverflow()
  for (ram <- io.bram) {
    ram.wrdata := 0
    ram.we   := B(0)
    ram.addr := address_counter.value
    ram.en   := True
    //ram.clk  := ClockDomain.current.clock
    //ram.rst  := ClockDomain.current.reset
  }

  // Read bits and stuff directly into registers, with no conversion logic
  val input_value = Reg(Bits(32 bits))//.keep() // TODO: Remove .keep()
  input_value := io.bram(0).rddata

  val weights = Vec(Reg(Bits(32 bits)), 7)//.keep() // TODO: Remove .keep()
  for (i <- 1 to 7) {
    weights(i-1) := io.bram(i).rddata
  }

  //----------------------------------------------------------------
  // MAC logic.

  // Pipelined MAC operation.
  val fsm = new StateMachine {
    val stateIdle = new State with EntryPoint
    val stateWork = new State
    val stateDone = new State

    val neuron_counter = Counter(0 to 6)
    val feedback_counter = Counter(0 to 6)
    val feedback = Vec(Reg(Bits(32 bits)) init(B"32'x0000"), 7) //.keep() // TODO: Remove .keep()

    // Choose floating-point or fixed-point pipeline based on config.
    //if (config.useFixPoint) {
      val mult_core = new fixedpoint_mul32()
      val add_core  = new fixedpoint_add32()
    val output_fifo = StreamFifo(Bits(32 bits), 7)
    //} else {
    //  val mult_core = new xilinx_fp_mul32()
    //  val add_core  = new xilinx_fp_add32()
    //}
    //mult_core.io.aclk    := io.clk
    //mult_core.io.aclken  := True
    //mult_core.io.aresetn := ~io.resetn
    //add_core.io.aclk    := io.clk
    //add_core.io.aclken  := True
    //add_core.io.aresetn := ~io.resetn

    // Default input signal assignments (mult core)
    mult_core.io.a.payload := input_value
    mult_core.io.b.payload := weights(neuron_counter)
    mult_core.io.a.valid := False
    mult_core.io.b.valid := False
    //mult_core.io.result.ready := False

    // Default input signal assignments (add core)
    add_core.io.b.payload := feedback(feedback_counter)
    add_core.io.b.valid := False
    // Connect output of multiplier core to adder core's a input.
    add_core.io.a <> mult_core.io.result
    add_core.io.result.ready := True // TODO: Give this a more precise halt condition later.

    // Connect output of adder core to output FIFO.
    output_fifo.io.push.payload := add_core.io.result.payload
    output_fifo.io.push.valid := False
    io.result << output_fifo.io.pop

    // Critical signals for knowing transition conditions.
    val inputs_exhausted = axi.work_items_remaining === 0
    val neurons_exhausted = neuron_counter.willOverflow
    val feedback_exhausted = feedback_counter.willOverflow
    val pipeline_complete = add_core.io.result.valid && neurons_exhausted && feedback_exhausted && inputs_exhausted

    // State state.
    stateIdle
      .whenIsActive {
        //axi.fsm_state := forward_pass_state.Idle
        // When AXI write to control register happens, transition: Idle -> Working
        when(io.control.valid) {
          goto(stateWork)
        }
      }

    stateWork
      .whenIsActive {
        //axi.fsm_state := forward_pass_state.Work
        // When multiplier core is ready, load a, b inputs.
        when(mult_core.io.a.ready && mult_core.io.b.ready) {
          mult_core.io.a.valid := True
          mult_core.io.b.valid := True
        }
        // When multiplier core is done, select next neuron.
        when(mult_core.io.result.valid) {
          neuron_counter.increment()
        }

        // When adder core is ready, load b input.
        when(add_core.io.b.ready) {
          add_core.io.b.valid := True
        }
        // When adder core is done, select next neuron's feedback slot.
        when(add_core.io.result.valid) {
          feedback(feedback_counter) := add_core.io.result.payload
          // TODO: Check that this is a sufficient termination condition.
          when(inputs_exhausted && output_fifo.io.push.ready) {
            output_fifo.io.push.payload := add_core.io.result.payload
            output_fifo.io.push.valid   := True
          } otherwise {
            output_fifo.io.push.payload := 0
            output_fifo.io.push.valid   := False
          }
          feedback_counter.increment()
        }

        // When inputs exhausted, and pipeline complete, transition: Working -> Done
        when(pipeline_complete) {
          goto(stateDone)
        }
      }

    stateDone
      // Mark all output regs as "done", transition back to idle?
      .whenIsActive {
        //axi.fsm_state := forward_pass_state.Done
        axi.work_item_count := 0

        address_counter.clear()
        neuron_counter.clear()
        feedback_counter.clear()

        // TODO: Check that FIFO has been emptied.
        goto(stateIdle)
      }


    // FIFO push logic.
    //when(inputs_exhausted && add_core.io.result.valid) {
    //  output_fifo.io.push := feedback(feedback_counter)
    //}

    // FIFO push on trigger
    //when() {
    //}

    //mult_core. // TODO: Wire this up to Streams.

    //for (i <- 0 to 6) {
    //  feedback(i) := ((input_value * weights(i)) + feedback(i)).truncated
    //}
  }

  //----------------------------------------------------------------
  // FIFO signals.
  //io.fifo_out   := 0     // TODO: Replace with output queue from MACs.
  //io.fifo_write := False // TODO: Replace with trigger once MACs are done.

  // Handshake signals.
  //io.ap_done  := False // TODO: Replace with check that internal FIFO is empty && MACs completed?
  //io.ap_idle  := True  // TODO: Replace with check
  //io.ap_ready := False // TODO: Replace with check. See docs for what this should do?

  //----------------------------------------------------------------
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

  // Execute the function renameBRAMIO after the creation of the component.
  addPrePopTask(() => renameBRAMIO())
  addPrePopTask(() => renameClockDomain())
}

// Generate the RTMLTop's Verilog.
/*object forward_pass_TopVerilog {
  def main(args: Array[String]) {
    SpinalVerilog(new forward_pass())
  }
}

// Generate the RTMLTop's VHDL.
object forward_pass_TopVhdl {
  def main(args: Array[String]) {
    SpinalVhdl(new forward_pass())
  }
}


// Define a custom SpinalHDL configuration with synchronous reset instead of the default asynchronous one. This configuration can be resued everywhere
object forward_pass_SpinalConfig extends SpinalConfig(defaultConfigForClockDomains = ClockDomainConfig(clockEdge = RISING, resetKind = ASYNC, resetActiveLevel = LOW, clockEnableActiveLevel = HIGH))

// Generate the MyTopLevel's Verilog using the above custom configuration.
object forward_pass_TopWithCustomConfig {
  def main(args: Array[String]) {
    forward_pass_SpinalConfig.generateVerilog(new forward_pass())
  }
}

// Generate the MyTopLevel's Verilog using the above custom configuration.
object forward_pass_TopWithCustomConfig2 {
  def main(args: Array[String]) {
    forward_pass_SpinalConfig.generateVerilog(ClockDomain.external("", withReset = true, withClockEnable = true) on new forward_pass())
  }
}


class add5 extends Component {
  val io = new Bundle {
    val clk    = in Bool
    val control = slave(Stream(UInt(32 bits)))
    val result_data = out UInt(32 bits)
  }
  val control_val = io.control.toFlow.toReg() init(0)
  io.result_data := control_val + 5
}*/
