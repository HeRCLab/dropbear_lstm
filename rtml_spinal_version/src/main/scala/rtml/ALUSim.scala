// Simulation testbench for the multiplication core.
// The core takes 2 clock cycles to produce an output, so all results have to
// be time-shifted by 2 places in the output queue.
// The ready/valid checking is minimal at this time.
package rtml

import spinal.core._
import spinal.sim._
import spinal.core.sim._
import spinal.lib._

import scala.collection.mutable
import scala.util.Random

object TestALU {
  def main(args: Array[String]): Unit = {

    // Compile the simulator
    val compiled = SimConfig.withWave.allOptimisation.compile {
      val dut = new ALU()
      dut
    }

    // Build reference model(s).
    val reference_payload = mutable.Queue[BigInt]()
    val actual_payload    = mutable.Queue[BigInt]()
    reference_payload.enqueue(0)
    reference_payload.enqueue(0)
    for (i <- 0 to 28) {
      val ab: BigInt      = (1 << i) & 0xFFFFFFFF
      val example: BigInt = (ab * ab) & 0x7FFFFFFF
      reference_payload.enqueue(example)
    }

    // Run the simulation
    compiled.doSimUntilVoid{dut =>
      dut.clockDomain.forkStimulus(period = 10)
      // Simulation steps.
      for (i <- 0 to 30) {
        val a: BigInt = (1 << i) & 0xFFFF //Random.nextInt(256)
        val b: BigInt = (1 << i) & 0xFFFF //Random.nextInt(256)
        dut.io.a #= a
        dut.io.b #= b
        dut.clockDomain.waitSampling() // Ensures 1+ clock cycles occur before sampling outputs.
        val clk = dut.clockDomain.clockSim.toBoolean // !!! ERROR HERE !!!
        val result: BigInt = dut.io.result.toBigInt
        actual_payload.enqueue(result)
        println(s"Clk: $i, Expected: ${reference_payload(i)}, Result: $result, A: $a, B: $b")
      }
      // After running the DUT, compare reference to actual outputs for payload/valid.
      var i = 0
      for ((r, a) <- (reference_payload zip actual_payload)) {
        assert(r == a, s"Clk: $i, Expected: $r, Actual: $a (payload signal)")
        i += 1
      }
      i = 0
      simSuccess()
    }
  }
}


