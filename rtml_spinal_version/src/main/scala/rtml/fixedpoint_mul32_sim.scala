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

object TestFixedPointMul32 {
  def main(args: Array[String]): Unit = {

    // Compile the simulator.
    val compiled = SimConfig.withConfig(SpinalConfig(
      defaultConfigForClockDomains=ClockDomainConfig(
        clockEdge        = RISING,
        resetKind        = ASYNC,
        resetActiveLevel = LOW
      ))).withWave.allOptimisation.compile {
      val dut = new fixedpoint_mul32()
      dut
    }

    // Build reference model(s).
    val reference_payload = mutable.Queue[BigInt]()
    val actual_payload    = mutable.Queue[BigInt]()
    val reference_valid   = mutable.Queue[Boolean]()
    val actual_valid      = mutable.Queue[Boolean]()
    reference_payload.enqueue(0)
    reference_payload.enqueue(0)
    reference_valid.enqueue(false)
    reference_valid.enqueue(true)
    for (i <- 0 to 28) {
      val ab: BigInt      = (1 << i) & 0xFFFFFFFF
      val example: BigInt = ((ab * ab) >> 24) & 0x7FFFFFFF
      reference_payload.enqueue(example)
      reference_valid.enqueue(true)
    }

    // Run the simulation
    compiled.doSimUntilVoid{dut =>
      dut.clockDomain.forkStimulus(period = 10)
      // Simulation steps.
      for (i <- 0 to 30) {
        val a: BigInt = (1 << i) & 0xFFFFFFFF //Random.nextInt(256)
        val b: BigInt = (1 << i) & 0xFFFFFFFF //Random.nextInt(256)
        dut.io.a.payload #= a
        dut.io.a.valid   #= true
        dut.io.b.payload #= b
        dut.io.b.valid   #= true
        dut.io.result.ready #= true
        dut.clockDomain.waitSampling() // Ensures 1+ clock cycles occur before sampling outputs.
        //val clk = dut.clockDomain.clockSim.toBoolean
        val result: BigInt = dut.io.result.payload.toBigInt
        actual_payload.enqueue(result)
        actual_valid.enqueue(dut.io.result.valid.toBoolean)
        //println(s"Clk: $clk, Result: $result, Expected: ${(((a * b) >> 24) & 0x7FFFFFFF)}")
        println(s"Clk: $i, Expected: ${reference_payload(i)}, Result: $result, A: $a, B: $b")

        //sleep(0)
        //assert(result == ((a * b) & 0xFFFF), s"Result: $result, Expected: ${((a * b) & 0xFFFF)}")
      }
      // After running the DUT, compare reference to actual outputs for payload/valid.
      var i = 0
      for ((r, a) <- (reference_payload zip actual_payload)) {
        assert(r == a, s"Clk: $i, Expected: $r, Actual: $a (payload signal)")
        i += 1
      }
      i = 0
      for ((r, a) <- (reference_valid zip actual_valid)) {
        assert(r == a, s"Clk: $i, Expected: $r, Actual: $a (valid signal)")
        i += 1
      }
      simSuccess()
    }
  }
}


