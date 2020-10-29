package rtml

import spinal.core._
import spinal.lib._

object ScoreboardInstructionState extends SpinalEnum {
  val scIssue, scReadOperands, sExecute, sWriteback = newElement()
}





// Homogeneous instruction type scoreboard. (Separate out non-homogeneous instructions beforehand.)
// Parameters:
// - Instruction set (an enum)
// - Register maps
// - Functional Units map (Maybe a fat Vec of signals? And blast it out with a for loop in the body?)
// - Operations enum for the functional units (optional?)
class Scoreboard(instruction_width: Int,
                 table_depth: Int,
                 register_count: Int,
                 functional_units: Int,
                 register_rw_rules: Seq[Seq[Int]]) extends Component {
  val io = new Bundle {
    val busy = Bits
    val op = Bits
  }

  // Table is expanded out to depth specified above.
}


