
import mlpx._
import rtml._

// Cite: https://stackoverflow.com/a/3183991
object MLPX2RTL {
  val usage = """
    Usage: mlpx2rtl [--min-size num] [--max-size num] filename
  """
  def main(args: Array[String]) {
    if (args.length == 0) println(usage)
    val arglist = args.toList
    type CLIOptionMap = Map[String, Any]

    def nextCLIOption(map : CLIOptionMap, list: List[String]) : CLIOptionMap = {
      def isSwitch(s : String) = (s(0) == '-')
      list match {
        case Nil => map
        case "--snapshot" :: value :: tail =>
                               nextCLIOption(map ++ Map("snapshot" -> value), tail)
        case "--min-size" :: value :: tail =>
                               nextCLIOption(map ++ Map("minsize" -> value.toInt), tail)
        case "--verbose" :: tail =>
                               nextCLIOption(map ++ Map("verbose" -> true), tail)
        case string :: opt2 :: tail if isSwitch(opt2) =>
                               nextCLIOption(map ++ Map("infile" -> string), list.tail)
        case string :: Nil =>  nextCLIOption(map ++ Map("infile" -> string), list.tail)
        case option :: tail => println("Unknown option "+option)
                               scala.sys.exit(1)
      }
    }
    var options = nextCLIOption(Map(), arglist)
    // Add default snapshot ID if no ID selected by user.
    if (!options.contains("snapshot")) {
      options = options ++ Map("snapshot" -> "initializer")
    }
    // Error if no input file provided.
    if (!options.contains("infile")) {
      println("Error: No input file specified.")
      println(usage)
      scala.sys.exit(1)
    }

    if (options.contains("verbose")) {
      println(options)
    }

    val mlpx_file = mlpx.Utils.fileRead(options("infile").asInstanceOf[String])
    val doc = new mlpx.MLPXDocument(mlpx_file)
    println(doc)
    println("Snapshots:")
    val snapshot_ids = doc.sortedSnapshotIDs
    // Cite: https://stackoverflow.com/a/6833653 (zipWithIndex example)
    for ((x, i) <- snapshot_ids.view.zipWithIndex) {
      if (i < snapshot_ids.size - 1) {
        println(s" ├─ $x")
      } else {
        println(s" └─ $x")
      }
    }
    println("Layers:")
    val layer_ids = doc.snapshots("initializer").layers.keys
    for ((x, i) <- layer_ids.view.zipWithIndex) {
      if (i < layer_ids.size - 1) {
        println(s" ├─ $x")
      } else {
        println(s" └─ $x")
      }
    }

    // Generate verilog for the MAC at will. (Just passing through the args for now.)
    // TODO: Proper selection between VHDL/Verilog.
    // TODO: Passing through configuration objects.
    rtml.RTMLTopVerilog.main(args, doc.initializer)
  }
}
