
package mlpx

import scala.io.Source


object Utils {
  def fileRead(filename: String): String = {
    val source = Source.fromFile(filename)
    val lines = try source.mkString finally source.close()
    lines
  }
}

// TODO: Return list
/*def fileReadLines(filename: String): List[String] {
  val bufferedSource = Source.fromFile(filename))
  val lines = try source.mkString finally source.close()
  lines
}*/
