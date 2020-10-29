package rtml

import spinal.core._
import scala.collection.mutable.ArrayBuffer
import scala.io.Source

object Tools {
  def readmemh(path: String): Array[Bits] = {
    val buffer = new ArrayBuffer[Bits]
    for (line <- Source.fromFile(path).getLines) {
      val tokens: Array[String] = line.split("(//)").map(_.trim)
      if (tokens.length > 0 && tokens(0) != "") {
        val i = BigInt(tokens(0), 16)
        buffer.append(B(i))
      }
    }
    buffer.toArray
  }
}
