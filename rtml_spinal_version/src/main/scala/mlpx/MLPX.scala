package mlpx

import scala.annotation.tailrec
import play.api.libs.json._

// Case classes are similar to C structs: just a data definition.
// These ones are carefully constructed to play nice with play-json.
case class Layer(predecessor: String,
                 successor: String,
                 neurons: Int,
                 weights: Option[List[Double]],
                 outputs: Option[List[Double]],
                 activations: Option[List[Double]],
                 deltas: Option[List[Double]],
                 biases: Option[List[Double]],
                 activation_function: Option[String]) {
  def size(): Int = {
    neurons
  }
}

case class Snapshot(layers: Map[String, Layer],
                    alpha: Option[Double]) {
  def input(): (String, Layer) = { ("input", layers("input")) }

  def inputLayer(): Layer = { layers("input") }

  def output(): (String, Layer) = { ("output", layers("output")) }

  def outputLayer(): Layer = { layers("output") }

  // Topologically-sorted list of IDs for layers. (Should be O(n^2))
  // We trace the doubly-linked list of layers by finding the root, and then
  // iterate across the successors until no more successors exist.
  def sortedLayerIDs(): List[String] = {
    // Find the root.
    val root = layers.find((kv) => kv._2.predecessor == "")
    val (rk, rv) = root.get

    // Tail-recursive function that walks the successors list efficiently.
    @tailrec
    def getSucc(acc:List[String],
                layer_map: Map[String, Layer],
                target: String): List[String] = {
      val root = layer_map(target)
      val next_root = root.successor
      if (next_root == "") { acc } // Base case
      else {
        getSucc(acc ++ List(next_root), layer_map, next_root)
      }
    }

    // Generate and return the list of sorted IDs.
    getSucc(List(rk), layers, rk)
  }

  // Sorted list of Layer objects.
  def sortedLayers(): List[Layer] = {
    for (k <- sortedLayerIDs) yield layers(k)
  }
}

// These companion object definitions allow automatic deserialization routines
// to be generated at compile time.
object Layer {
  implicit val layerReads = Json.reads[Layer]
}

object Snapshot {
  implicit val snapshotReads = Json.reads[Snapshot]
}

// Note: Some hackery required here due to weird schema key format.
class MLPXDocument(jsonSource: String) {
  private val _raw_json = Json.parse(jsonSource)
  var version = (_raw_json \ "schema").get(1).as[Double]
  var snapshots = (_raw_json \ "snapshots").as[Map[String, Snapshot]]

  // Returns [] if no snapshots present.
  // Returns ["initializer", "1", "2", ...] if IDs present.
  def sortedSnapshotIDs(): List[String] = {
    var initializer = List[String]()
    var numeric_ids = List[String]()
    var other_ids   = List[String]()
    for (k <- snapshots.keys) {
      if (k == "initializer") {
        initializer = List("initializer")
      // Cite: https://stackoverflow.com/a/9946141
      } else if (k.forall(_.isDigit)) {
        numeric_ids = numeric_ids :+ k // Append k to numeric ID list
      } else {
        other_ids = other_ids :+ k
      }
    }
    numeric_ids = numeric_ids.sortWith(_.toInt < _.toInt)
    initializer ++ numeric_ids ++ other_ids
  }

  // Sorted list of Snapshot objects.
  def sortedSnapshots(): List[Snapshot] = {
    for (k <- sortedSnapshotIDs) yield snapshots(k)
  }

  def initializer(): Snapshot = {
    snapshots("initializer")
  }

  // Topologically-sorted list of IDs for layers. (Should be O(n^2))
  // We trace the doubly-linked list of layers by finding the root, and then
  // iterate across the successors until no more successors exist.
  // Defaults to the initializer snapshot.
  def sortedLayerIDs(snapshot: String = "initializer"): List[String] = {
    snapshots(snapshot).sortedLayerIDs
  }

  // Sorted list of Layer objects.
  // Defaults to the initializer snapshot.
  def sortedLayers(snapshot: String = "initializer"): List[Layer] = {
    snapshots(snapshot).sortedLayers
  }
}
