// Databricks notebook source
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.Partitioner
import org.apache.spark.HashPartitioner
import org.apache.spark.rdd.RDD
import scala.reflect.ClassTag
import scala.collection.immutable.Map

// COMMAND ----------

// MAGIC %md
// MAGIC 0. Function Implementation

// COMMAND ----------

val NB_ITERATIONS = 50;

// COMMAND ----------

def PageRank(links: RDD[(String, Iterable[String])]) : Array[(String, Double)] = {
  var ranks = links.mapValues(v => 1.0)
  for (i <- 1 to NB_ITERATIONS)
  {
      val contributions = links.join(ranks).flatMap { 
        case (url, (links, rank)) => links.map(dest => (dest, rank / links.size)) 
      }
      ranks = contributions.reduceByKey((x, y) => x + y).mapValues(v => 0.15 + 0.85*v)
  }
  return ranks.collect
}

// COMMAND ----------

// MAGIC %md
// MAGIC I. provide a first implementation without any kind of optimisation, and show how it works on a simple A-B-C-D graph, by putting a cul-de-sac and a self edge on one node

// COMMAND ----------

//example with both self-edge & cul-de-sac : TO REMOVE but scores are displayed for the 4 nodes
val links = sc.parallelize(List(
  ("A", Iterable("B")),
  ("B", Iterable("B", "C")), //Node B has a self-loop and deadlock
  ("C", Iterable("D")),
  ("D", Iterable("A"))
))

// COMMAND ----------

// Function call: return a table of pairs with PageRank scores given for each link
val pageRank = PageRank(links)
//We can see that Node B has the highest PageRank score.

// COMMAND ----------

// MAGIC %md
// MAGIC Sliced Version :

// COMMAND ----------

// I. Baseline graph
val graph = List(
  ("A", List("B", "C")),
  ("B", List("C", "A")),
  ("C", List("D")),
  ("D", List("A"))
)

// II.1 Graph with 'cul-de-sac' (C points nothing)
val graphWithCulDeSac = List(
  ("A", List("B", "C")),
  ("B", List("C", "A")),
  ("C", List()), // C: cul-de-sac
  ("D", List("A", "C"))
)

// II.3 Graph with self edge (A points to itself)
val graphWithSelfEdge = List(
  ("A", List("A", "C")), 
  ("B", List("C", "A")),
  ("C", List("D")),
  ("D", List("A"))
)

// RDDs Creation
val links1 = sc.parallelize(graph.map { case (k, v) => (k, v.toIterable) })
val linksWithCulDeSac = sc.parallelize(graphWithCulDeSac.map { case (k, v) => (k, v.toIterable) })
val linksWithSelfEdge = sc.parallelize(graphWithSelfEdge.map { case (k, v) => (k, v.toIterable) })

// COMMAND ----------

//I. Analysis on a simple graph without optimisation
PageRank(links1) 

// COMMAND ----------

//II.1 Analysis on a simple graph with deadlock
PageRank(links_with_cul_de_sac) //D value = none

// COMMAND ----------

//II.2 Analysis on a simple graph with self-edge
PageRank(links_with_self_edge) //B value = none

// COMMAND ----------

// MAGIC %md
// MAGIC Performances Analysis

// COMMAND ----------

def PageRank_Optimised(graph: List[(String, List[String])]) : Array[(String, Double)] = {
  val links1 = sc.parallelize(graph).partitionBy(new HashPartitioner(8)).persist()
  var ranks = links1.mapValues(v => 1.0)
  for (i <- 1 to NB_ITERATIONS )
  {
      val contributions = links1.join(ranks).flatMap { 
        case (url, (links1, rank)) => links1.map(dest => (dest, rank / links1.size)) 
      }
      ranks = contributions.reduceByKey((x, y) => x + y).mapValues(v => 0.15 + 0.85*v)
  }
  return ranks.collect
}

// COMMAND ----------

// Conversion of the previous Optimised function into RDD
def PageRank_Optimised2(links: RDD[(String, Iterable[String])]): Array[(String, Double)] = {
  val links1 = links.partitionBy(new HashPartitioner(8)).persist()
  var ranks = links1.mapValues(v => 1.0)
  for (i <- 1 to NB_ITERATIONS) {
    val contributions = links1.join(ranks).flatMap {
      case (url, (links1, rank)) => links1.map(dest => (dest, rank / links1.size))
    }
    ranks = contributions.reduceByKey((x, y) => x + y).mapValues(v => 0.15 + 0.85 * v)
  }
  return ranks.collect
}

// COMMAND ----------

val pageRankOpti = PageRank_Optimised2(links)
//Graph WL above named "links"  scoring on all nodes given with improved performance in calculation duration

// COMMAND ----------

// MAGIC %md
// MAGIC III. Implementation adapted to a larger graph providing both the baseline implementation and the optimised one :

// COMMAND ----------

// Import CSV file
var data = sc.textFile("/FileStore/tables/201201.csv")
data.collect()

// COMMAND ----------

val header = data.first() //header extraction
data = data.filter(x => x != header) //removes the header
data.collect

// COMMAND ----------

//extraction of columns 4 and 5 values "ORIGIN_AIRPORT_ID","DEST_AIRPORT_ID" into a tuple, stored in new RDD data2
val data2 = data.map{ row =>
    val fields = row.split(",").map(_.trim) //trim function removes spaces at the beginning and end of each string.
    (fields(3), fields(4))
}

data2.collect
//header removed? put var instead of val?

// COMMAND ----------

//for each "ORIGIN_AIRPORT_ID" == key, we get the list of "DEST_AIRPORT_ID" == values & we get an array of tuples, where each tuple contains the origin airport and the list of corresponding destination airports
val final_data = data2.groupBy(_._1).mapValues(_.map(_._2))
final_data.collect

// COMMAND ----------

// MAGIC %md
// MAGIC III.1. Implementation of PageRank algo baseline implementation 

// COMMAND ----------

PageRank(final_data)

// COMMAND ----------

// MAGIC %md
// MAGIC III.2. Implementation optimised with partitioning tuning.

// COMMAND ----------

def PageRank_Optimised_Broadcast(links: RDD[(String, Iterable[String])]): Array[(String, Double)] = {
  val links1 = links.partitionBy(new HashPartitioner(8)).persist()
  var ranks = links1.mapValues(v => 1.0)
  
  for (i <- 1 to NB_ITERATIONS) {
    val contributions = links1.join(ranks).flatMap {
      case (url, (links1, rank)) => links1.map(dest => (dest, rank / links1.size))
    }
    ranks = contributions.reduceByKey((x, y) => x + y).mapValues(v => 0.15 + 0.85 * v)
  }
  return ranks.collect
}

// Convert final_data to an RDD of type (String, Iterable[String])
val finalDataRDD = final_data.map { case (k, v) => (k, v.toIterable) }

// Call PageRank_Optimised_Broadcast function on finalDataRDD
val pageRankResult = PageRank_Optimised_Broadcast(finalDataRDD)

