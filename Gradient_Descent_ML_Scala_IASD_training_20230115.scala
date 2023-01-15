// Databricks notebook source
import org.apache.spark.rdd._

// COMMAND ----------

//Identification of the function type with the "size":
def essai(vec1: Array[Double], vec2: Array[Double]) =
  (for (i <-0 to vec1.size-1) yield (vec1(i) * vec2(i)))

//Same with "length"
def essai_2(vec1:Array[Double], vec2:Array[Double])=
  (for (i <-0 to vec1.length-1) yield (vec1(i) * vec2(i)))

// COMMAND ----------

//Another try
val Y = (1.0 to 20.0).by(2).toArray
val X = (1.0 to 30.0).by(3).toArray
essai(X,Y)
essai_2(X,Y)

// COMMAND ----------

//test to check empty array filling and how the data is returned:
def essai_1(vec1:Array[Double], vec2:Array[Double])={
  var result = Array.fill(vec1.size)(0.0)
  (for (i<-0 to vec1.size-1) result(i) = vec1(i) * vec2(i) )
  result  
}

// COMMAND ----------

//First method analysis :
def scalar_product_analysis(vec1: Array[Double], vec2: Array[Double]) =
  (for (i <-0 to vec1.size-1) yield (vec1(i) * vec2(i)))

// COMMAND ----------

scalar_product_analysis(X,Y)

// COMMAND ----------

// method 1 scalar product yield == return in lazy mode
//yield(vec1 * vec2) produces an array 
def prods(vec1: Array[Double], vec2: Array[Double]) =
  (for (i <-0 to vec1.size-1) yield (vec1(i) * vec2(i))).reduce( (a,b) => a+b )
prods(X,Y)

// COMMAND ----------

//method 2 scalar product // the zip operator concatenates the two arrays Array(a,b,c) zip Array(e,f,g) -> Array((a,e),(b,f),(c,g)) 
// "case" identifies  tuples in order to apply the function defined in the map
def prods2(vec1: Array[Double],vec2: Array[Double]) =
  (vec1 zip vec2).map{case (a,b) => a*b}.reduce( (a,b) => a+b)
prods2(X,Y)

// COMMAND ----------

//Scalar product method 1
def prodbyscal(scalar: Double, vec1: Array[Double]) =
  (for (i <-0 to vec1.size-1) yield (vec1(i) * scalar)).toArray[Double]
prodbyscal(5.0,X)

// COMMAND ----------

//Scalar product method 2 :
def prodbyscal2(scalar: Double, vec1: Array[Double]) =
  vec1.map(x => scalar * x).toArray[Double]
prodbyscal2(5.0,X)

// COMMAND ----------

//vector subtraction1 :
def subvec(vec1: Array[Double], vec2: Array[Double]) =
  (for (i <-0 to vec1.size-1) yield (vec1(i)-vec2(i)))
subvec(X,Y)

// COMMAND ----------

//vector subtraction2 :
// Zip operator concatenates 2 arrays Array(a,b,c) zip Array(e,f,g) -> Array((a,e),(b,f),(c,g)) 
def subtr(vec1: Array[Double], vec2: Array[Double]) =
  (vec1 zip vec2).map{case (a,b) => a-b}.toArray[Double]
subtr(X,Y)

// COMMAND ----------

val ZZ = X zip Y
ZZ(2)._2

// COMMAND ----------

//vector subtraction3:
// Zip operator concatenates 2 arrays Array(a,b,c) zip Array(e,f,g) -> Array((a,e),(b,f),(c,g))
// Tuple Array / Scala method to perform nomtuple._index
def divvec2(vec1: Array[Double], vec2: Array[Double]) =
  (vec1 zip vec2).map(a => a._1 - a._2)

// COMMAND ----------

//Method1 : vectors SUM
def sum(vec1: Array[Double],vec2: Array[Double]) =
  (vec1 zip vec2).map(a => a._1 + a._2)  

//Method2 : vectors SUM
def sumvec1(vec1: Array[Double], vec2: Array[Double]) =
  (vec1 zip vec2).map{case(a,b) => a + b }.toArray[Double]

// COMMAND ----------

//scalar product w.x => prodsc(w,x)
//difference between scalar product and y => (prodsc(w,x)-y)
//product of two 2.0* doubles (prodsc(w,x)-y)
//product between scalar and a vector prodscalvec(2.0*(prodsc(w,x)-y),x)

// COMMAND ----------

// MAGIC %md
// MAGIC Implementation of gradient descent : definition of a function in scala

// COMMAND ----------

def sigmaNF(train : Array[(Double, Array[Double])], current_w : Array[Double],
           sizefeat: Int): Array[Double] = {
          //sizefeat == x dimension (0.0) (functional programming) 
  
  var sigma = Array.fill(sizefeat)(0.0) //Cumulative and Variable
  train.foreach{case (y,x) => sigma = sum(sigma,(prodbyscal(2.0*(prods(current_w, x ) -y), x)))};
  sigma  //foreach == sigma increm
}

// each element is of type y,x (case (x,y)) then we make the vector sum of the scalar product

// COMMAND ----------

//MORE ELEGANT FUNCTION OF SIGMA
def sigmaA(train : Array [(Double, Array[Double])], current_w : Array[Double]) : Array[Double] = {
  train.map{case (y,x) => 
    (prodbyscal (2.0 * (prods(current_w, x) -y), x))}.reduce(sum)
}

// COMMAND ----------

def sigma(train : Array[(Double, Array[Double])],current_w : Array[Double],  sizefeat : Int): Array[Double] = {
    train.map{case (y,x)=>(prodbyscal(2.0*(prods(current_w,x)-y),x))}.reduce(sum)
}

// COMMAND ----------

//Batch Gradient Descent : Dtrain generation
def batchGD(dtrain: Array[(Double, Array[Double])], init_w: Array[Double], sizefeat: Int, nb_of_epochs: Int, stepsize : Double): Array[Double] =
{
  val m = dtrain.size // DTrain
  var w = init_w
  for (i <- 1 to nb_of_epochs) {
      val s = sigma(dtrain, w, sizefeat);
      w = subtr(w,prodbyscal((stepsize/m) , s))
  }
  w
}

// COMMAND ----------

val X = (1 to 20).by(2).toArray //step by 2

// COMMAND ----------

val dtrain = X.map(x => ( (1 * x + 2).toDouble, Array(x,1.0) ) ) // and so on...

// COMMAND ----------

def batchGDA(dtrain : Array[(Double, Array[Double])], init_w: Array[Double], sizefeat: Int, nb_of_epochs: Int, stepsize:Double): Array[Double] =
{
  val m = dtrain.size // |Dtrain|
  var w = init_w
      for (i <- 1 to nb_of_epochs){
      val s = sigmaA(dtrain, w);
      w = subtr(w,prodbyscal((stepsize/m),s))
      println(w(0),w(1))
    } 
  w  
}

// COMMAND ----------

// MAGIC %md
// MAGIC Target : representation of a linear function in the form of a scalar product

// COMMAND ----------

// we make a small stepsize then we ask to display the 2 constants, the slop w1(0) then the intercept w1(1)
// The idea is to find the best convergence with iterative stepsizes
val sizefeat = 2
var w = Array.fill(sizefeat)(0.0)
val stepsize = 0.0000001

val w1 = batchGDA(dtrain,w,sizefeat,15,stepsize)
println(w1(0),w1(1))

// COMMAND ----------

val sizefeat = 2
var w = Array.fill(sizefeat)(0.0)
val stepsize = 0.0000001

val w1 = batchGD(dtrain,w,sizefeat,15,stepsize)
println(w1(0),w1(1))

// COMMAND ----------

//stepsize of 10^-7 and 15 epochs
val sizefeat = 2
var w = Array.fill(sizefeat)(0.0)
val stepsize = 0.0000001

val w2 = batchGD(dtrain,w,sizefeat,15,stepsize)
println(w2(0),w2(1))

// COMMAND ----------

//stepsize of 10^-6 and 100 epochs
val sizefeat = 2
var w = Array.fill(sizefeat)(0.0)
val stepsize=0.000001

val w2=batchGD(dtrain,w,sizefeat,100,stepsize)
println(w2(0),w2(1))

// COMMAND ----------

//stepsize of 10^-5 and 100 epochs
val sizefeat = 2
var w = Array.fill(sizefeat)(0.0)
val stepsize=0.00001

val w2=batchGD(dtrain,w,sizefeat,100,stepsize)
println(w2(0),w2(1))

// COMMAND ----------

//stepsize of 10^-4 and 100 epochs
val sizefeat = 2
var w = Array.fill(sizefeat)(0.0)
val stepsize=0.0001

val w2=batchGD(dtrain,w,sizefeat,100,stepsize)
println(w2(0),w2(1))

// COMMAND ----------

//stepsize of 10^-3 and 100 epochs
val sizefeat = 2
var w = Array.fill(sizefeat)(0.0)
val stepsize=0.001

val w2=batchGD(dtrain,w,sizefeat,100,stepsize)
println(w2(0),w2(1))

// COMMAND ----------

//stepsize of 10^-2 and 100 epochs
val sizefeat = 2
var w = Array.fill(sizefeat)(0.0)
val stepsize=0.01

val w2=batchGD(dtrain,w,sizefeat,100,stepsize)
println(w2(0),w2(1))

// COMMAND ----------

//stepsize of 10^-1 and 100 epochs
val sizefeat = 2
var w = Array.fill(sizefeat)(0.0)
val stepsize=0.1

val w2=batchGD(dtrain,w,sizefeat,100,stepsize)
println(w2(0),w2(1))

// COMMAND ----------

//exercise
val Y  = (1 to 20).by(2).toArray
val dtrain_ex = Y.map(x =>((1*x + 2).toDouble,Array(x,1.0)))

// COMMAND ----------

dtrain_ex
