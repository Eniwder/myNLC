import org.apache.log4j.{Level, LogManager}
import org.apache.spark.ml.feature.Word2Vec
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.log4j.ConsoleAppender
import org.apache.log4j.SimpleLayout

object Test extends App {

  val log = LogManager.getRootLogger
  log.setLevel(Level.WARN)
  log.addAppender(new ConsoleAppender(new SimpleLayout))

  val conf = new SparkConf().setAppName("Simple Application").setMaster("local[2]")
  val sc = new SparkContext(conf)
  val spark = SparkSession.builder.appName("nlc").getOrCreate()

  ///////////////////////////////////////////////////////////////////////////////

  val twoGram = (s: String) => ngram(s, 2)

  val data = Seq(("A001", twoGram("数値列")),
    ("A002", twoGram("文字列長")),
    ("A002", twoGram("文字列の長さ")),
    ("A002", twoGram("文字列のアレ")),
    ("A003", twoGram("Scala好き")),
    ("A003", twoGram("アイラブScala")),
    ("A003", twoGram("Javaは微妙")),
    ("A003", twoGram("Scalaですから")),
    ("A003", twoGram("昔はJava好きでした")))

  val documentDF = spark.createDataFrame(data).toDF("label", "text")
  documentDF.show()

  // Learn a mapping from words to Vectors.
  val word2Vec = new Word2Vec()
    .setInputCol("text")
    .setOutputCol("features")
    .setVectorSize(3)
    .setMinCount(0)
  val model = word2Vec.fit(documentDF)

  val result = model.transform(documentDF)
  result.show()


  //result.collect().foreach { case Row(text: Seq[_],id:Int ,features: Vector) =>    println(s"Text: [${text.mkString(", ")}] => \nVector: $features\n")
  val Array(trainingData, testData) = result.randomSplit(Array(0.7, 0.3))

  val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(result)
  val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(result)
  val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)
  val rf = new RandomForestClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setNumTrees(10)
  val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, rf, labelConverter))

  val model2 = pipeline.fit(result)
  val predictions = model2.transform(model.transform(spark.createDataFrame(Seq(("A003", twoGram("俺Scala命")), ("A002", twoGram("長さ文字列")))).toDF("label", "text")))
  //val model2 = pipeline.fit(trainingData)
  //val predictions = model2.transform(testData)

  predictions.show()

  val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
  val accuracy = evaluator.evaluate(predictions)
  println("Test Error = " + (1.0 - accuracy))

  val rfModel = model2.stages(2).asInstanceOf[RandomForestClassificationModel]
  //println("Learned classification forest model:\n" + rfModel.toDebugString)

  spark.stop()

  def ngram(str: String, n: Int, acc: List[String] = Nil): List[String] = str match {
    case _ if str.length < n => Nil
    case _                   => ngram(str.tail, n, str.take(n) :: acc)
  }

}
