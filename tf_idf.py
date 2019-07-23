#!/usr/bin/python3
# -*- encoding: utf-8 -*-
import os
from pyspark.sql import SparkSession
from pyspark.ml.feature import HashingTF, IDF, Tokenizer

os.environ['SPARK_HOME'] = '/usr/local/Cellar/apache-spark/spark-2.4.3-bin-hadoop2.7'
os.environ["PYSPARK_PYTHON"] = "/usr/local/bin/python3.7"

spark = SparkSession \
    .builder \
    .appName("pysparkpro") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

sentenceData = spark.createDataFrame([
    (0.0, "Hi I heard about Spark"),
    (0.0, "I wish Java could use case classes"),
    (1.0, "Logistic regression models are neat")
], ["label", "sentence"])
tokenizer = Tokenizer(inputCol="sentence", outputCol="words")
wordsData = tokenizer.transform(sentenceData)

hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures",
                      numFeatures=20)  # numFeatures指哈希表的大小，输出格式[哈希表大小,[哈希出来的索引列表],[对哈希索引计算出来的词频列表]]
featurizedData = hashingTF.transform(wordsData)
# alternatively, CountVectorizer can also be used to get term frequency vectors
a = featurizedData.toPandas()
featurizedData.select("rawFeatures").show()
idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)

rescaledData.select("label", "features").show()

spark.stop()
