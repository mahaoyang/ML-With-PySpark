#!/usr/bin/python3
# -*- encoding: utf-8 -*-
from spark_session import spark
from pyspark.ml.feature import CountVectorizer

"""
可以认为是one hot编码的spark版
"""

# Input data: Each row is a bag of words with a ID.
df = spark.createDataFrame([
    (0, "a b c d d".split(" ")),
    (1, "d a b b c a".split(" "))
], ["id", "words"])

# fit a CountVectorizerModel from the corpus.
cv = CountVectorizer(inputCol="words", outputCol="features", vocabSize=4,
                     minDF=2.0)  # vocabSize 词组大小，minDF 所有文档合成一个大词袋，该词的最小词频

model = cv.fit(df)

result = model.transform(df)
result.show(truncate=False)

spark.stop()
