#!/usr/bin/python3
# -*- encoding: utf-8 -*-
from spark_session import spark
from pyspark.ml.feature import PolynomialExpansion
from pyspark.ml.linalg import Vectors

"""
多项式展开
以2变量特征向量为例： （x，y），如果我们想用2度展开它，那么得到（x，x * x，x * y，y * y，y）
"""
df = spark.createDataFrame([
    (Vectors.dense([2.0, 1.0]),),
    (Vectors.dense([0.0, 0.0]),),
    (Vectors.dense([3.0, -1.0]),)
], ["features"])

polyExpansion = PolynomialExpansion(degree=2, inputCol="features", outputCol="polyFeatures")
polyDF = polyExpansion.transform(df)

polyDF.show(truncate=False)

spark.stop()
