#!/usr/bin/python3
# -*- encoding: utf-8 -*-
from spark_session import spark
from pyspark.ml.feature import Binarizer
"""
取个中间值，大于它输出1，小于它输出0
"""
continuousDataFrame = spark.createDataFrame([
    (0, 0.1),
    (1, 0.8),
    (2, 2.0)
], ["id", "feature"])

binarizer = Binarizer(threshold=1, inputCol="feature", outputCol="binarized_feature")

binarizedDataFrame = binarizer.transform(continuousDataFrame)

print("Binarizer output with Threshold = %f" % binarizer.getThreshold())
binarizedDataFrame.show()

spark.stop()
