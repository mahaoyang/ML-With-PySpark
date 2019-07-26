#!/usr/bin/python3
# -*- encoding: utf-8 -*-
import os
from pyspark.sql import SparkSession

os.environ['SPARK_HOME'] = '/usr/local/Cellar/apache-spark/spark-2.4.3-bin-hadoop2.7'
os.environ["PYSPARK_PYTHON"] = "/usr/local/bin/python3.7"
if os.environ.get('OS') == 'windows':
    os.environ['SPARK_HOME'] = 'E:/spark-2.4.3-bin-hadoop2.7'
    os.environ["PYSPARK_PYTHON"] = "C:/Users/99263/PycharmProjects/ML-With-PySpark/venv/Scripts/python.exe"

spark = SparkSession \
    .builder \
    .appName("pysparkpro") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()
