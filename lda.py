#!/usr/bin/python3
# -*- encoding: utf-8 -*-
from spark_session import spark
from pyspark.ml.clustering import LDA


def lda_train():
    # Loads data.
    dataset = spark.read.format("libsvm").load("train.libsvm")

    # Trains a LDA model.
    lda = LDA(k=10, maxIter=100)
    model = lda.fit(dataset)

    ll = model.logLikelihood(dataset)
    lp = model.logPerplexity(dataset)
    print("The lower bound on the log likelihood of the entire corpus: " + str(ll))
    print("The upper bound on perplexity: " + str(lp))

    # Describe topics.
    topics = model.describeTopics(3)
    print("The topics described by their top-weighted terms:")
    topics.show(truncate=False)

    # Shows the result
    transformed = model.transform(dataset)
    transformed.show(truncate=False)

    # Save & Stop
    # model.save('lda.model')
    spark.stop()


if __name__ == '__main__':
    lda_train()

    from spark_session import spark
    from pyspark.ml.linalg import Vectors, SparseVector
    from pyspark.ml.clustering import LDA
    from pyspark.mllib.linalg import DenseMatrix
    import pandas as pd
    import numpy as np

    df = spark.createDataFrame([[1, Vectors.dense([0.0, 1.0])], [2, SparseVector(2, {0: 1.0})],
                                [3, SparseVector(2, {0: 2.0})], [4, SparseVector(2, {1: 3.0})]],
                               ["id", "features"])
    lda = LDA(k=3, seed=1, optimizer="em")
    model = lda.fit(df)
    model.describeTopics().show(truncate=False)
    model.topicsMatrix()
    # tm = model.describeTopics().select('termWeights').collect()
    # tm = np.array(tm)
    tm = model.topicsMatrix().toArray().T

    transformed = model.transform(df)
    transformed.show(truncate=False)

    df = transformed.select('topicDistribution').collect()
    for i in df:
        i = i[0].toArray()
        res = i.dot(tm) / (np.linalg.norm(i) * np.linalg.norm(tm))
        print()

    spark.stop()
