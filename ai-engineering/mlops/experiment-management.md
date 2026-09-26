# ML Experiment Management

Runs multiply faster than anyone can remember them. The page starts with the experiment platforms that track those runs, then drops to Spark, the distributed layer they often sit on, and ends with Databricks for tracking, training, and deploying them.

The same notes are in [General patterns](mlops.md#general-patterns) and [Hyper Parameter Optimization](../../evals/hyper-parameter-optimization.md).

## Experiment platforms

Experiment tracking starts with picking a platform, before Spark or Databricks enter the picture.

The same notes are in [Weights & Biases](mlops-course.md#weights--biases).

The market is large, so begin with a map: [All the alternatives](https://blog.valohai.com/top-machine-learning-platforms) is Valohai's comprehensive comparison of the top machine learning platforms, the services that support organizations developing machine learning solutions, and which of them suit your needs.

Cnvrg.io splits its pitch into three parts:

 - Manage — Easily navigate machine learning with dashboards, reproducible data science, dataset organization, experiment tracking and visualization, a model repository and more
 - Build — Run and track experiments in hyperspeed with the freedom to use any compute environment, framework, programming language or tool — no configuration required
 - Automate — Build more models and automate your machine learning from research to production using reusable components and drag-n-drop interface

Comet.ml is narrower: Comet lets you track code, experiments, and results on ML projects. It’s fast, simple, and free for open source projects. Floyd is the compute side, notebooks on the cloud, similar to Colab / Kaggle, etc., where the GPU costs $4/h. One more entry is a missing link — RIP.

## Spark

With the platforms named, the runs still need somewhere to execute at scale, and Spark is the distributed API layer they often sit on.

The same notes are in [Spark](../../data/engineering/data-platforms.md#spark).

Spark has three APIs, and choosing between them is the first decision. [RDDs vs datasets vs dataframes](https://databricks.com/blog/2016/07/14/a-tale-of-three-apache-spark-apis-rdds-dataframes-and-datasets.html) is Databricks' tale of the three, covering their performance, optimization benefits, and when to use each. [What are RDDs?](https://www.quora.com/What-are-resilient-distributed-datasets-RDDs-How-do-they-help-Spark-with-its-awesome-speed) goes one level down: resilient distributed datasets are Spark's data abstraction, and the features they are built with are responsible for its speed. With the abstraction in hand, [Keras, TF, Spark](https://medium.com/qubida-analytics-blog/build-a-deep-learning-image-classification-pipeline-with-spark-keras-and-tensorflow-3bf26fda15e6) builds a deep learning image classification pipeline on all three. Because Spark splits data into partitions and computes on them in parallel, Matthew Powers' [Repartition vs coalesce](https://medium.com/@mrpowers/managing-spark-partitions-with-coalesce-and-repartition-4050c57ad5c4) explains how data is partitioned and when to adjust it by hand to keep computations efficient.

## Databricks

On top of Spark, Databricks packages MLflow, notebooks, and spark-sklearn for training and deploy.

The same notes are in [Databricks](../../data/engineering/data-platforms.md#databricks) and [Databricks Delta Lake](../../data/engineering/lakes-and-warehouses.md#databricks-delta-lake).

The gentlest entry for a pandas user is [Koalas](https://github.com/databricks/koalas), the pandas API on Apache Spark. The [Intro to DB on Spark](https://www.youtube.com/watch?v=DqihOzZl5jM&list=PLTPXxbhUt-YV-CwJTiE36C-0le8wlFJ5G&index=5) video, has some basic sklearn-like tools and other custom operations such as a single-vector-based aggregator for using features as an input to a model. Those tools live in [Pyspark.ml](http://web.archive.org/web/20210120184921/https://spark.apache.org/docs/latest/api/python/pyspark.ml.html), the pyspark.ml package documentation for PySpark 3.0.1.

Deep learning on Databricks comes in two sizes. [Keras as a single node (no Spark)](https://docs.databricks.com/applications/deep-learning/single-node-training/keras.html) trains models on one node with TensorFlow and debugs them with inline TensorBoard, including a 10-minute tutorial notebook on tabular data. [Horovod for distributed Keras (and more)](https://docs.databricks.com/applications/deep-learning/distributed-training/mnist-tensorflow-keras.html) is the distributed version, training Keras on MNIST with HorovodRunner and the recommended model development workflow. The full [Documentation](https://docs.databricks.com/index.html) holds the how-to guides and reference for data teams on the Databricks Data + AI Platform.

The [Medium tutorial](https://medium.com/data-science/how-to-train-your-neural-networks-in-parallel-with-keras-and-apache-spark-ea8a3f48cae6), explains the three pros of Databricks with examples of using native and non-native algorithms: Spark SQL, MLflow, and Streaming, plus SystemML DML using Keras models.

Classical models follow the same path. [Sklearn notebook example](https://docs.databricks.com/_static/notebooks/scikit-learn.html) trains a basic scikit-learn classification model in Databricks, end to end. [Utilizing Spark nodes](https://databricks.com/blog/2016/02/08/auto-scaling-scikit-learn-with-apache-spark.html) introduces the scikit-learn integration package for Apache Spark, designed to distribute the most repetitive tasks of model tuning on a Spark cluster without impacting the workflow of data scientists. The import is a drop-in:

```python
from spark_sklearn import GridSearchCV
```

The Databricks scikit-learn notebook asks the question directly: [How can we leverage](https://databricks-prod-cloudfront.cloud.databricks.com/public/13fe59d17777de29f8a2ffdf85f52925/5638528096339357/1867405/6918044996430578/latest.html) our existing experience with modeling libraries like [scikit-learn](http://scikit-learn.org/stable/index.html)? We'll explore three approaches that make use of existing libraries, but still benefit from the parallelism provided by Spark.

 These approaches are:

 - Grid Search
 - Cross Validation
 - Sampling (random, chronological subsets of data across clusters)

The package behind those approaches is GitHub [spark-sklearn](https://github.com/databricks/spark-sklearn) (needs to be compared to what Spark has internally). The MapR write-up makes that comparison in three steps:

 1. [Ref](http://web.archive.org/web/20190920232508/https://mapr.com/blog/predicting-airbnb-listing-prices-scikit-learn-and-apache-spark/): It's worth pausing here to note that the architecture of this approach is different than that used by MLlib in Spark. Using spark-sklearn, we're simply distributing the cross-validation run of each model (with a specific combination of hyperparameters) across each Spark executor. Spark MLlib, on the other hand, will distribute the internals of the actual learning algorithms across the cluster.
 2. The main advantage of spark-sklearn is that it enables leveraging the very rich set of [machine learning](http://web.archive.org/web/20200122215157/https://mapr.com/ebook/machine-learning-logistics/) algorithms in scikit-learn. These algorithms do not run natively on a cluster (although they can be parallelized on a single machine) and by adding Spark, we can unlock a lot more horsepower than could ordinarily be used.
 3. Using [spark-sklearn](https://github.com/databricks/spark-sklearn) is a straightforward way to throw more CPU at any machine learning problem you might have. We used the package to reduce the time spent searching and reduce the error for our estimator

That passage comes from Predicting Airbnb Listing Prices with Scikit-Learn and Apache Spark on MapR, the [Airbnb example using spark and sklearn, cross_val and grid search comparison vs joblib](http://web.archive.org/web/20190920232508/https://mapr.com/blog/predicting-airbnb-listing-prices-scikit-learn-and-apache-spark/).

Once runs are distributed, they have to be recorded, which brings the page back to experiment management. [Tracking experiments](https://docs.databricks.com/applications/mlflow/tracking.html) covers MLflow experiments and tracking for agents, LLM applications, and ML model training runs, and the scikit-learn [example](https://docs.databricks.com/applications/mlflow/tracking-examples.html#train-a-scikit-learn-model-and-save-in-scikit-learn-format) on the same page trains a model and saves it in scikit-learn format. [Saving, loading, deployment](https://docs.databricks.com/applications/mlflow/models.html#examples) is how to log, load, and register MLflow models for deployment, including logging model dependencies so they are reproduced in the deployment environment. The Databricks Model Examples page covers deployment to [AWS SageMaker](http://web.archive.org/web/20190920061714/https://docs.databricks.com/applications/mlflow/model-examples.html#scikit-learn-model-deployment-on-sagemaker), and on [Medium](https://medium.com/data-science/a-different-way-to-deploy-a-python-model-over-spark-2da4d625f73e) Schaun Wheeler describes a different way to deploy a Python model over Spark. For the full production story, see [How to productionalize your model using Databricks Spark 2.0 on YouTube](https://databricks.com/session/how-to-productionize-your-machine-learning-models-using-apache-spark-mllib-2-x).

Two last notes sit beside the Databricks path. Apache SystemDS is an open source ML system for the end-to-end data science lifecycle, and its systemML notebooks (didnt read) are here: [http://systemml.apache.org/get-started.html#sample-notebook](http://systemml.apache.org/get-started.html#sample-notebook). Schaun Wheeler's deployment post, to separate the prediction method from the rest of the Python class and then implement in Scala, is also at its original address, with Medium and sklearn random trees: [https://towardsdatascience.com/a-different-way-to-deploy-a-python-model-over-spark-2da4d625f73e](https://towardsdatascience.com/a-different-way-to-deploy-a-python-model-over-spark-2da4d625f73e)

## Deprecated links

{% hint style="warning" %}
These links and images no longer work. The original wording is kept here. A same-resource copy, when one was checked, is used above.
{% endhint %}

- Trains - open source. This address no longer opens: https://heartbeat.fritz.ai/trains-all-aboard-ba92a728eb6d
- Best practices. This address no longer opens: https://www.bi4all.pt/en/news/en-blog/apache-spark-best-practices/
- Pyspark.ml. This address no longer opens: https://spark.apache.org/docs/latest/api/python/pyspark.ml.html
- Medium tutorial. This address no longer opens: https://towardsdatascience.com/how-to-train-your-neural-networks-in-parallel-with-keras-and-apache-spark-ea8a3f48cae6
- Ref. This address no longer opens: https://mapr.com/blog/predicting-airbnb-listing-prices-scikit-learn-and-apache-spark/
- machine learning (ebook). This address no longer opens: https://mapr.com/ebook/machine-learning-logistics/
- Airbnb example using spark and sklearn,cross_val& grid search comparison vs joblib. This address no longer opens: https://mapr.com/blog/predicting-airbnb-listing-prices-scikit-learn-and-apache-spark/
- Sklearn example 2, tfidf. This address no longer opens: http://cdn2.hubspot.net/hubfs/438089/notebooks/ML/scikit-learn/demo_-_1_-_sklearn.html
- Aws sagemaker. This address no longer opens: https://docs.databricks.com/applications/mlflow/model-examples.html#scikit-learn-model-deployment-on-sagemaker
