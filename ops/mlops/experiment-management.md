# ML Experiment Management

This page covers experiment tracking tools, Spark APIs, and Databricks notes for training and deploying models.

## Experiment platforms

This section lists experiment management products and short notes on each one.

- [All the alternatives](https://blog.valohai.com/top-machine-learning-platforms)
- Cnvrg.io
   - Manage — Easily navigate machine learning with dashboards, reproducible data science, dataset organization, experiment tracking and visualization, a model repository and more
   - Build — Run and track experiments in hyperspeed with the freedom to use any compute environment, framework, programming language or tool — no configuration required
   - Automate — Build more models and automate your machine learning from research to production using reusable components and drag-n-drop interface
- Comet.ml — Comet lets you track code, experiments, and results on ML projects. It’s fast, simple, and free for open source projects.
- Floyd — notebooks on the cloud, similar to Colab / Kaggle, etc. GPU costs $4/h
- Missing link — RIP

## Spark

This section is Spark API primers and related reading.

- [RDDs vs datasets vs dataframes](https://databricks.com/blog/2016/07/14/a-tale-of-three-apache-spark-apis-rdds-dataframes-and-datasets.html)
- [What are RDDs?](https://www.quora.com/What-are-resilient-distributed-datasets-RDDs-How-do-they-help-Spark-with-its-awesome-speed)
- [Keras, TF, Spark](https://medium.com/qubida-analytics-blog/build-a-deep-learning-image-classification-pipeline-with-spark-keras-and-tensorflow-3bf26fda15e6)
- [Repartition vs coalesce](https://medium.com/@mrpowers/managing-spark-partitions-with-coalesce-and-repartition-4050c57ad5c4)

## Databricks

This section is Databricks tools, Spark ML, MLflow, and spark-sklearn notes.

1. [Koalas](https://github.com/databricks/koalas) — pandas API on Apache Spark
2. [Intro to DB on Spark](https://www.youtube.com/watch?v=DqihOzZl5jM&list=PLTPXxbhUt-YV-CwJTiE36C-0le8wlFJ5G&index=5), has some basic sklearn-like tools and other custom operations such as a single-vector-based aggregator for using features as an input to a model
3. [Pyspark.ml](http://web.archive.org/web/20210120184921/https://spark.apache.org/docs/latest/api/python/pyspark.ml.html)
4. [Keras as a single node (no Spark)](https://docs.databricks.com/applications/deep-learning/single-node-training/keras.html)
5. [Horovod for distributed Keras (and more)](https://docs.databricks.com/applications/deep-learning/distributed-training/mnist-tensorflow-keras.html)
6. [Documentation](https://docs.databricks.com/index.html) (read me; has all libraries)
7. [Medium tutorial](https://medium.com/data-science/how-to-train-your-neural-networks-in-parallel-with-keras-and-apache-spark-ea8a3f48cae6), explains the three pros of Databricks with examples of using native and non-native algorithms
   - Spark SQL
   - MLflow
   - Streaming
   - SystemML DML using Keras models
8. [Sklearn notebook example](https://docs.databricks.com/_static/notebooks/scikit-learn.html)
9. [Utilizing Spark nodes](https://databricks.com/blog/2016/02/08/auto-scaling-scikit-learn-with-apache-spark.html) for grid searching with sklearn

```python
from spark_sklearn import GridSearchCV
```

10. [How can we leverage](https://databricks-prod-cloudfront.cloud.databricks.com/public/13fe59d17777de29f8a2ffdf85f52925/5638528096339357/1867405/6918044996430578/latest.html) our existing experience with modeling libraries like [scikit-learn](http://scikit-learn.org/stable/index.html)? We'll explore three approaches that make use of existing libraries, but still benefit from the parallelism provided by Spark.

    These approaches are:

    - Grid Search
    - Cross Validation
    - Sampling (random, chronological subsets of data across clusters)
11. GitHub [spark-sklearn](https://github.com/databricks/spark-sklearn) (needs to be compared to what Spark has internally)
    1. [Ref](http://web.archive.org/web/20190920232508/https://mapr.com/blog/predicting-airbnb-listing-prices-scikit-learn-and-apache-spark/): It's worth pausing here to note that the architecture of this approach is different than that used by MLlib in Spark. Using spark-sklearn, we're simply distributing the cross-validation run of each model (with a specific combination of hyperparameters) across each Spark executor. Spark MLlib, on the other hand, will distribute the internals of the actual learning algorithms across the cluster.
    2. The main advantage of spark-sklearn is that it enables leveraging the very rich set of [machine learning](http://web.archive.org/web/20200122215157/https://mapr.com/ebook/machine-learning-logistics/) algorithms in scikit-learn. These algorithms do not run natively on a cluster (although they can be parallelized on a single machine) and by adding Spark, we can unlock a lot more horsepower than could ordinarily be used.
    3. Using [spark-sklearn](https://github.com/databricks/spark-sklearn) is a straightforward way to throw more CPU at any machine learning problem you might have. We used the package to reduce the time spent searching and reduce the error for our estimator
12. [Airbnb example using spark and sklearn, cross_val and grid search comparison vs joblib](http://web.archive.org/web/20190920232508/https://mapr.com/blog/predicting-airbnb-listing-prices-scikit-learn-and-apache-spark/)
13. [Tracking experiments](https://docs.databricks.com/applications/mlflow/tracking.html)
    - [example](https://docs.databricks.com/applications/mlflow/tracking-examples.html#train-a-scikit-learn-model-and-save-in-scikit-learn-format)
14. [Saving, loading, deployment](https://docs.databricks.com/applications/mlflow/models.html#examples)
    - [AWS SageMaker](http://web.archive.org/web/20190920061714/https://docs.databricks.com/applications/mlflow/model-examples.html#scikit-learn-model-deployment-on-sagemaker)
    - [Medium](https://medium.com/data-science/a-different-way-to-deploy-a-python-model-over-spark-2da4d625f73e) and sklearn random trees
15. [How to productionalize your model using Databricks Spark 2.0 on YouTube](https://databricks.com/session/how-to-productionize-your-machine-learning-models-using-apache-spark-mllib-2-x)

## Deprecated links

{% hint style="warning" %}
These links and images no longer work. The original wording is kept here. A same-resource copy, when one was checked, is used above.
{% endhint %}

- Trains - open source. This address no longer opens: https://heartbeat.fritz.ai/trains-all-aboard-ba92a728eb6d
- Best practices. This address no longer opens: https://www.bi4all.pt/en/news/en-blog/apache-spark-best-practices/
- Pyspark.ml. This address no longer opens: https://spark.apache.org/docs/latest/api/python/pyspark.ml.html
- Medium tutorial. This address no longer opens: https://towardsdatascience.com/how-to-train-your-neural-networks-in-parallel-with-keras-and-apache-spark-ea8a3f48cae6
- systemML notebooks (didnt read). This address no longer opens: http://systemml.apache.org/get-started.html#sample-notebook
- Ref. This address no longer opens: https://mapr.com/blog/predicting-airbnb-listing-prices-scikit-learn-and-apache-spark/
- machine learning (ebook). This address no longer opens: https://mapr.com/ebook/machine-learning-logistics/
- Airbnb example using spark and sklearn,cross_val& grid search comparison vs joblib. This address no longer opens: https://mapr.com/blog/predicting-airbnb-listing-prices-scikit-learn-and-apache-spark/
- Sklearn example 2, tfidf. This address no longer opens: http://cdn2.hubspot.net/hubfs/438089/notebooks/ML/scikit-learn/demo_-_1_-_sklearn.html
- Aws sagemaker. This address no longer opens: https://docs.databricks.com/applications/mlflow/model-examples.html#scikit-learn-model-deployment-on-sagemaker
- Medium and sklearn random trees. This address no longer opens: https://towardsdatascience.com/a-different-way-to-deploy-a-python-model-over-spark-2da4d625f73e
