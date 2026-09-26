# Data Platforms

A data platform is where compute and storage meet so teams can run jobs without reinventing the lake each time. The page starts with Databricks as the platform idea in practice, then moves to the query engines and their costs in Athena, Spark, and BigQuery, compares Glue DataBrew with Data Wrangler, and closes with a GCP disk note. The same notes are in [Data Product](data-product.md) and [ML Architecture](../../ai-engineering/mlops/ml-architecture.md).

## Databricks

The first platform is Databricks, and the story there runs from the guarantee it makes (ACID), through its courses and core concepts, into the Spark APIs, optimizations, and best practices underneath.

The same notes are in [Databricks](../../ai-engineering/mlops/experiment-management.md#databricks).

A lake only behaves like a database if writes are safe, so the first question is what ACID transactions are. [Databricks is ACID](https://databricks.com/glossary/acid-transactions) is the Databricks glossary entry that answers it, and the figure below is the same point.

<figure><img src="../../ops/.gitbook/assets/image (40).png" alt=""><figcaption><p>Databricks is ACID</p></figcaption></figure>

With the guarantee in place, the DB Learning Library is where to learn the platform itself. The [Free courses](https://www.databricks.com/training/catalog?costs=free) are the free slice of the Databricks training catalog, expert-led courses in data science, machine learning, and big data analytics. The [Docs Optimization recommendations](https://docs.databricks.com/en/optimizations/index.html) are the documentation on optimizations and performance recommendations on Databricks, and the [Comprehensive Guide to Optimize Databricks, Spark and Delta Lake Workloads](https://www.databricks.com/discover/pages/optimize-data-workloads-guide) collects best practices and strategies for making data workloads faster and more efficient. Two starter courses split the platform by role: DB [for ML](https://www.databricks.com/training/catalog/get-started-with-databricks-for-machine-learning-2460) tours the workspace and notebooks, then exploratory data analysis, feature engineering, and tracking and managing models with MLflow, while [DB for Data Engineering](https://www.databricks.com/training/catalog/get-started-with-databricks-for-data-engineering-1511) covers the foundations of a basic data engineering workflow, the workspace, Unity Catalog, and the day-to-day building blocks, through paired demos and labs.

Outside the official library, the (good) [Introduction & Tutorial](https://medium.com/@chuck.connell.3/databricks-a-history-and-introduction-438ce827227) is a good walk through cluster, notebook, table, SQL, DataFrame, and connections. The [must know 7 concepts](https://www.datacamp.com/tutorial/introduction-to-databricks) are DataCamp's tutorial on the seven must-know Databricks concepts for any data specialist, and the two figures below come from it.

<figure><img src="../../ops/.gitbook/assets/image (41).png" alt=""><figcaption><p>must know 7 concepts</p></figcaption></figure>

<figure><img src="../../ops/.gitbook/assets/image (42).png" alt=""><figcaption><p>must know 7 concepts</p></figcaption></figure>

Under the notebooks sits Spark, and the first API choice is RDD vs Dataframe vs Dataset. The [2016 official blog post](https://www.databricks.com/blog/2016/07/14/a-tale-of-three-apache-spark-apis-rdds-dataframes-and-datasets.html) is the tale of three Apache Spark APIs: their performance, their optimization benefits, and when to use each. The [linkedin blog post](https://www.linkedin.com/pulse/rdd-vs-dataframe-dataset-sanyam-jain-iwsfe/) goes over the differences among the three again, the [comparison on youtube](https://www.youtube.com/watch?v=aBUqIAGxeg8) is the video version of the same comparison, and [RDDs vs. Dataframes vs. Datasets – What is the Difference and Why Should Data Engineers Care?](https://www.analyticsvidhya.com/blog/2020/11/what-is-the-difference-between-rdds-dataframes-and-datasets/) starts from Apache Spark being the first choice for data engineers and explains why the difference matters to them.

Once the API is chosen, the next concern is optimizations. [Optimization recommendations on Databricks](https://docs.databricks.com/en/optimizations/index.html) is the same documentation page on performance recommendations named above, and the [Comprehensive Guide to Optimize Databricks, Spark and Delta Lake Workloads](https://www.databricks.com/discover/pages/optimize-data-workloads-guide) is the same best-practices guide, here read for tuning rather than learning. [Why and How: Partitioning in Databricks](https://medium.com/@eduard2popa/why-and-how-partitioning-in-databricks-e9e6f960db43) is Eduard Popa on the single most common lever, partitioning.

Best practices close the Databricks story: the [official docs](https://docs.databricks.com/en/delta/best-practices.html) give the best practices and recommendations for using Delta Lake on Databricks.


## Athena

After Databricks, Athena is the serverless query engine where the questions turn to engine, cost, and performance tuning. The list below is a question checklist; the tuning item links the AWS post with its top-10 tips.

1. What is the engine behind Athena
2. How is Presto different from Spark? How does it affect your query planning?
3. [Performance tuning](https://aws.amazon.com/blogs/big-data/top-10-performance-tuning-tips-for-amazon-athena/) — Top 10: partitioning, bucketing, compression, optimize file sizes, optimize columnar data store generation, query tuning, optimize order by, optimize group by, use approx functions, column selection. What are the tradeoffs (time vs cost)?
4. What is the cost composed of?
5. How can you calculate cost?
6. How can you optimize your queries (partitions, join order, limit tricks, etc)
7. What options do you have to limit the cost of Athena?
8. When would you use Athena vs Spark

## Spark

The last Athena question, when to use Athena vs Spark, leads straight into Spark itself: join strategies, DataFrame vs Dataset, and adaptive query execution.

The same notes are in [Spark](../../ai-engineering/mlops/experiment-management.md#spark).

The articles by sivaprasad mandapati are a good pool here: [Several spark articles that can be used as candidate questions](https://medium.com/@sivaprasad-mandapati) is his Medium page, and several of the notes below come from it.

Joins are where most Spark jobs spend their time. [Join strategies #1](https://medium.com/datakaresolutions/optimize-spark-sql-joins-c81b4e3ed7da) is Prabhath Vemula's best practices and optimization techniques from joining two big tables in Spark SQL, and [Join strategies #2](https://medium.com/data-science/strategies-of-spark-join-c0e7b4572bcf) walks the join strategies Spark uses internally, which helps with tricky joins and out-of-memory root causes. The questions to ask of both are how? Pros and cons. (broadcast hash, shuffle hash, shuffle sort merge, cartesian). A related question is what’s the difference between a data frame and a dataset?

The choice between the two most common strategies is [Sort merge vs broadcast](https://medium.com/swlh/spark-joins-tuning-part-1-sort-merge-vs-broadcast-a98d82610cf0), a tuning article that starts from partitions spread across executors as Spark's bread and butter. Its finding is that a broadcast join is 4 times faster if one of the table is small and enough to fit in memory. Is broadcasting always a good solution? Absolutely no. If you are joining two data sets both are very large broadcasting any table would kill your spark cluster and fails your job.

When neither side is small, the lever is the shuffle, which is the subject of [Shuffle & AQE](https://medium.com/@sivaprasad-mandapati/spark-joins-tuning-part-2-shuffle-partitions-aqe-8688cb23317b), the second part of that tuning series. Adaptive Query Execution (AQE) is an optimization technique in Spark SQL that makes use of the runtime statistics to choose the most efficient query execution plan. It does three things:

- Dynamically coalescing shuffle partitions
- Dynamically switching join strategies
- Dynamically optimizing skew joins

## BigQuery

After Spark, BigQuery brings back the Athena kind of question on another cloud: cost, partitions, clustering, and access. The checklist is:

1. What is the difference in the implementation between partitions and clustering in BQ?
2. What ways do you know to reduce query cost in BigQuery?
3. What is the BigQuery cost composed of? How can you reduce storage cost?
4. Did you ever encounter a memory error when running BigQuery? Why does it happen and how is it related to the Dremel implementations
5. How can you control the access to sensitive data in BigQuery?
6. What options do you have to limit the cost of BigQuery?
7. When using BigQuery ML to train TF models — what happens in the background?

## AWS Glue DataBrew vs Data Wrangler

Beside the query engines, data still has to be prepared before training, and AWS offers two tools for it. [Julien simon on AWS glue data brew vs data wrangler](https://julsimon.medium.com/data-preparation-aws-glue-data-brew-or-amazon-sagemaker-data-wrangler-d8e76d1510cb) asks which one to use for data preparation, AWS Glue DataBrew or Amazon SageMaker Data Wrangler.

## GCP

Closing the cloud shelf is a practical GCP problem: running out of disk mid-download or mid-training. [Resize google disk size](https://medium.com/google-cloud/resize-your-persist-disk-on-google-cloud-on-the-fly-b3491277b718) shows how to resize a persistent disk on the fly instead of stopping the VM and attaching a new disk. The references [1,](https://cloud.google.com/compute/docs/disks/add-persistent-disk) and [2](https://www.cloudbooklet.com/how-to-resize-disk-of-a-vm-instance-in-google-cloud/) back it up: the first is the Google Cloud documentation on creating and attaching persistent disk volumes on Windows and Linux VMs, and the second resizes a Compute Engine persistent disk without downtime or restarting the instance.


## Deprecated links

{% hint style="warning" %}
These links and images no longer work. The original wording is kept here. A same-resource copy, when one was checked, is used above.
{% endhint %}

- Vector Search. This address no longer opens: https://www.databricks.com/training/catalog/new-capability-overview-vector-search-2535
- How I Use Caching in Databricks to Increase Performance and Save Costs. This address no longer opens: https://blog.det.life/caching-in-databricks-explained-68c07bf1f76b
