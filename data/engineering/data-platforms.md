# Data Platforms

A data platform is where compute and storage meet so teams can run jobs without reinventing the lake each time.
This page starts with Databricks, then Athena, Spark, BigQuery, Glue versus Wrangler, and the GCP notes.
The same notes are in [Data Product](data-product.md) and [ML Architecture](../../ai-engineering/mlops/ml-architecture.md).

## Databricks

This section is the platform idea in practice on Databricks: ACID, courses, Spark APIs, optimizations, and best practices.

The same notes are in [Databricks](../../ai-engineering/mlops/experiment-management.md#databricks).

- What are ACID Transactions? What are ACID Transactions? [Databricks is ACID](https://databricks.com/glossary/acid-transactions)

<figure><img src="../../ops/.gitbook/assets/image (40).png" alt=""><figcaption><p>Databricks is ACID</p></figcaption></figure>

- DB Learning Library
 - Explore Databricks' comprehensive training catalog featuring expert-led courses in data science, machine learning, and big data analytics. [Free courses](https://www.databricks.com/training/catalog?costs=free)
 - Learn about optimizations and performance recommendations on Databricks. [Docs Optimization recommendations](https://docs.databricks.com/en/optimizations/index.html)
 - Discover best practices and strategies to optimize your data workloads with Databricks, enhancing performance and efficiency. [Comprehensive Guide to Optimize Databricks, Spark and Delta Lake Workloads](https://www.databricks.com/discover/pages/optimize-data-workloads-guide)
 - Get Started with Databricks for Machine Learning. DB [for ML](https://www.databricks.com/training/catalog/get-started-with-databricks-for-machine-learning-2460)
 - Get Started with Databricks for Data Engineering. [DB for Data Engineering](https://www.databricks.com/training/catalog/get-started-with-databricks-for-data-engineering-1511)
- (good) [Introduction & Tutorial](https://medium.com/@chuck.connell.3/databricks-a-history-and-introduction-438ce827227) good cluster notebook table SQL DataFrame connections
- [must know 7 concepts](https://www.datacamp.com/tutorial/introduction-to-databricks)

<figure><img src="../../ops/.gitbook/assets/image (41).png" alt=""><figcaption><p>must know 7 concepts</p></figcaption></figure>

<figure><img src="../../ops/.gitbook/assets/image (42).png" alt=""><figcaption><p>must know 7 concepts</p></figcaption></figure>

- RDD vs Dataframe vs Dataset
 - Explore Apache Spark's RDDs, DataFrames, and Datasets APIs, their performance, optimization benefits, and when to use each for efficient data processing. [2016 official blog post](https://www.databricks.com/blog/2016/07/14/a-tale-of-three-apache-spark-apis-rdds-dataframes-and-datasets.html)
 - Before delving deep into the differences among the three mentioned above, let. [linkedin blog post](https://www.linkedin.com/pulse/rdd-vs-dataframe-dataset-sanyam-jain-iwsfe/)
 - [comparison on youtube](https://www.youtube.com/watch?v=aBUqIAGxeg8)
 - Apache Spark continues to be the first choice for data engineers. [RDDs vs. Dataframes vs. Datasets – What is the Difference and Why Should Data Engineers Care?](https://www.analyticsvidhya.com/blog/2020/11/what-is-the-difference-between-rdds-dataframes-and-datasets/)
- Optimizations
 - Learn about optimizations and performance recommendations on Databricks. [Optimization recommendations on Databricks](https://docs.databricks.com/en/optimizations/index.html)
 - Discover best practices and strategies to optimize your data workloads with Databricks, enhancing performance and efficiency. [Comprehensive Guide to Optimize Databricks, Spark and Delta Lake Workloads](https://www.databricks.com/discover/pages/optimize-data-workloads-guide)
 - [Why and How: Partitioning in Databricks](https://medium.com/@eduard2popa/why-and-how-partitioning-in-databricks-e9e6f960db43)
- Best Practices
 - Best practices and recommendations for using Delta Lake on Databricks. [official docs](https://docs.databricks.com/en/delta/best-practices.html)


## Athena

After Databricks, this section lists Athena engine, cost, and performance-tuning questions.

1. What is the engine behind Athena
2. How is Presto different from Spark? How does it affect your query planning?
3. [Performance tuning](https://aws.amazon.com/blogs/big-data/top-10-performance-tuning-tips-for-amazon-athena/) — Top 10: partitioning, bucketing, compression, optimize file sizes, optimize columnar data store generation, query tuning, optimize order by, optimize group by, use approx functions, column selection. What are the tradeoffs (time vs cost)?
4. What is the cost composed of?
5. How can you calculate cost?
6. How can you optimize your queries (partitions, join order, limit tricks, etc)
7. What options do you have to limit the cost of Athena?
8. When would you use Athena vs Spark

## Spark

Beside Athena, this section lists Spark join strategies, DataFrame vs Dataset, and AQE notes.

The same notes are in [Spark](../../ai-engineering/mlops/experiment-management.md#spark).

- [Several spark articles that can be used as candidate questions](https://medium.com/@sivaprasad-mandapati) sivaprasad mandapati
2. [Join strategies #1](https://medium.com/datakaresolutions/optimize-spark-sql-joins-c81b4e3ed7da), [Join strategies #2](https://medium.com/data-science/strategies-of-spark-join-c0e7b4572bcf) — how? Pros and cons. (broadcast hash, shuffle hash, shuffle sort merge, cartesian).
3. What’s the difference between a data frame and a dataset?
- [Sort merge vs broadcast](https://medium.com/swlh/spark-joins-tuning-part-1-sort-merge-vs-broadcast-a98d82610cf0)
 - broadcast join is 4 times faster if one of the table is small and enough to fit in memory
 - Is broadcasting always a good solution? Absolutely no. If you are joining two data sets both are very large broadcasting any table would kill your spark cluster and fails your job.
- [Shuffle & AQE](https://medium.com/@sivaprasad-mandapati/spark-joins-tuning-part-2-shuffle-partitions-aqe-8688cb23317b)
 - Adaptive Query Execution (AQE) is an optimization technique in Spark SQL that makes use of the runtime statistics to choose the most efficient query execution plan.
 - Dynamically coalescing shuffle partitions
 - Dynamically switching join strategies
 - Dynamically optimizing skew joins

## BigQuery

After Spark, this section lists BigQuery cost, partitions, clustering, and access questions.

1. What is the difference in the implementation between partitions and clustering in BQ?
2. What ways do you know to reduce query cost in BigQuery?
3. What is the BigQuery cost composed of? How can you reduce storage cost?
4. Did you ever encounter a memory error when running BigQuery? Why does it happen and how is it related to the Dremel implementations
5. How can you control the access to sensitive data in BigQuery?
6. What options do you have to limit the cost of BigQuery?
7. When using BigQuery ML to train TF models — what happens in the background?

## AWS Glue DataBrew vs Data Wrangler

Beside BigQuery, this section points at Julien Simon on AWS Glue DataBrew vs Data Wrangler.

[Julien simon on AWS glue data brew vs data wrangler](https://julsimon.medium.com/data-preparation-aws-glue-data-brew-or-amazon-sagemaker-data-wrangler-d8e76d1510cb)

## GCP

Closing the cloud shelf, this section is resizing Google Cloud persistent disks.

[Resize google disk size](https://medium.com/google-cloud/resize-your-persist-disk-on-google-cloud-on-the-fly-b3491277b718), [1,](https://cloud.google.com/compute/docs/disks/add-persistent-disk) [2](https://www.cloudbooklet.com/how-to-resize-disk-of-a-vm-instance-in-google-cloud/)


## Deprecated links

{% hint style="warning" %}
These links and images no longer work. The original wording is kept here. A same-resource copy, when one was checked, is used above.
{% endhint %}

- Vector Search. This address no longer opens: https://www.databricks.com/training/catalog/new-capability-overview-vector-search-2535
- How I Use Caching in Databricks to Increase Performance and Save Costs. This address no longer opens: https://blog.det.life/caching-in-databricks-explained-68c07bf1f76b
