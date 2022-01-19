# Data Engineering Questions & Training

#### Data Engineering Questions / Training <a href="#_wd2yffq1woi9" id="_wd2yffq1woi9"></a>

1. General
   1. [What scales of data have you worked with in the past?](https://business.linkedin.com/talent-solutions/resources/interviewing-talent/data-engineer)
   2. [How do you generally work with the departments that make use of your data?](https://business.linkedin.com/talent-solutions/resources/interviewing-talent/data-engineer)
   3. [Tell me about a time you had performance issues with an ETL. How did you identify this as a performance issue and how did you fix it?](https://business.linkedin.com/talent-solutions/resources/interviewing-talent/data-engineer)
   4. [Describe a time when you found a new use case for an existing database.](https://business.linkedin.com/talent-solutions/resources/interviewing-talent/data-engineer)
   5. [Describe the most challenging project you’ve worked on. What was your role?](https://business.linkedin.com/talent-solutions/resources/interviewing-talent/data-engineer)
   6. [Think back to a project you’re proud of. What was it that gave you that sense of pride and accomplishment?](https://business.linkedin.com/talent-solutions/resources/interviewing-talent/data-engineer)
   7. [What do you consider to be one of the biggest mistakes you’ve ever made in your previous job?](https://business.linkedin.com/talent-solutions/resources/interviewing-talent/data-engineer)
   8. [Explain the differences between stream processing and data processing, with one caveat: pretend that I’m not familiar with data at all.](https://business.linkedin.com/talent-solutions/resources/interviewing-talent/data-engineer)
   9. What are the considerations when choosing methods of ingesting data to bigquery?
   10. Have you worked with data science teams? What were your responsibilities?
   11. What are the considerations of choosing spark vs bigquery?
   12. What are the differences between ETL and ELT? [1](https://www.guru99.com/etl-vs-elt.html), [2](https://www.xplenty.com/blog/etl-vs-elt/), [3](https://blog.panoply.io/etl-vs-elt-the-difference-is-in-the-how)\
       ![](../../.gitbook/assets/0)\
       By guru99/david taylor

       ![](../../.gitbook/assets/1)

       ![](../../.gitbook/assets/2)

By mark smallcombe\


1. CAP Theorem
   1. [What Is CAP Theorem?](https://www.fullstack.cafe/blog/cap-theorem-interview-questions)
   2. [Can you 'get around' or 'beat' the CAP Theorem?](https://www.fullstack.cafe/blog/cap-theorem-interview-questions)
   3. [Name some types of Consistency patterns](https://www.fullstack.cafe/blog/cap-theorem-interview-questions)
   4. [What Do You Mean By High Availability (HA)?](https://www.fullstack.cafe/blog/cap-theorem-interview-questions)
   5. [What are A and P in CAP and the difference between them?](https://www.fullstack.cafe/blog/cap-theorem-interview-questions)
   6. [What does the CAP Theorem actually say?](https://www.fullstack.cafe/blog/cap-theorem-interview-questions)
   7. how it effect real world application (latency is availability in real world)
   8. [Another great resource for CAP questions](https://github.com/henryr/cap-faq)
   9. ![](../../.gitbook/assets/3)
   10. ![](../../.gitbook/assets/4)
   11. ![](../../.gitbook/assets/5)
   12. ![](../../.gitbook/assets/6)
   13. ![](../../.gitbook/assets/7)\

2. Explain the difference and the reason to choose using NoSQL {mongoDB | DynamoDB | .. } over Relational database {Postgress |MySQL} and vice versa. Give an example for a project where you had to make this choice, and walk through your reasoning.

(This question can be modified for the relevant technologies.. )\


1. Streaming vs Batch “Explain the difference and the reason to choose using Streaming over Batch and vice versa. Give an example for a project where you had to make this choice, and walk through your reasoning.”\

2. Job vs Service “Explain the difference and the reason to choose using Job over Service and vice versa. Give an example for a project where you had to make this choice, in the context of ML pipelines and walk through your reasoning.”
3. Athena
   1. What is the engine behind athena
   2. How is presto different from Spark? How does it affect your query planning?
   3. [Performance tuning](https://aws.amazon.com/blogs/big-data/top-10-performance-tuning-tips-for-amazon-athena/) - Top 10: partitioning, bucketing, compression, optimize file sizes, optimize columnar data store generation, query tuning, optimize order by, optimize group by, use approx functions, column selection. What are the tradeoffs (time vs cost)?
   4. What is the cost composed of?
   5. How can you calculate cost?
   6. How can you optimize your queries (partitions, join order, limit tricks, etc)
   7. What options do you have to limit the cost of athena?
   8. when would u use athena vs spark\

4. Spark -
   1. [Several spark articles that can be used as candidate questions](https://medium.com/@sivaprasad-mandapati) by sivaprasad mandapati.
   2. [Join strategies #](https://medium.com/datakaresolutions/optimize-spark-sql-joins-c81b4e3ed7da)1, [Join strategies](https://towardsdatascience.com/strategies-of-spark-join-c0e7b4572bcf) #2 - how? Pros and cons. (broadcast hash, shuffle hash, shuffle sort merge, cartesian).
   3. What’s the difference between a data frame and a dataset?
   4. [Sort merge vs broadcast](https://medium.com/swlh/spark-joins-tuning-part-1-sort-merge-vs-broadcast-a98d82610cf0)
      1. broadcast join is 4 times faster if one of the table is small and enough to fit in memory
      2. Is broadcasting always a good solution ? Absolutely no. If you are joining two data sets both are very large broad casting any table would kill your spark cluster and fails your job.
   5. [Shuffle & AQE](https://medium.com/@sivaprasad-mandapati/spark-joins-tuning-part-2-shuffle-partitions-aqe-8688cb23317b)
      1. Adaptive Query Execution (AQE) is an optimization technique in Spark SQL that makes use of the runtime statistics to choose the most efficient query execution plan.
      2. Dynamically coalescing shuffle partitions
      3. Dynamically switching join strategies
      4. Dynamically optimizing skew joins\

5. BigQuery
   1. What is the difference in the implementation between partitions and clustering in BQ?
   2. What ways do you know to reduce query cost in BigQuery?
   3. What is the BigQuery cost composed of? How can you reduce storage cost?
   4. Did you ever encounter a memory error when running BigQuery? Why does it happen and how is it related to the Dremel implementations
   5. How can you control the access to sensitive data in BigQuery?
   6. What options do you have to limit the cost of BigQuery?
   7. When using BigQuery ML to train TF models - what happens in the background?\

6. Airflow
   1. What is airflow?
   2. How do you transfer information between tasks in airflow?
   3. Please give me a real-world example of using spark and airflow together\

7. Data Validation
   1. How can you protect yourself from bad data? Data validation, TDDA, monitoring.
   2. Tools:
      1. Type validation: [typeguard](https://github.com/agronholm/typeguard)
      2. Data validation [pydantic](https://pydantic-docs.helpmanual.io/usage/dataclasses/)
      3. Test driven: [tdda](https://github.com/tdda/tdda)
      4. Data quality: [great expectations](https://greatexpectations.io)
      5. Saas: [SuperConductive by GE](https://superconductive.ai)\

8. File formats
   1. Can you explain the parquet file format? [https://parquet.apache.org/documentation/latest/](https://parquet.apache.org/documentation/latest/)
   2. How is this leveraged by Spark? [https://databricks.com/session/spark-parquet-in-depth](https://databricks.com/session/spark-parquet-in-depth)
   3. What are the shortcomings of parquet and how is it solved by file formats like hudi, delta, iceberg? [https://lakefs.io/hudi-iceberg-and-delta-lake-data-lake-table-formats-compared/](https://lakefs.io/hudi-iceberg-and-delta-lake-data-lake-table-formats-compared/)\

9. Julien simon on [AWS glue data brew vs data wrangler](https://julsimon.medium.com/data-preparation-aws-glue-data-brew-or-amazon-sagemaker-data-wrangler-d8e76d1510cb)\

10. [What is a CDC and why do you need it, or how do you use it?](https://rockset.com/blog/change-data-capture-what-it-is-and-how-to-use-it/) - Change data capture (CDC) is the process of recognising when data has been changed in a source system so a downstream process or system can action. A common use case is to reflect (replication) the change in a different target system so that the data in the systems stay in sync.\

11. Outage handling and the differences between stream-based processing vs concurrent isolated worker-based processing using

Q: you have a real time stream - what is better? A stream-based processing system, or a worker-based, that can be triggered on different time ranges, in the context of recovery from outage.\
![](../../.gitbook/assets/8)

By nielsen Ilai Malka

1. [How to design a](https://github.com/donnemartin/system-design-primer#system-design-interview-questions-with-solutions) .

![](../../.gitbook/assets/9)

* How would you design and implement an API rate limiter?
  1. [The twitter question](https://github.com/donnemartin/system-design-primer/blob/master/solutions/system\_design/twitter/README.md)

![](../../.gitbook/assets/10)\


1. [More than 2000 questions for data engineers](https://github.com/OBenner/data-engineering-interview-questions)

![](../../.gitbook/assets/11)\


1. [More data engineering questions](https://realpython.com/data-engineer-interview-questions-python/)

![](../../.gitbook/assets/12)\


1. [Even more qs](https://www.softwaretestinghelp.com/data-engineer-interview-questions/)

![](../../.gitbook/assets/13)

References:

1. [DS leads](https://docs.google.com/document/d/1gdfJce0p7jx0ptHJt3NE3NIhvzZCXJ5O0V-dvJse0HI/edit)
2. [System design interview q’s with solutions](https://github.com/donnemartin/system-design-primer#system-design-interview-questions-with-solutions)
3. [Cap theorem](https://github.com/donnemartin/system-design-primer#system-design-interview-questions-with-solutions), [2](https://github.com/henryr/cap-faq) (which is great), [3](https://github.com/kislerdm/data-engineering-interviews) (isn't complete)
4. [ACID](https://bardoloi.com/blog/2017/02/26/db-deep-dive/), [CAP](https://bardoloi.com/blog/2017/03/06/cap-theorem/), [PACLEC](https://bardoloi.com/blog/2017/03/06/pacelc-theorem/)
5. [Why do we need Data engineering?](https://podcastaddict.com/episode/116229803) (podcast)
