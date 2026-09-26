# Databases

Data has to live somewhere queryable, and the choice is lake, warehouse, table format, or a specialized engine.
This page moves from lakes and warehouses, through comparisons and NoSQL versus relational, into table formats and file formats, then Snowflake, ClickHouse, vector databases, feature engineering, and use cases.
## Lakes and warehouses

This section introduces data warehouses, lakehouses, Snowflake, data marts, and data lakes.

- Learn the latest on data warehouse and how it can benefit your business. [What is a DWH? a comprehensive guide](https://www.oracle.com/database/what-is-a-data-warehouse/)
- News, updates, and engineering posts from the Firebolt team. [DataLakeHouse](https://www.firebolt.io/blog/snowflake-vs-databricks-vs-firebolt)
 - Delta Lake is a data lake that can store raw unstructured, semi-structured, and structured data. When combined with Delta Engine it becomes a data lakehouse.
3. [What is SnowFlake](https://www.stitchdata.com/resources/snowflake/), [2](http://web.archive.org/web/20230401004452/https://www.slalom.com/insights/snowflake-implementation-success) — Snowflake decouples the storage and compute functions, which means organizations that have high storage demands but less need for CPU cycles, or vice versa, don’t have to pay for an integrated bundle that requires them to pay for both. Users can scale up or down as needed and pay for only the resources they use.
 - This guide explores the Modern Data Stack, its technologies, effective strategies for seamless migration, and best practices from phData. [get started with SF](https://www.phdata.io/blog/getting-started-with-snowflake/)
4. Data mart
 - Data mart is a structured data repository purpose-built to support the analytical needs of a particular department, line of business, or geographic region. [talend on data marts](https://www.talend.com/resources/what-is-data-mart/)
 - (good) [netsuite on data marts](https://www.netsuite.com/portal/resource/articles/data-warehouse/data-mart.shtml) — the three types above, plus structures (star, snowflake, denormalized) and comparisons
 - study.com. study.com. [basic intro](https://study.com/academy/lesson/what-is-a-data-mart-design-types-example.html)
5. Data lake
 - [monitoring health status at scale](https://medium.com/data-science/how-to-monitor-data-lake-health-status-at-scale-d0eb058c85aa) using Great Expectations and Spark

The same notes are in [Data & Model Tests](../../evals/data-and-model-tests.md).

## Comparisons

After the lake and warehouse definitions, this section compares lakes, warehouses, and major warehouse engines.

- A data lake holds structured and unstructured data. [Data Lake vs Data Warehouse](https://www.talend.com/resources/data-lake-vs-data-warehouse/)

<figure><img src="../../ops/.gitbook/assets/image (14).png" alt=""><figcaption><p>Data Lake vs Data Warehouse</p></figcaption></figure>

- Silver Jackpot - PR agency, by Rank Math. [Top 5 differences between DL and DWH](https://www.bluegranite.com/blog/bid/402596/top-five-differences-between-data-lakes-and-data-warehouses)
- What is a Data Lake how and why businesses use Data Lake, and how to use Data Lake with AWS. [Amazon on DL vs DWH](https://aws.amazon.com/big-data/datalakes-and-analytics/what-is-a-data-lake/)

<figure><img src="../../ops/.gitbook/assets/image (36).png" alt=""><figcaption><p>Amazon on DL vs DWH</p></figcaption></figure>

- News, updates, and engineering posts from the Firebolt team. [Snowflake vs Delta Lake vs Fire Bolt](https://www.firebolt.io/blog/snowflake-vs-databricks-vs-firebolt)

> Databricks Delta Lake and Delta Engine is a lakehouse. You choose it as a data lake, and for data lakehouse-based workloads including ELT for data warehouses, data science and machine learning, even static reporting and dashboards if you don’t mind the performance difference and don’t have a data warehouse.
>
> Most companies still choose a data warehouse like Snowflake, BigQuery, Redshift or Firebolt for general-purpose analytics over a data lakehouse like Delta Lake and Delta Engine because they need performance.
>
> But it doesn’t matter. You need more than one engine. Don’t fight it. You will end up with multiple engines for very good reasons. It’s just a matter of when.

- Snowflake vs Redshift - Sphere Partners. Snowflake vs Redshift - Sphere Partners. [Snowflake vs Amazon Redshift](https://www.sphereinc.com/blogs/snowflake-vs-aws-redshift-which-should-you-use-for-your-data-warehouse/)
- Snowflake [Intro and demo](https://www.youtube.com/watch?v=dUL8GO4ZK9s)
- [The three pillars - snowflake](https://medium.com/data-science/why-you-need-to-know-snowflake-as-a-data-scientist-d4e5a87c2f3d)
- [Snowflake vs redshift on medium](https://medium.com/data-science/redshift-or-snowflake-e0e3ea427dbc)
- The critical difference of Amazon Redshift vs, by Mark Smallcombe. [SF vs RS](https://www.xplenty.com/blog/redshift-vs-snowflake/)
- Choosing the right data warehouse is a critical component of your general data and analytic business needs, by Donal Tobin. [RS vs BQ](https://www.xplenty.com/blog/redshift-vs-bigquery-comprehensive-guide/)
- Snowflake vs. BigQuery, by Donal Tobin. [SF vs BQ](https://www.xplenty.com/blog/snowflake-vs-bigquery/)

## NoSQL vs relational

Beside the comparisons, this section asks when to choose NoSQL over a relational database and the reverse.

The same notes are in [Database Modeling](database-architecture-and-modeling.md).

Explain the difference and the reason to choose using NoSQL {mongoDB | DynamoDB | .. } over Relational database {Postgress |MySQL} and vice versa. Give an example for a project where you had to make this choice, and walk through your reasoning.

(This question can be modified for the relevant technologies.)

## Data lake table formats

After the NoSQL question, this section covers open table formats used on data lakes.

The same notes are in [File formats](lakes-and-warehouses.md#file-formats) and [Tools](tools.md).

### Apache Iceberg

This subsection lists introductions, benchmarks, and migration notes for Apache Iceberg.

- [A short Intro](https://medium.com/expedia-group-tech/a-short-introduction-to-apache-iceberg-d34f628b6799)
- [A Primer](https://thedatafreak.medium.com/apache-iceberg-a-primer-75a63470bfa2)
- After comparing [delta vs iceberg](https://databeans-blogs.medium.com/delta-vs-iceberg-performance-as-a-decisive-criteria-add7bcdde03d) in our previous blog, a lot of people asked for benchmarking their latest versions and for Apache. [Benchmarking Delta vs Iceberg vs Hudi](https://databeans-blogs.medium.com/delta-vs-iceberg-vs-hudi-reassessing-performance-cb8157005eb0)
- [How we migrated our production data lake to iceberg](https://medium.com/insiderengineering/how-we-migrated-our-production-data-lake-to-apache-iceberg-4d6892eca6e6)
- [How we reduced our cost by 90%](https://medium.com/insiderengineering/apache-iceberg-reduced-our-amazon-s3-cost-by-90-997cde5ce931)
- [Top 5 Features](https://dipankar-tnt.medium.com/apache-iceberg-features-101-331a254a7ada)

### Databricks Delta Lake

This subsection points at integrating Delta Lake with other platforms.

The same notes are in [Databricks](../../ai-engineering/mlops/experiment-management.md#databricks).

- Discover how to integrate Delta Lakehouse with various platforms to enhance data management and analytics capabilities. [integrating delta lake into other platforms](https://www.databricks.com/blog/integrating-delta-lakehouse-other-platforms)

## File formats

Beside the table formats, this section covers Parquet, Spark, and lake table formats such as Hudi, Delta, and Iceberg.

The same notes are in [Data lake table formats](lakes-and-warehouses.md#data-lake-table-formats).

1. Can you explain the parquet file format?
- How is this leveraged by Spark? [https://databricks.com/session/spark-parquet-in-depth](https://databricks.com/session/spark-parquet-in-depth) How this leveraged Spark
3. What are the shortcomings of parquet and how is it solved by file formats like hudi, delta, iceberg? [https://lakefs.io/hudi-iceberg-and-delta-lake-data-lake-table-formats-compared/](https://lakefs.io/hudi-iceberg-and-delta-lake-data-lake-table-formats-compared/)

## Snowflake

After the file formats, this section lists Snowflake guides, tasks, and cost notes.

The same notes are in [DBT](data-quality.md#dbt) and [Tools](tools.md).

- [(good) Guides](https://www.snowflake.com/guides/)
- [getting started with SF tasks](https://medium.com/snowflake/getting-started-with-snowflake-tasks-945ecd54c77b) SQL procedures schedules tree tasks
- Learn how the Snowflake Data Cloud breaks down data silos and increases efficiency with stronger data sharing and data exchange, by Justin Delisi. [cost](https://www.phdata.io/blog/what-is-the-snowflake-data-cloud/)

## ClickHouse

Beside Snowflake, this section points at ClickHouse as an open source database for real-time apps and analytics.

- ClickHouse is a fast open-source column-oriented database management system that allows generating analytical data reports in real-time using SQL queries. [open source database for real time apps and analytics](https://clickhouse.com/)

## Vector databases

After the OLAP engines, this section lists a Gartner note and Chroma for embeddings.

The same notes are in [RAG](../../generative-ai/rag.md), [Search](../../language-ai/search.md), and [VECTOR SIMILARITY SEARCH](../../deep-learning/representations.md#vector-similarity-search).

- Gartner — [Innovation Insight: Vector Databases](https://www.gartner.com/doc/reprints?id=1-2HBZK5EN&ct=240418&st=sb)
- Chroma — AI native open source embedding database. [github](https://github.com/chroma-core/chroma)

## Feature engineering

With storage named, this section points at feature engineering in Snowflake.

The same notes are in [FEATURE ENGINEERING](../feature-engineering.md#feature-engineering).

- [Feature engineering in snowflake](https://medium.com/data-science/feature-engineering-in-snowflake-1730a1b84e5b)

## Use cases

Closing the page, this section points at one architecture talk that uses these systems.

- [Hunters on their architecture](https://www.youtube.com/watch?v=S78gCJ3tdc4), Airflow, Snowflake, Snowpipe, Flink, RocksDB, cluster optimization during ingestion, monitoring metrics, cost.

- Build a Data Quality workflow with Great Expectations and Spark, by Davide Romano. monitoring health status at scale. [https://towardsdatascience.com/how-to-monitor-data-lake-health-status-at-scale-d0eb058c85aa](https://towardsdatascience.com/how-to-monitor-data-lake-health-status-at-scale-d0eb058c85aa)
- The only data warehouse built for the cloud, by Admond Lee. The three pillars - snowflake. [https://towardsdatascience.com/why-you-need-to-know-snowflake-as-a-data-scientist-d4e5a87c2f3d](https://towardsdatascience.com/why-you-need-to-know-snowflake-as-a-data-scientist-d4e5a87c2f3d)


## Deprecated links

{% hint style="warning" %}
These links and images no longer work. The original wording is kept here. A same-resource copy, when one was checked, is used above.
{% endhint %}

- 2. This address no longer opens: https://www.slalom.com/insights/snowflake-implementation-success
- Snowflake vs redshift on medium. This address no longer opens: https://towardsdatascience.com/redshift-or-snowflake-e0e3ea427dbc
- Feature engineering in snowflake. This address no longer opens: https://towardsdatascience.com/feature-engineering-in-snowflake-1730a1b84e5b
