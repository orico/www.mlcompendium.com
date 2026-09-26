# Databases

Data has to live somewhere queryable, and the choice is lake, warehouse, table format, or a specialized engine.
This page moves from lakes and warehouses, through comparisons and NoSQL versus relational, into table formats and file formats, then Snowflake, ClickHouse, vector databases, feature engineering, and use cases.

## Lakes and warehouses

The first choice is the shape of the store itself, and the warehouse is where most analytics teams start. [What is a DWH? a comprehensive guide](https://www.oracle.com/database/what-is-a-data-warehouse/) is Oracle's "What Is a Data Warehouse?", on the latest in data warehousing and how it can benefit your business. The [DataLakeHouse](https://www.firebolt.io/blog/snowflake-vs-databricks-vs-firebolt) link now opens the Firebolt team's blog of news, updates, and engineering posts. The idea it pointed at is that Delta Lake is a data lake that can store raw unstructured, semi-structured, and structured data. When combined with Delta Engine it becomes a data lakehouse.

Snowflake is the warehouse that keeps coming up on this page, so it is worth defining early. [What is SnowFlake](https://www.stitchdata.com/resources/snowflake/), [2](http://web.archive.org/web/20230401004452/https://www.slalom.com/insights/snowflake-implementation-success) — Snowflake decouples the storage and compute functions, which means organizations that have high storage demands but less need for CPU cycles, or vice versa, don’t have to pay for an integrated bundle that requires them to pay for both. Users can scale up or down as needed and pay for only the resources they use. The first of those is Stitch on Snowflake as built for the cloud from the ground up, and the second is Slalom's four keys to success with Snowflake. To [get started with SF](https://www.phdata.io/blog/getting-started-with-snowflake/), phData's guide explores the Modern Data Stack, its technologies, strategies for migration, and best practices.

A warehouse is often too broad for one team, which is where the data mart comes in. [talend on data marts](https://www.talend.com/resources/what-is-data-mart/) defines a data mart as a structured data repository purpose-built to support the analytical needs of a particular department, line of business, or geographic region. (good) [netsuite on data marts](https://www.netsuite.com/portal/resource/articles/data-warehouse/data-mart.shtml) — the three types above, plus structures (star, snowflake, denormalized) and comparisons. The [basic intro](https://study.com/academy/lesson/what-is-a-data-mart-design-types-example.html) is the study.com lesson.

The opposite of the curated mart is the data lake, and a lake needs its health watched. The guide to [monitoring health status at scale](https://medium.com/data-science/how-to-monitor-data-lake-health-status-at-scale-d0eb058c85aa) using Great Expectations and Spark is by Davide Romano.

The same notes are in [Data & Model Tests](../../evals/data-and-model-tests.md).

## Comparisons

With lake, warehouse, and mart defined, the next question is which one to pick. [Data Lake vs Data Warehouse](https://www.talend.com/resources/data-lake-vs-data-warehouse/) is Qlik's six key differences: a data lake holds structured and unstructured data, while a data warehouse holds highly structured data processed for a defined purpose. The figure below is that comparison.

<figure><img src="../../ops/.gitbook/assets/image (14).png" alt=""><figcaption><p>Data Lake vs Data Warehouse</p></figcaption></figure>

A BlueGranite list of the top 5 differences between DL and DWH used to sit here; that address now points to an unrelated site and is kept at the end of the page. [Amazon on DL vs DWH](https://aws.amazon.com/big-data/datalakes-and-analytics/what-is-a-data-lake/) is AWS on what a data lake is, how and why businesses use one, and how to use it with AWS; the figure below is from that comparison.

<figure><img src="../../ops/.gitbook/assets/image (36).png" alt=""><figcaption><p>Amazon on DL vs DWH</p></figcaption></figure>

The lakehouse blurs the line, and the Firebolt comparison argues you will not pick just one. [Snowflake vs Delta Lake vs Fire Bolt](https://www.firebolt.io/blog/snowflake-vs-databricks-vs-firebolt) now lands on the Firebolt team's blog; the passage kept from it reads:

> Databricks Delta Lake and Delta Engine is a lakehouse. You choose it as a data lake, and for data lakehouse-based workloads including ELT for data warehouses, data science and machine learning, even static reporting and dashboards if you don’t mind the performance difference and don’t have a data warehouse.
>
> Most companies still choose a data warehouse like Snowflake, BigQuery, Redshift or Firebolt for general-purpose analytics over a data lakehouse like Delta Lake and Delta Engine because they need performance.
>
> But it doesn’t matter. You need more than one engine. Don’t fight it. You will end up with multiple engines for very good reasons. It’s just a matter of when.

Once the choice is a warehouse, the question narrows to which engine. [Snowflake vs Amazon Redshift](https://www.sphereinc.com/blogs/snowflake-vs-aws-redshift-which-should-you-use-for-your-data-warehouse/) is Sphere Partners' comparison of the two cloud warehouses, their benefits and functions, and the Snowflake AWS marketplace. For Snowflake on its own there is an [Intro and demo](https://www.youtube.com/watch?v=dUL8GO4ZK9s) video, and [The three pillars - snowflake](https://medium.com/data-science/why-you-need-to-know-snowflake-as-a-data-scientist-d4e5a87c2f3d) is "Why You Need To Know Snowflake As A Data Scientist", written for readers who may be hearing of the company for the first time. [Snowflake vs redshift on medium](https://medium.com/data-science/redshift-or-snowflake-e0e3ea427dbc) is "Redshift or Snowflake!". [SF vs RS](https://www.xplenty.com/blog/redshift-vs-snowflake/) is Mark Smallcombe on the critical differences between Amazon Redshift and the Snowflake data warehouse. Donal Tobin wrote the other two pairings: [RS vs BQ](https://www.xplenty.com/blog/redshift-vs-bigquery-comprehensive-guide/), on choosing the right data warehouse as a critical component of your general data and analytic business needs, and [SF vs BQ](https://www.xplenty.com/blog/snowflake-vs-bigquery/), Snowflake vs. BigQuery.

## NoSQL vs relational

Lakes and warehouses are for analytics; the application behind them still has to choose between a document store and a relational database. The same notes are in [Database Modeling](database-architecture-and-modeling.md).

The question is kept as an interview question:

Explain the difference and the reason to choose using NoSQL {mongoDB | DynamoDB | .. } over Relational database {Postgress |MySQL} and vice versa. Give an example for a project where you had to make this choice, and walk through your reasoning.

(This question can be modified for the relevant technologies.)

## Data lake table formats

A lake of raw files only behaves like a database when a table format sits on top of it. The same notes are in [File formats](lakes-and-warehouses.md#file-formats) and [Tools](tools.md).

### Apache Iceberg

Iceberg is the table format with the most notes here, from first introduction to production cost. [A short Intro](https://medium.com/expedia-group-tech/a-short-introduction-to-apache-iceberg-d34f628b6799) is Christine Mathiesen's short introduction to Apache Iceberg, and [A Primer](https://thedatafreak.medium.com/apache-iceberg-a-primer-75a63470bfa2) is Ani's primer.

Performance is where the formats get compared. The earlier post, [delta vs iceberg](https://databeans-blogs.medium.com/delta-vs-iceberg-performance-as-a-decisive-criteria-add7bcdde03d), treats performance as the deciding criterion and describes a data lakehouse as an open architecture that brings the scalability and cost-effectiveness of data lakes together with the reliability and performance of data warehouses on one platform. After comparing delta vs iceberg in our previous blog, a lot of people asked for benchmarking their latest versions and for Apache Hudi to be thrown into the mix, and that follow-up is [Benchmarking Delta vs Iceberg vs Hudi](https://databeans-blogs.medium.com/delta-vs-iceberg-vs-hudi-reassessing-performance-cb8157005eb0).

Production is the last test. [How we migrated our production data lake to iceberg](https://medium.com/insiderengineering/how-we-migrated-our-production-data-lake-to-apache-iceberg-4d6892eca6e6) is Deniz Parmaks on moving tens of terabytes in Amazon S3 from Apache Hive to Apache Iceberg, and [How we reduced our cost by 90%](https://medium.com/insiderengineering/apache-iceberg-reduced-our-amazon-s3-cost-by-90-997cde5ce931) is the companion post on how Iceberg reduced their Amazon S3 cost by 90%. [Top 5 Features](https://dipankar-tnt.medium.com/apache-iceberg-features-101-331a254a7ada) is Dipankar Mazumdar on why choosing a table format is one of the most critical decisions in a cloud data lake architecture, and Iceberg's top five features.

### Databricks Delta Lake

Delta Lake is the other format in those benchmarks, and its question is how it connects to the rest of the stack. The same notes are in [Databricks](../../ai-engineering/mlops/experiment-management.md#databricks).

[integrating delta lake into other platforms](https://www.databricks.com/blog/integrating-delta-lakehouse-other-platforms) is Databricks on integrating Delta Lakehouse with various platforms to unify the data ecosystem and enhance data management and analytics.

## File formats

Under every table format is a file format, usually Parquet, and these are the interview questions about it. The same notes are in [Data lake table formats](lakes-and-warehouses.md#data-lake-table-formats).

1. Can you explain the parquet file format?
2. How is this leveraged by Spark? [https://databricks.com/session/spark-parquet-in-depth](https://databricks.com/session/spark-parquet-in-depth) How this leveraged Spark
3. What are the shortcomings of parquet and how is it solved by file formats like hudi, delta, iceberg? [https://lakefs.io/hudi-iceberg-and-delta-lake-data-lake-table-formats-compared/](https://lakefs.io/hudi-iceberg-and-delta-lake-data-lake-table-formats-compared/) answers it by comparing hudi, iceberg, and delta.

## Snowflake

After the formats, the page returns to Snowflake, the warehouse that most of the comparisons above measured against. The same notes are in [DBT](data-quality.md#dbt) and [Tools](tools.md).

[(good) Guides](https://www.snowflake.com/guides/) is Snowflake's AI Data Cloud Fundamentals, the resource for foundational AI, cloud, and data concepts. [getting started with SF tasks](https://medium.com/snowflake/getting-started-with-snowflake-tasks-945ecd54c77b) SQL procedures schedules tree tasks, by Rajiv Gupta. For the bill, [cost](https://www.phdata.io/blog/what-is-the-snowflake-data-cloud/) is Justin Delisi on what the Snowflake Data Cloud is, how much it costs, and how it breaks down data silos with stronger data sharing and data exchange.

## ClickHouse

Snowflake is a managed warehouse; ClickHouse is the open-source engine for when reports have to be real-time. The [open source database for real time apps and analytics](https://clickhouse.com/) is ClickHouse, a fast open-source column-oriented database management system that generates analytical data reports in real-time using SQL queries.

## Vector databases

After the OLAP engines, embeddings need their own kind of store. The same notes are in [RAG](../../generative-ai/rag.md), [Search](../../language-ai/search.md), and [VECTOR SIMILARITY SEARCH](../../deep-learning/representations.md#vector-similarity-search).

The analyst view is Gartner — [Innovation Insight: Vector Databases](https://www.gartner.com/doc/reprints?id=1-2HBZK5EN&ct=240418&st=sb), a Gartner reprint. The open-source option is Chroma — AI native open source embedding database, on [github](https://github.com/chroma-core/chroma), where it now calls itself search infrastructure for AI.

## Feature engineering

With storage named, the warehouse can also do some of the model's work. The same notes are in [FEATURE ENGINEERING](../feature-engineering.md#feature-engineering).

[Feature engineering in snowflake](https://medium.com/data-science/feature-engineering-in-snowflake-1730a1b84e5b) is James Weakley's follow-up to showing that a true cloud data warehouse can handle common machine learning tasks for structured data, like training tree-based models.

## Use cases

Closing the page, one architecture talk puts these systems together. [Hunters on their architecture](https://www.youtube.com/watch?v=S78gCJ3tdc4), Airflow, Snowflake, Snowpipe, Flink, RocksDB, cluster optimization during ingestion, monitoring metrics, cost.

Two notes from earlier sections also had towardsdatascience.com addresses, kept here. Monitoring health status at scale is Davide Romano's guide to building a Data Quality workflow with Great Expectations and Spark: [https://towardsdatascience.com/how-to-monitor-data-lake-health-status-at-scale-d0eb058c85aa](https://towardsdatascience.com/how-to-monitor-data-lake-health-status-at-scale-d0eb058c85aa). The three pillars - snowflake is Admond Lee on the only data warehouse built for the cloud: [https://towardsdatascience.com/why-you-need-to-know-snowflake-as-a-data-scientist-d4e5a87c2f3d](https://towardsdatascience.com/why-you-need-to-know-snowflake-as-a-data-scientist-d4e5a87c2f3d)


## Deprecated links

{% hint style="warning" %}
These links and images no longer work. The original wording is kept here. A same-resource copy, when one was checked, is used above.
{% endhint %}

- 2. This address no longer opens: https://www.slalom.com/insights/snowflake-implementation-success
- Snowflake vs redshift on medium. This address no longer opens: https://towardsdatascience.com/redshift-or-snowflake-e0e3ea427dbc
- Feature engineering in snowflake. This address no longer opens: https://towardsdatascience.com/feature-engineering-in-snowflake-1730a1b84e5b
- Top 5 differences between DL and DWH. This address now points to an unrelated site: https://www.bluegranite.com/blog/bid/402596/top-five-differences-between-data-lakes-and-data-warehouses
