# Lakes & Warehouses

## Definitions

1. [What is a DWH? a comprehensive guide](https://www.oracle.com/database/what-is-a-data-warehouse/)
2. [DataLakeHouse](https://www.firebolt.io/blog/snowflake-vs-databricks-vs-firebolt) - Databricks Delta Lake is a data lake that can store raw unstructured, semi-structured, and structured data. When combined with Delta Engine it becomes a data lakehouse.
3. [What is SnowFlake](https://www.stitchdata.com/resources/snowflake/), [2](https://www.slalom.com/insights/snowflake-implementation-success) - Snowflake decouples the storage and compute functions, which means organizations that have high storage demands but less need for CPU cycles, or vice versa, don’t have to pay for an integrated bundle that requires them to pay for both. Users can scale up or down as needed and pay for only the resources they use.
   1. [get started with SF](https://www.phdata.io/blog/getting-started-with-snowflake/)
4. data mart
   1. [talend on data marts](https://www.talend.com/resources/what-is-data-mart/) + 3 types (dependent, independent, hybrid)
   2. (good) [netsuite on data marts](https://www.netsuite.com/portal/resource/articles/data-warehouse/data-mart.shtml) - the three types ^ + structures (star, snowflake, denormalized) + comparisons
   3. [basic intro](https://study.com/academy/lesson/what-is-a-data-mart-design-types-example.html)
5. Data Lake
   1. [monitoring health status at scale](https://towardsdatascience.com/how-to-monitor-data-lake-health-status-at-scale-d0eb058c85aa) using great expectations and spark

## Comparisons

1.  [Data Lake vs Data Warehouse](https://www.talend.com/resources/data-lake-vs-data-warehouse/)

    ![](<../.gitbook/assets/image (12) (1) (1).png>)
2. [Top 5 differences between DL & DWH](https://www.bluegranite.com/blog/bid/402596/top-five-differences-between-data-lakes-and-data-warehouses)
3.  [Amazon on DL vs DWH](https://aws.amazon.com/big-data/datalakes-and-analytics/what-is-a-data-lake/)

    ![](<../.gitbook/assets/image (13) (1) (1) (1) (1).png>)
4.  [Snowflake vs Delta Lake vs Fire Bolt](https://www.firebolt.io/blog/snowflake-vs-databricks-vs-firebolt) - "Databricks Delta Lake and Delta Engine is a lakehouse. You choose it as a data lake, and for data lakehouse-based workloads including ELT for data warehouses, data science and machine learning, even static reporting and dashboards if you don’t mind the performance difference and don’t have a data warehouse.\


    Most companies still choose a data warehouse like Snowflake, BigQuery, Redshift or Firebolt for general-purpose analytics over a data lakehouse like Delta Lake and Delta Engine because they need performance.\


    But it doesn’t matter. You need more than one engine. Don’t fight it. You will end up with multiple engines for very good reasons. It’s just a matter of when. "
5. [Snowflake vs Amazon Redshift](https://www.sphereinc.com/blogs/snowflake-vs-aws-redshift-which-should-you-use-for-your-data-warehouse/)

## Use Cases

1. [Hunters on their architecture](https://www.youtube.com/watch?v=S78gCJ3tdc4), airflow, snowflake, snowpipe, flink, rockdb, cluster optimization during ingestion, monitoring metrics, cost.

## Snowflake

1. [(good) Guides ](https://www.snowflake.com/guides/)
2. [getting started with SF tasks](https://medium.com/snowflake/getting-started-with-snowflake-tasks-945ecd54c77b) - sql or procedures, schedules, B-tree tasks.
3. [cost](https://www.phdata.io/blog/what-is-the-snowflake-data-cloud/)

## Data lake Table Formats

### Apache Iceberg

1. [A short Intro](https://medium.com/expedia-group-tech/a-short-introduction-to-apache-iceberg-d34f628b6799)
2. [A Primer](https://thedatafreak.medium.com/apache-iceberg-a-primer-75a63470bfa2)
3. [Benchmarking Delta vs Iceberg vs Hudi](https://databeans-blogs.medium.com/delta-vs-iceberg-vs-hudi-reassessing-performance-cb8157005eb0)
4. [How we migrated our production data lake to iceberg](https://medium.com/insiderengineering/how-we-migrated-our-production-data-lake-to-apache-iceberg-4d6892eca6e6)
5. [How we reduced our cost by 90%](https://medium.com/insiderengineering/apache-iceberg-reduced-our-amazon-s3-cost-by-90-997cde5ce931)
6. [Top 5 Features](https://dipankar-tnt.medium.com/apache-iceberg-features-101-331a254a7ada)
