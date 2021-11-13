# Lakes & Warehouses

## Definitions

1. [What is a DWH? a comprehensive guide](https://www.oracle.com/database/what-is-a-data-warehouse/)
2. [DataLakeHouse](https://www.firebolt.io/blog/snowflake-vs-databricks-vs-firebolt) - Databricks Delta Lake is a data lake that can store raw unstructured, semi-structured, and structured data. When combined with Delta Engine it becomes a data lakehouse.
3. [What is SnowFlake](https://www.stitchdata.com/resources/snowflake/), [2](https://www.slalom.com/insights/snowflake-implementation-success) - Snowflake decouples the storage and compute functions, which means organizations that have high storage demands but less need for CPU cycles, or vice versa, don’t have to pay for an integrated bundle that requires them to pay for both. Users can scale up or down as needed and pay for only the resources they use.
4.

## Comparisons

1.  [Data Lake vs Data Warehouse](https://www.talend.com/resources/data-lake-vs-data-warehouse/)

    ![](<../.gitbook/assets/image (12) (1) (1).png>)
2. [Top 5 differences between DL & DWH](https://www.bluegranite.com/blog/bid/402596/top-five-differences-between-data-lakes-and-data-warehouses)
3.  [Amazon on DL vs DWH](https://aws.amazon.com/big-data/datalakes-and-analytics/what-is-a-data-lake/)

    ![](<../.gitbook/assets/image (13) (1).png>)
4.  [Snowflake vs Delta Lake vs Fire Bolt](https://www.firebolt.io/blog/snowflake-vs-databricks-vs-firebolt) - "Databricks Delta Lake and Delta Engine is a lakehouse. You choose it as a data lake, and for data lakehouse-based workloads including ELT for data warehouses, data science and machine learning, even static reporting and dashboards if you don’t mind the performance difference and don’t have a data warehouse.\


    Most companies still choose a data warehouse like Snowflake, BigQuery, Redshift or Firebolt for general-purpose analytics over a data lakehouse like Delta Lake and Delta Engine because they need performance.\


    But it doesn’t matter. You need more than one engine. Don’t fight it. You will end up with multiple engines for very good reasons. It’s just a matter of when. "
5. [Snowflake vs Amazon Redshift](https://www.sphereinc.com/blogs/snowflake-vs-aws-redshift-which-should-you-use-for-your-data-warehouse/)

## Use Cases

1. [Hunters on their architecture](https://www.youtube.com/watch?v=S78gCJ3tdc4), airflow, snowflake, snowpipe, flink, rockdb, cluster optimization during ingestion, monitoring metrics, cost.
