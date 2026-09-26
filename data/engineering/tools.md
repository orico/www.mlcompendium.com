# Tools

Data work needs a shelf of tools for change capture, lakes, pipelines, BI, integration, and quality checks.
This page is that tool shelf, opened by the job each group is for, in the order data moves: captured at the source, landed in a lake, transformed, shown in BI, moved between systems, and finally tested.

The same notes are in [CDC](../processing/data-pipelines.md#cdc).
The same notes are in [Data lake table formats](lakes-and-warehouses.md#data-lake-table-formats).
The same notes are in [DBT](data-quality.md#dbt) and [Snowflake](lakes-and-warehouses.md#snowflake).
The same notes are in [Data Validation](data-quality.md#data-validation).

Data starts as changes in someone else's database, so the first tool captures those changes. [Debezium](https://debezium.io/) — an open source distributed platform for change data capture.

The captured changes have to land somewhere that can take a stream and still be queried in batch. [Hudi](https://hudi.apache.org/) is Apache Hudi, an open source data lake platform, described in its own words:

> Hudi is a rich platform to build streaming data lakes with incremental data pipelines on a self-managing database layer, while being optimized for lake engines and regular batch processing.

[Upsolver](https://www.upsolver.com/) takes the same lake and puts SQL pipelines on it:

> Continuous SQL Pipelines for Cloud Data Lakes. No custom coding. No orchestration. No infrastructure maintenance.

Once data is in the lake or warehouse, the transformations become code. [DBT](https://www.getdbt.com/) is dbt Labs, which empowers data teams to build reliable, governed data pipelines, accelerating analytics and AI initiatives with speed and confidence. The older pitch reads:

> dbt helps data teams work like software engineers—to ship trusted data, faster. collaboratively deploy analytics code following software engineering best practices like modularity, portability, CI/CD, and documentation. Now anyone who knows SQL can build production-grade data pipelines.

The videos that go with dbt run from a first look to production setups. The [intro](https://www.youtube.com/watch?v=R2nr1uZ8ffc) link now opens Streamlit's "1/4: What is Streamlit". The [in depth intro](https://www.youtube.com/watch?v=MHnJDqEKyUY) is AI Council's "Meet dbt: The Data Transformation Tool Used by JetBlue, GitLab, Wistia & Away", from Fishtown Analytics, and [dbt in one hour](https://www.youtube.com/watch?v=na6eu9WSXGY) is dbt Labs' dbt How-to Hour. Moving to deployment, [CI/CD with dbt](https://www.youtube.com/watch?v=snp2hxxWgqk) is HASHMAP on automation and continuous delivery with Snowflake using dbt. [snowflake terraform and dbt](https://www.youtube.com/watch?v=r4tAyiTgwRw) is dbt Labs' talk with Ritual on managing Snowflake architecture with Terraform and dbt, and [hubspot snowflake and dbt](https://www.youtube.com/watch?v=qDHgknWW_oo) is dbt Labs on building a productive data organization at Hubspot with Snowflake and dbt. When the transformation lives in Spark rather than SQL, the tool is [Metorikku](https://github.com/YotpoLtd/metorikku) — a simplified, lightweight ETL framework based on Apache Spark.

Transformed tables are only useful when people can see them, which is the job of BI tools that connect directly to a database. [redash](https://redash.io/) — connect and query your data sources, build dashboards to visualize data, and share them with your company. [Metabase](https://www.metabase.com/) grounds every answer in your semantic layer, your metrics and business logic, so you can inspect the query behind it. Growth FullStack's Metabase page describes it this way:

> [is an easy-to-use, open source business intelligence tool that lets you analyze data from a variety of data destinations and sources. It also follows a simple and fast setup process. Its data visualization capabilities are exceptional and can be showcased in a user-friendly way, without using SQL. With Metabase, you can easily share live dashboards, automated reports, and questions with the rest of your team.](https://growthfullstack.com/analyse/metabase-bi-tool/)

The third BI option is [Superset](https://superset.apache.org/) — Apache Superset is a modern data exploration and visualization platform.

Between the sources and the warehouse sit the tools that move and generate data without custom code. [Stitch](https://www.stitchdata.com/) — Stitch rapidly moves data from 130+ sources into a data warehouse so you can get to answers faster, no coding required. [SnowPlow](https://snowplowanalytics.com/) — generate [complete, accurate and well-structured event data](http://web.archive.org/web/20220601204852/https://snowplowanalytics.com/web-and-mobile-data/) across all platforms and channels in a common format, with the Snowplow Behavioral Data Platform. The archived page makes the case for richness, structure defined before collection, and accuracy, and the live site now calls Snowplow a customer context layer. [Workato](https://www.workato.com/) — a single platform for integration and workflow automation across your organization.

The last tool on the shelf checks that what arrived is right. [AWS Deequ](https://aws.amazon.com/blogs/big-data/test-data-quality-at-scale-with-deequ/) is the AWS Big Data Blog post on testing data quality at scale with Deequ.

## Deprecated links

{% hint style="warning" %}
These links and images no longer work. The original wording is kept here. A same-resource copy, when one was checked, is used above.
{% endhint %}

- Generate complete, accurate and well-structured event data. This address no longer opens: https://snowplowanalytics.com/web-and-mobile-data/
