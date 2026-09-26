# Data Pipelines

Data has to move from source systems into a place a model or warehouse can use, and the choice is which integration path and which orchestrator owns that path.
This page starts with the pipeline and integration notes, then Streaming vs Batch, Airflow, and CDC.
The same notes are in [Airflow](../../ai-engineering/mlops/full-stack-and-ops.md#airflow) and [General](../engineering/data-engineering-questions.md#general).

- Get expert advice on choosing the right solution in today’s crowded market with our comprehensive guide. [Decision guide for data integration tools](https://www.metaplane.dev/blog/decision-guide-to-choosing-a-data-integration-tool)
- An ETL pipeline is a type of data pipeline in which a set of processes extracts data from one system, transforms it, and loads it into a target repository. [ETL vs ELT](https://www.qlik.com/us/etl/etl-vs-elt)
- ETL vs ELT – Difference Between Them. [2](https://www.guru99.com/etl-vs-elt.html)
- The data engineering industry has evolved from Extract, Transform, Load (ETL) to ELT, where raw data is copied from the source system loaded into a data... [What is reverse ETL (ELT)](https://torbjornzetterlund.com/what-is-reverse-etl/)
- [Airbyte](https://airbyte.com/) — open-source ELT data integration for modern data teams
- Apache Airflow, Rivery and Stitch are all popular ETL tools for data ingestion into cloud data warehouses. [Stitch vs Airflow vs Rivery](https://www.stitchdata.com/vs/airflow/rivery/)
- Rivery
- Fivetran
 - A new Python package from Fivetran and Astronomer enables connector management in Airflow. [Operators for AirFlow](https://www.fivetran.com/blog/announcing-the-fivetran-airflow-provider)
- Learn from our experts about how to select the best ETL tool, and why it pays to integrate Fivetran, Airbyte, and Azure Data Factory with Apache Airflow®. [Astronomer AirFlow as an orchestrator scheduler for FT, AB, etc](https://www.astronomer.io/blog/best-etl-tools-airflow/)
- [The 7 Principles of reliable data pipelines](https://medium.com/bigeye/seven-principles-for-reliable-data-pipelines-e82a82810e4f) Bigeye Medium article
- [Hevo](https://hevodata.com/integrations/pipeline/) — Hevo is a Fully Automated, No-code Data Pipeline Platform that supports 150+ ready-to-use integrations across Databases, SaaS Applications, Cloud Storage, SDKs, and Streaming Services.

## Streaming vs Batch

After the pipeline shelf, this section asks when to choose streaming over batch and the reverse.

Explain the difference and the reason to choose using Streaming over Batch and vice versa. Give an example for a project where you had to make this choice, and walk through your reasoning.

## Airflow

With batch and streaming named, this section lists Airflow basics and using it with Spark.

The same notes are in [Airflow](../../ai-engineering/mlops/full-stack-and-ops.md#airflow).

1. What is Airflow?
2. How do you transfer information between tasks in Airflow?
3. Please give me a real-world example of using Spark and Airflow together

## CDC

After Airflow, this section defines change data capture and why you need it.

The same notes are in [Tools](../engineering/tools.md).

[What is a CDC and why do you need it, or how do you use it?](https://rockset.com/blog/change-data-capture-what-it-is-and-how-to-use-it/) — Change data capture (CDC) is the process of recognising when data has been changed in a source system so a downstream process or system can action. A common use case is to reflect (replication) the change in a different target system so that the data in the systems stay in sync.
