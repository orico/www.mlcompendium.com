# Data Pipelines

Data has to move from source systems into a place a model or warehouse can use, and the choice is which integration path and which orchestrator owns that path.
This page starts with the pipeline and integration notes, then Streaming vs Batch, Airflow, and CDC.
The same notes are in [Airflow](../../ai-engineering/mlops/full-stack-and-ops.md#airflow) and [General](../engineering/data-engineering-questions.md#general).

The first choice is the integration tool, and the market is crowded. Metaplane's [Decision guide for data integration tools](https://www.metaplane.dev/blog/decision-guide-to-choosing-a-data-integration-tool) is the guide for choosing the right solution in it.

Before picking a tool, the order of the steps matters. An ETL pipeline is a type of data pipeline in which a set of processes extracts data from one system, transforms it, and loads it into a target repository; Qlik's [ETL vs ELT](https://www.qlik.com/us/etl/etl-vs-elt) compares that with a general data pipeline, and Guru99's ETL vs ELT – Difference Between Them is the [2](https://www.guru99.com/etl-vs-elt.html) second explanation. The data engineering industry has evolved from Extract, Transform, Load (ETL) to ELT, where raw data is copied from the source system and loaded into a data store first, and [What is reverse ETL (ELT)](https://torbjornzetterlund.com/what-is-reverse-etl/) picks up from that shift.

With the pattern chosen, the tools are the ones that carry data in. [Airbyte](https://airbyte.com/) — open-source ELT data integration for modern data teams. Apache Airflow, Rivery and Stitch are all popular ETL tools for data ingestion into cloud data warehouses, and [Stitch vs Airflow vs Rivery](https://www.stitchdata.com/vs/airflow/rivery/) is the quick guide comparing their features, pricing, and services. Rivery and Fivetran are the other names on the shelf. Fivetran connects to Airflow directly: a new Python package from Fivetran and Astronomer enables connector management in Airflow, announced as [Operators for AirFlow](https://www.fivetran.com/blog/announcing-the-fivetran-airflow-provider). Astronomer's own guide to the best ETL tools to integrate with Airflow, on how to select one and why it pays to integrate Fivetran, Airbyte, and Azure Data Factory with Apache Airflow®, is [Astronomer AirFlow as an orchestrator scheduler for FT, AB, etc](https://www.astronomer.io/blog/best-etl-tools-airflow/).

A tool only helps if the pipeline stays reliable. [The 7 Principles of reliable data pipelines](https://medium.com/bigeye/seven-principles-for-reliable-data-pipelines-e82a82810e4f) Bigeye Medium article opens with the familiar pressure: “Yea when can you have that delivered by? We wanna do the science part soon.” For teams that want no code at all, [Hevo](https://hevodata.com/integrations/pipeline/) — Hevo is a Fully Automated, No-code Data Pipeline Platform that supports 150+ ready-to-use integrations across Databases, SaaS Applications, Cloud Storage, SDKs, and Streaming Services.

## Streaming vs Batch

Every one of those pipelines still has to decide how often data moves. The question to be able to answer is this: explain the difference and the reason to choose using Streaming over Batch and vice versa. Give an example for a project where you had to make this choice, and walk through your reasoning.

## Airflow

Whichever cadence wins, something has to schedule it, and Airflow is the orchestrator named throughout the tool shelf above. The same notes are in [Airflow](../../ai-engineering/mlops/full-stack-and-ops.md#airflow).

The questions to be able to answer about it are:

1. What is Airflow?
2. How do you transfer information between tasks in Airflow?
3. Please give me a real-world example of using Spark and Airflow together

## CDC

An orchestrator moves whole batches; change data capture moves only what changed. The same notes are in [Tools](../engineering/tools.md).

[What is a CDC and why do you need it, or how do you use it?](https://rockset.com/blog/change-data-capture-what-it-is-and-how-to-use-it/) — Change data capture (CDC) is the process of recognising when data has been changed in a source system so a downstream process or system can action. A common use case is to reflect (replication) the change in a different target system so that the data in the systems stay in sync.
