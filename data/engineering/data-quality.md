# Data Quality

Bad rows break models and dashboards, so quality has to be defined and tested, not assumed.
This page is what quality means, then dbt as the tool that tests it, then data validation notes.
The same notes are in [Data Contract](data-contract.md), [Data Lineage](data-lineage.md), [Data Observability](data-observability.md), [Data Testing](../../evals/data-and-model-tests.md#data-testing), and [Data Validation](data-quality.md#data-validation).

- [why cant data quality be fixed with tech](https://www.analytics8.com/blog/why-data-quality-cannot-be-fixed-with-technology/)
- Learn what data quality really means, how to measure your data quality, and some common best practices to follow and pitfalls to avoid. [CloverDX - data quality](https://www.cloverdx.com/explore/data-quality)
- Discover the 6 core data quality metrics, how they differ from dimensions and KPIs, and how to measure and track them for reliable AI outcomes. [CloverDX - 6 Data Quality Metrics You Can't Afford To Ignore](https://www.cloverdx.com/blog/6-data-quality-metrics-you-cant-ignore)
- Learn key data quality metrics, examples, and best practices to improve data quality, accuracy, and reliability on Azure & Snowflake, by Seth Rao. [FirstEigan - data quality metrics](https://firsteigen.com/blog/6-key-data-quality-metrics-you-should-be-tracking/)
- 10 Data Quality Metrics You Should Measure - Data Ladder. [Dataladder - data quality metrics](https://dataladder.com/10-data-quality-metrics-you-should-measure/)
- Data Quality - Metrics. Data Quality - Metrics. [Datacademia - data quality metrics](https://datacadamia.com/data/quality/metric)
- we tell you everything you need to know about the Four V's of Big Data, by Rank Math. [Open sistemas - big data challenges - veracity](https://opensistemas.com/en/the-four-vs-of-big-data/)

## DBT

After the quality articles, this section is dbt testing tools.

The same notes are in [Snowflake](lakes-and-warehouses.md#snowflake) and [Tools](tools.md).

- Configure dbt data tests to assess the quality of your input data and ensure accuracy in resulting datasets. [DBT builtin tests](https://docs.getdbt.com/docs/building-a-dbt-project/tests)
- dbt - Package hub. dbt - Package hub. [DBT expectations](https://hub.getdbt.com/calogica/dbt_expectations/0.1.2/)
- Learn how to define your own custom generic data tests. [DBT custom generic tests](https://docs.getdbt.com/guides/legacy/writing-custom-generic-tests)
- Introduction to Jinja Templating, by Real Python. [Jinja](https://www.youtube.com/watch?v=OraYXEr0Irg)


## Data Validation

After dbt, this section is data validation notes from the question bank.

The same notes are in [Data Quality](data-quality.md), [Data Testing](../../evals/data-and-model-tests.md#data-testing), and [Tools](tools.md).

1. How can you protect yourself from bad data? Data validation, TDDA, monitoring.
2. Tools:
 - GitHub - agronholm/typeguard: Run-time type checker for Python. Type validation: [typeguard](https://github.com/agronholm/typeguard)
 - Dataclasses. Dataclasses. Data validation [pydantic](https://pydantic-docs.helpmanual.io/usage/dataclasses/)
 - GitHub - tdda/tdda: Test-Driven Data Analysis Functions. Test driven: [tdda](https://github.com/tdda/tdda)
 - GX Core is an open source framework for testing, validating, and documenting data quality across modern data pipelines, workflows, and teams. Data quality: [great expectations](https://greatexpectations.io/)
 - GX Core is an open source framework for testing, validating, and documenting data quality across modern data pipelines, workflows, and teams. Saas: [SuperConductive by GE](https://superconductive.ai/)
