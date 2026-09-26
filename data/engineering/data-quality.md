# Data Quality

Bad rows break models and dashboards, so quality has to be defined and tested, not assumed. The page starts with what quality means and how to measure it, then turns to dbt as the tool that tests it, and ends with data validation notes and tools.
The same notes are in [Data Contract](data-contract.md), [Data Lineage](data-lineage.md), [Data Observability](data-observability.md), [Data Testing](../../evals/data-and-model-tests.md#data-testing), and [Data Validation](data-quality.md#data-validation).

The first point is that quality is not a tool you buy. [why cant data quality be fixed with tech](https://www.analytics8.com/blog/why-data-quality-cannot-be-fixed-with-technology/) is Sharon Rehana on why data quality matters, where bad data comes from, and how to fix those issues. Once the problem is owned, it has to be defined: [CloverDX - data quality](https://www.cloverdx.com/explore/data-quality) explains what data quality really means, how to measure it, and the common best practices and pitfalls.

Measuring means picking metrics. [CloverDX - 6 Data Quality Metrics You Can't Afford To Ignore](https://www.cloverdx.com/blog/6-data-quality-metrics-you-cant-ignore) names six core data quality metrics, how they differ from dimensions and KPIs, and how to track them for reliable AI outcomes. [FirstEigan - data quality metrics](https://firsteigen.com/blog/6-key-data-quality-metrics-you-should-be-tracking/) is Seth Rao's version of six key metrics, with examples and ways to automate them on Azure and Snowflake. [Dataladder - data quality metrics](https://dataladder.com/10-data-quality-metrics-you-should-measure/) widens the list to ten metrics you should measure. [Datacademia - data quality metrics](https://datacadamia.com/data/quality/metric) adds a warning about naming, that these metrics are often called dimensions although in a dimensional model they act as attributes of the data rules, and defines timeliness as the time between when information is expected and when it is readily available. At big-data scale quality becomes veracity, one of the V's: [Open sistemas - big data challenges - veracity](https://opensistemas.com/en/the-four-vs-of-big-data/) is the OpenSistemas overview of the Four V's of Big Data.

## DBT

Once the metrics are named, they need to run as tests on every build, and dbt is where that happens in the warehouse.

The same notes are in [Snowflake](lakes-and-warehouses.md#snowflake) and [Tools](tools.md).

The starting point is the [DBT builtin tests](https://docs.getdbt.com/docs/building-a-dbt-project/tests), the dbt docs on configuring data tests to assess the quality of input data and the accuracy of the resulting datasets. [DBT expectations](https://hub.getdbt.com/calogica/dbt_expectations/0.1.2/) is the dbt package-hub entry that extends those tests. When neither fits, [DBT custom generic tests](https://docs.getdbt.com/guides/legacy/writing-custom-generic-tests) shows how to define your own custom generic data tests. Those tests are written in Jinja, and [Jinja](https://www.youtube.com/watch?v=OraYXEr0Irg) is Real Python's introduction to Jinja templating.


## Data Validation

dbt tests the warehouse; validation is the wider habit of guarding every place bad data can enter, and these notes come from the question bank.

The same notes are in [Data Quality](data-quality.md), [Data Testing](../../evals/data-and-model-tests.md#data-testing), and [Tools](tools.md).

The question is: how can you protect yourself from bad data? Data validation, TDDA, monitoring. Each answer has tools. Type validation: [typeguard](https://github.com/agronholm/typeguard) is a run-time type checker for Python. Data validation: [pydantic](https://pydantic-docs.helpmanual.io/usage/dataclasses/) validates Python dataclasses. Test driven: [tdda](https://github.com/tdda/tdda) is the test-driven data analysis functions library. Data quality: [great expectations](https://greatexpectations.io/) is GX Core, an open source framework for testing, validating, and documenting data quality across pipelines, workflows, and teams. Saas: [SuperConductive by GE](https://superconductive.ai/) is the hosted side of the same Great Expectations platform.
