# Database Modeling

This page covers database types, warehouse modeling, data fabric, virtualization, and deployment tools.

- [Types of DBs](https://simonatta.medium.com/database-types-2dac81461709)

## Data Warehouse

This section points at a guide from strategy to implementation for data warehouses.

- (good) [a guide from strategy to implementation](https://www.analytics8.com/blog/what-is-a-data-warehouse/)

## Data Vault Modeling

This section defines data vault modeling and how hubs, links, and satellites fit a lakehouse.

- [**Data vault modeling**](https://en.wikipedia.org/wiki/Data_vault_modeling) is a [database](https://en.wikipedia.org/wiki/Database) modeling method that is designed to provide long-term historical storage of [data](https://en.wikipedia.org/wiki/Data) coming in from multiple operational systems. It is also a method of looking at historical data that deals with issues such as auditing, tracing of data, loading speed and resilience to change as well as emphasizing the need to trace where all the data in the database came from. This means that every [row](https://en.wikipedia.org/wiki/Row_(database)) in a data vault must be accompanied by record source and load date attributes, enabling an auditor to trace values back to the source. It was developed by [Daniel (Dan) Linstedt](https://en.wikipedia.org/w/index.php?title=Daniel_Linstedt&action=edit&redlink=1) in 2000. — Wikipedia
- It is a design pattern to build a DWH for enterprise analytics. It has hubs (core business concepts), links (relationships between hubs), and satellites that store info about these two. Good for the lakehouse paradigm. [link has a good image.](https://www.databricks.com/glossary/data-vault) — Databricks

## Data Fabric

This section defines data fabric and points at NetApp, IBM, Gartner, and mesh-vs-fabric comparisons.

1. [Data Fabric](https://preetihemant.medium.com/modern-data-architecture-models-69e90b725a05) is a data architecture that follows a set of steps that determine its flow. The first step takes data through an integration phase. In the integration phase, data is ingested and then cleaned, transformed and loaded into storage. Then, there is the data quality phase where quality assessment is performed on the stored data. This data is then made available for different use cases through a combination of a data lake and a data warehouse. Typical use cases are BI, analytics and machine learning. Data governance policies are defined for the ingested data and a data catalog is used for discoverability — by Preeti Hemant.
2. (short) [Netapp](https://www.netapp.com/data-fabric/what-is-data-fabric/)
3. (good) [IBM](https://www.ibm.com/topics/data-fabric) — Data Management layer, Data Ingestion Layer, Data Processing, Data Orchestration, Data Discovery, Data Access.
4. (good) [Gartner](https://www.gartner.com/smarterwithgartner/data-fabric-architecture-is-key-to-modernizing-data-management-and-integration) — the pillars of data fabric
5. [talend](https://www.talend.com/resources/what-is-data-fabric/)
6. (good but lengthy) [spiceworks](https://www.spiceworks.com/tech/big-data/articles/what-is-data-fabric/) — architectural components, best practices
7. (good) [k2view](https://www.k2view.com/what-is-data-fabric) — has a great figure of integration, storage, catalog, cleansing and masking, transformation and enrichment, governance, web services.
8. (good) [tibco](https://www.tibco.com/reference-center/what-is-data-fabric) — application and services, dev and integration, security, storage management, transport, endpoints.
9. [mesh vs fabric](https://www.datanami.com/2021/10/25/data-mesh-vs-data-fabric-understanding-the-differences/)

## Data virtualization

This section points at a single view of data without moving it.

- [data virtualization](https://www.ibm.com/analytics/data-virtualization) — single view without movement of data.

## Tools for deploying data models

This section points at tools for deploying data models in production.

- [tools for deploying data models in prod](https://www.superdatascience.com/podcast/tools-for-deploying-data-models-into-production)
