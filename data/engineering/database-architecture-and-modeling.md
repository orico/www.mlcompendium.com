# Database Modeling

Storage choices and models decide how a warehouse can be queried later.
This page covers warehouse modeling, Data Vault, data fabric, virtualization, then tools for deploying models.

The same notes are in [NoSQL vs relational](lakes-and-warehouses.md#nosql-vs-relational).

- [Types of DBs](https://simonatta.medium.com/database-types-2dac81461709)

## Data Warehouse

This section opens with data warehouse modeling before vault and fabric.

The same notes are in [Data Patterns](patterns.md).

- (good) [a guide from strategy to implementation](https://www.analytics8.com/blog/what-is-a-data-warehouse/)

## Data Vault Modeling

After the warehouse notes, this section is Data Vault modeling.

- [**Data vault modeling**](https://en.wikipedia.org/wiki/Data_vault_modeling) is a [database](https://en.wikipedia.org/wiki/Database) modeling method that is designed to provide long-term historical storage of [data](https://en.wikipedia.org/wiki/Data) coming in from multiple operational systems. It is also a method of looking at historical data that deals with issues such as auditing, tracing of data, loading speed and resilience to change as well as emphasizing the need to trace where all the data in the database came from. This means that every [row](https://en.wikipedia.org/wiki/Row_(database)) in a data vault must be accompanied by record source and load date attributes, enabling an auditor to trace values back to the source. It was developed by [Daniel (Dan) Linstedt](https://en.wikipedia.org/w/index.php?title=Daniel_Linstedt&action=edit&redlink=1) in 2000. — Wikipedia
- It is a design pattern to build a DWH for enterprise analytics. It has hubs (core business concepts), links (relationships between hubs), and satellites that store info about these two. Good for the lakehouse paradigm. [link has a good image.](https://www.databricks.com/glossary/data-vault) — Databricks

## Data Fabric

Beside the vault, this section is data fabric.

The same notes are in [Data Catalogs](../datasets/data-catalogs.md), [Data Governance](data-governance.md), and [Data Mesh](data-mesh.md).

1. [Data Fabric](https://preetihemant.medium.com/modern-data-architecture-models-69e90b725a05) is a data architecture that follows a set of steps that determine its flow. The first step takes data through an integration phase. In the integration phase, data is ingested and then cleaned, transformed and loaded into storage. Then, there is the data quality phase where quality assessment is performed on the stored data. This data is then made available for different use cases through a combination of a data lake and a data warehouse. Typical use cases are BI, analytics and machine learning. Data governance policies are defined for the ingested data and a data catalog is used for discoverability — by Preeti Hemant.
- (short) [Netapp](https://www.netapp.com/data-fabric/what-is-data-fabric/)
3. (good) [IBM](https://www.ibm.com/topics/data-fabric) — Data Management layer, Data Ingestion Layer, Data Processing, Data Orchestration, Data Discovery, Data Access.
- (good) [Gartner](https://www.gartner.com/smarterwithgartner/data-fabric-architecture-is-key-to-modernizing-data-management-and-integration) good the pillars data fabric
- Data fabric refers to a machine-enabled data integration architecture utilizing metadata assets to unify, integrate, and govern disparate data environments. [talend](https://www.talend.com/resources/what-is-data-fabric/)
- (good but lengthy) [spiceworks](https://www.spiceworks.com/tech/big-data/articles/what-is-data-fabric/) good but lengthy architectural components best practices
7. (good) [k2view](https://www.k2view.com/what-is-data-fabric) — has a great figure of integration, storage, catalog, cleansing and masking, transformation and enrichment, governance, web services.
8. (good) [tibco](https://www.tibco.com/reference-center/what-is-data-fabric) — application and services, dev and integration, security, storage management, transport, endpoints.
9. [mesh vs fabric](https://www.datanami.com/2021/10/25/data-mesh-vs-data-fabric-understanding-the-differences/)

## Data virtualization

After fabric, this section is data virtualization.

- Discover enterprise solutions created by IBM to address your specific business challenges and needs. [data virtualization](https://www.ibm.com/analytics/data-virtualization)

## Tools for deploying data models

Closing the page, this section lists tools for deploying data models.

- SDS 619: Tools for Deploying Data Models into Production - SuperDataScience | Machine Learning | AI | Data Science Career | Analytics | Success. [tools for deploying data models in prod](https://www.superdatascience.com/podcast/tools-for-deploying-data-models-into-production)
