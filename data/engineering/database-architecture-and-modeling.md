# Database Modeling

Storage choices and models decide how a warehouse can be queried later.
This page starts from the kinds of databases and the warehouse, then moves to Data Vault, data fabric, and virtualization as ways to model and connect data, and ends with tools for deploying models.

The same notes are in [NoSQL vs relational](lakes-and-warehouses.md#nosql-vs-relational).

Before modeling anything, it helps to know which kind of database you are modeling for. [Types of DBs](https://simonatta.medium.com/database-types-2dac81461709) is part of an article series on data and information that walks through database types.

## Data Warehouse

The warehouse is the first target most models are built for, so the page starts there. The same notes are in [Data Patterns](patterns.md).

The (good) [a guide from strategy to implementation](https://www.analytics8.com/blog/what-is-a-data-warehouse/) is Sean Costello on what a data warehouse is, what your business may be missing without one, and three questions to ask when deciding on one.

## Data Vault Modeling

A warehouse still needs a modeling method, and Data Vault is the one built for history and auditing. Wikipedia defines it this way:

[**Data vault modeling**](https://en.wikipedia.org/wiki/Data_vault_modeling) is a [database](https://en.wikipedia.org/wiki/Database) modeling method that is designed to provide long-term historical storage of [data](https://en.wikipedia.org/wiki/Data) coming in from multiple operational systems. It is also a method of looking at historical data that deals with issues such as auditing, tracing of data, loading speed and resilience to change as well as emphasizing the need to trace where all the data in the database came from. This means that every [row](https://en.wikipedia.org/wiki/Row_(database)) in a data vault must be accompanied by record source and load date attributes, enabling an auditor to trace values back to the source. It was developed by [Daniel (Dan) Linstedt](https://en.wikipedia.org/w/index.php?title=Daniel_Linstedt&action=edit&redlink=1) in 2000. — Wikipedia

Databricks gives the working shape. It is a design pattern to build a DWH for enterprise analytics. It has hubs (core business concepts), links (relationships between hubs), and satellites that store info about these two. Good for the lakehouse paradigm. The [link has a good image.](https://www.databricks.com/glossary/data-vault) — Databricks

## Data Fabric

A vault models one warehouse; a data fabric is the architecture that ties many stores together. The same notes are in [Data Catalogs](../datasets/data-catalogs.md), [Data Governance](data-governance.md), and [Data Mesh](data-mesh.md).

The clearest walk through the flow is Preeti Hemant's. [Data Fabric](https://preetihemant.medium.com/modern-data-architecture-models-69e90b725a05) is a data architecture that follows a set of steps that determine its flow. The first step takes data through an integration phase. In the integration phase, data is ingested and then cleaned, transformed and loaded into storage. Then, there is the data quality phase where quality assessment is performed on the stored data. This data is then made available for different use cases through a combination of a data lake and a data warehouse. Typical use cases are BI, analytics and machine learning. Data governance policies are defined for the ingested data and a data catalog is used for discoverability — by Preeti Hemant.

The vendors each draw the same idea as layers. The (short) [Netapp](https://www.netapp.com/data-fabric/what-is-data-fabric/) link now opens NetApp's unified storage page, about storage that flexes as your data needs scale and evolve. (good) [IBM](https://www.ibm.com/topics/data-fabric) — Data Management layer, Data Ingestion Layer, Data Processing, Data Orchestration, Data Discovery, Data Access. (good) [Gartner](https://www.gartner.com/smarterwithgartner/data-fabric-architecture-is-key-to-modernizing-data-management-and-integration) good the pillars data fabric, from Gartner's business insights for executives. [talend](https://www.talend.com/resources/what-is-data-fabric/) defines data fabric as a machine-enabled data integration architecture utilizing metadata assets to unify, integrate, and govern disparate data environments, and covers why you need it and best practices. (good but lengthy) [spiceworks](https://www.spiceworks.com/tech/big-data/articles/what-is-data-fabric/) good but lengthy architectural components best practices.

Two more break the fabric into components you can draw. (good) [k2view](https://www.k2view.com/what-is-data-fabric) — has a great figure of integration, storage, catalog, cleansing and masking, transformation and enrichment, governance, web services. (good) [tibco](https://www.tibco.com/reference-center/what-is-data-fabric) — application and services, dev and integration, security, storage management, transport, endpoints.

Fabric is often confused with mesh, and [mesh vs fabric](https://www.datanami.com/2021/10/25/data-mesh-vs-data-fabric-understanding-the-differences/) is Datanami on understanding the differences.

## Data virtualization

Fabric integrates data by moving and cataloging it; virtualization is the option of querying it where it already lives. The [data virtualization](https://www.ibm.com/analytics/data-virtualization) link now opens IBM's solutions page, which presents enterprise solutions created to address specific business challenges and needs.

## Tools for deploying data models

Closing the page, a model is only done when it runs in production. [tools for deploying data models in prod](https://www.superdatascience.com/podcast/tools-for-deploying-data-models-into-production) is SuperDataScience episode SDS 619, where Jon Krohn speaks with Erik Bernhardsson, who invented Spotify's original music recommendation system, about interviewing data science candidates, deploying a data model into the cloud, and how Spotify went from digital music startup to AI-driven streaming giant.
