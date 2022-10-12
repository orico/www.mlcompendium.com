# Database Modeling

1. Data Vault Modeling
   1. [**Data vault modeling**](https://en.wikipedia.org/wiki/Data\_vault\_modeling) is a [database](https://en.wikipedia.org/wiki/Database) modeling method that is designed to provide long-term historical storage of [data](https://en.wikipedia.org/wiki/Data) coming in from multiple operational systems. It is also a method of looking at historical data that deals with issues such as auditing, tracing of data, loading speed and resilience to change as well as emphasizing the need to trace where all the data in the database came from. This means that every [row](https://en.wikipedia.org/wiki/Row\_\(database\)) in a data vault must be accompanied by record source and load date attributes, enabling an auditor to trace values back to the source. It was developed by [Daniel (Dan) Linstedt](https://en.wikipedia.org/w/index.php?title=Daniel\_Linstedt\&action=edit\&redlink=1) in 2000. - wikipedia
   2. its a design pattern to build dwh for enterprise analytics. it has hubs (core business concepts) links (relationshipts between hubs) satellites store info about these two. good for lakehouse paradigm. [link has a good image.](https://www.databricks.com/glossary/data-vault) - databricks