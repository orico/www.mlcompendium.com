# Data Lineage

When a number is wrong, someone has to walk the path the data took. The page defines data lineage and why it matters, then points at a few good articles, and ends with the vendors that build lineage tools.

Lineage is where governance and quality meet: you cannot own or trust a number you cannot trace. The same notes are in [Data Governance](data-governance.md) and [Data Quality](data-quality.md).

Data lineage refers to the detailed history of data as it moves through various stages and transformations in an information system. It is essentially the life cycle of data, from its origins to its endpoint, including how it is modified and processed over time. Understanding data lineage is crucial for several reasons:

- Traceability — it helps track where data comes from, which is vital for debugging issues, understanding dependencies, and ensuring data quality.
- Compliance — many regulatory requirements, such as GDPR and HIPAA, require knowing the flow of data to ensure it is handled securely and within legal parameters.
- Data Governance — it aids in managing data, understanding its utility, and ensuring that data usage is consistent with organizational policies.
- Impact Analysis — it allows organizations to assess the potential impact of changes in the data environment. This is crucial for risk management and strategic planning.
- Audit and Reporting — data lineage provides transparency for audits, ensuring that all data used in financial reporting, for instance, is accurate and verifiable.

Those reasons all depend on the trail actually existing. Tools and systems that manage data lineage collect metadata from various parts of data handling systems, providing a visual or documented trail of how data flows through software and systems, which transformations it undergoes, and how it is used in different analyses and decisions. This capability is particularly important in complex systems where data is handled across various platforms and services.

## Good articles

The definition above is the short version; these three articles are the longer reads. The first two answer what data lineage is, from Octopai and Ardoq, and the third is Select Star's complete guide to its benefits, techniques, and best practices.

{% cards %}
{% card title="What is Data Lineage?" href="https://www.octopai.com/what-is-data-lineage/" %}
{% endcard %}

{% card title="Data Lineage" href="https://www.ardoq.com/knowledge-hub/data-lineage" %}
{% endcard %}

{% card title="The Complete Guide to Data Lineage: Benefits, Techniques, and Best Practices" href="https://www.selectstar.com/resources/the-complete-guide-to-data-lineage-benefits-techniques-and-best-practices" %}
{% endcard %}
{% endcards %}

## Data lineage vendors

Once the trail is understood, someone has to collect it, and that is what the lineage vendors sell. Many of them are also data catalogs.

The same notes are in [Data Catalogs](../datasets/data-catalogs.md).

The cards below are Octopai, Collibra, Azure Purview, Cloudera, Alation, and the open-source Apache Atlas. Alation now presents its product as the Alation Intelligence Operating System, aimed at getting enterprise AI right and keeping it right.

{% cards %}
{% card title="Octopai" href="https://octopai.com/" %}
{% endcard %}

{% card title="Collibra" href="https://www.collibra.com/" %}
{% endcard %}

{% card title="Azure Purview" href="https://learn.microsoft.com/en-us/purview/purview" %}
{% endcard %}

{% card title="Cloudera" href="https://www.cloudera.com/" %}
{% endcard %}

{% card title="Alation" href="https://www.alation.com/" %}
{% endcard %}

{% card title="Apache Atlas" href="https://atlas.apache.org/" %}
{% endcard %}
{% endcards %}
