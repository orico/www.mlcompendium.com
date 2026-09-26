# Data Lineage

This page defines data lineage, why it matters, and points at articles and vendors.

Data lineage refers to the detailed history of data as it moves through various stages and transformations in an information system. It is essentially the life cycle of data, from its origins to its endpoint, including how it is modified and processed over time. Understanding data lineage is crucial for several reasons:

- Traceability — it helps track where data comes from, which is vital for debugging issues, understanding dependencies, and ensuring data quality.
- Compliance — many regulatory requirements, such as GDPR and HIPAA, require knowing the flow of data to ensure it is handled securely and within legal parameters.
- Data Governance — it aids in managing data, understanding its utility, and ensuring that data usage is consistent with organizational policies.
- Impact Analysis — it allows organizations to assess the potential impact of changes in the data environment. This is crucial for risk management and strategic planning.
- Audit and Reporting — data lineage provides transparency for audits, ensuring that all data used in financial reporting, for instance, is accurate and verifiable.

Tools and systems that manage data lineage collect metadata from various parts of data handling systems, providing a visual or documented trail of how data flows through software and systems, which transformations it undergoes, and how it is used in different analyses and decisions. This capability is particularly important in complex systems where data is handled across various platforms and services.

## Good articles

This section lists guides that explain data lineage in more depth.

{% cards %}
{% card title="What is Data Lineage?" href="https://www.octopai.com/what-is-data-lineage/" %}
{% endcard %}

{% card title="Data Lineage" href="https://www.ardoq.com/knowledge-hub/data-lineage" %}
{% endcard %}

{% card title="The Complete Guide to Data Lineage: Benefits, Techniques, and Best Practices" href="https://www.selectstar.com/resources/the-complete-guide-to-data-lineage-benefits-techniques-and-best-practices" %}
{% endcard %}
{% endcards %}

## Data lineage vendors

This section lists vendors and open tools that manage data lineage.

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
