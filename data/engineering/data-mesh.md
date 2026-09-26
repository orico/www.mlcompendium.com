# Data Mesh

A central data team can become the bottleneck, so mesh treats domains as owners of data products. The page goes from what a data mesh is, through Zhamak Dehghani's own introductions and principles, to comparisons with lakes and fabric, topologies, and the practice notes of vendors and teams who built one.

Mesh is an architecture choice and a sibling of fabric, so it is worth reading with both. The same notes are in [Data Architecture](data-architecture.md) and [Data Fabric](database-architecture-and-modeling.md#data-fabric).

The first question is what a [data mesh](https://databricks.com/session_na20/data-mesh-in-practice-how-europes-leading-online-platform-for-fashion-goes-beyond-the-data-lake) is, and in practice: the Databricks session on how Europe's leading online fashion platform went beyond the data lake.

The idea is Zhamak Dehghani's, so her own material comes next. [Introduction to Data Mesh](https://www.youtube.com/watch?v=_bmYXWCxF_Q) is her Thoughtworks introduction. [how to move from data lake to distributed data mesh](https://martinfowler.com/articles/data-monolith-to-mesh.html) is the article that names the problems with the centralized data lake and says a data mesh needs domains, self-service platforms, and product thinking. [principles and logical architecture](https://martinfowler.com/articles/data-mesh-principles.html) is the follow-up with the four principles that drive a mesh's logical architecture. The (good) [Keynote - Data Mesh by Zhamak Dehghani](https://www.youtube.com/watch?v=L_-fHo0ZkAo) makes the core claim that the pipeline of oltp→etl→olap is broken. [lessons from the trenches](https://www.youtube.com/watch?v=Nw_bxIyR1L0) is Zhamak Dehghani and Sina Jahan on what they learned implementing it.

With the principles in hand, the next step is placing mesh next to its neighbors. [lake vs mesh, he probably means fabric vs mesh](https://medium.com/codex/data-lakehouse-vs-data-mesh-bfa1132f94b) is the data lakehouse vs data mesh comparison, read with that correction in mind. [data mesh 101](https://www.youtube.com/watch?v=hgKOpEQaqdY&list=PLa7VYi0yPIH0L8ahQYbyBFkGc6a949-Lj&index=10) — by Confluent is the hands-on episode on creating a data product. The (good) [mesh topologies](https://medium.com/data-science/data-mesh-topologies-and-domain-granularity-65290a4ebb90) — by Piethein Strengholt covers topologies and domain granularity, drawn from his customer conversations and architecture sessions at Microsoft where mesh kept coming up. The figure below is from it.

<figure><img src="../../ops/.gitbook/assets/image (37).png" alt=""><figcaption><p>mesh topologies</p></figcaption></figure>

The remaining sources are practice. [Data Mesh & MLOps using a data platform.](https://medium.com/swlh/building-a-data-platform-to-enable-analytics-and-ai-driven-innovation-1bd95e37efb9) ties the mesh to MLOps through a data platform. [Explained](https://medium.com/@david.c.dupuis/data-mesh-explained-a95b6ae50878) is an in-depth explainer written by someone learning data mesh from scratch and sharing it. [The next gen of architecture](https://datagrad.medium.com/data-mesh-transition-to-next-generation-of-data-architecture-832c4bc27e9f) argues that architecture defines the efficiency, scalability, and usability of data, and presents mesh as the transition to the next generation. [how not to mesh it - monte carlo](https://www.montecarlodata.com/blog-what-is-a-data-mesh-and-how-not-to-mesh-it-up/) is Monte Carlo's beginner's guide to implementing a mesh without messing it up. [building data mesh using lake house approach](https://www.youtube.com/watch?v=YPYODx4Pfdc) by AWS is the tech talk on building a mesh architecture with AWS Lake Formation. The [enterprise data mesh by oracle](https://www.oracle.com/a/ocom/docs/datamesh-ebook.pdf) is Oracle's ebook on the enterprise version. [deconstructing data mesh principles](https://medium.com/slalom-data-ai/data-mesh-232e50f42e66) closes by taking the principles apart again.

## Deprecated links

{% hint style="warning" %}
These links and images no longer work. The original wording is kept here. A same-resource copy, when one was checked, is used above.
{% endhint %}

- (good) mesh topologies - by Piethein Strengholt. This address no longer opens: https://towardsdatascience.com/data-mesh-topologies-and-domain-granularity-65290a4ebb90
