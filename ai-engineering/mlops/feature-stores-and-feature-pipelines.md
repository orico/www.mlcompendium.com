# Feature Stores & Feature Pipelines

Training and serving need the same features. The page starts with the pipeline that builds them, then moves to the store that serves them: what a store is, how it differs from a warehouse, the products, and why a store alone is not enough.

The same notes are in [FEATURE STORE](#feature-store) and [Monitoring & alerts](mlops-monitoring-and-alerts.md#monitoring--alerts).

Feature calculation and serving are the heart of every classical machine learning system, and that is where [Feature store and feature pipelines](https://medium.com/data-for-ai/feature-pipelines-and-feature-stores-deep-dive-into-system-engineering-and-analytical-tradeoffs-3c208af5e05f), Assaf Pinhasi, begins: a deep dive into the system-engineering and analytical tradeoffs of both.
### **FEATURE STORE**

With the pipeline named, the next question is where its output lives, so this section is the store products and the articles that compare them.

The same notes are in [Feature Stores & Feature Pipelines](feature-stores-and-feature-pipelines.md).

The case for a store starts with **The importance of having one -** medium, an article that is now kept at the end of the page. Feast's [**What is?**](https://feast.dev/blog/what-is-a-feature-store/) is the comprehensive guide to feature stores, their role in machine learning operations, and the key challenges they solve in the ML lifecycle. The obvious objection is that a warehouse already stores data, and KDnuggets' [**Feature store vs data warehouse**](https://www.kdnuggets.com/2020/12/feature-store-vs-data-warehouse.html) answers it: a feature store is a data warehouse of features, but dual-database, one side serving features at low latency to online applications and the other storing large volumes of features.

Once the idea is clear, the open-source reference is [**Feast**](https://docs.feast.dev/), the introduction to Feast, the open source feature store. The same Feast guide doubles as [**what is 1**](https://feast.dev/blog/what-is-a-feature-store/), and [**what is 2**](https://neptune.ai/blog/feature-stores-components-of-a-data-science-factory-guide) is the second explainer on feature stores as components of a data science factory. The figure below, copied from the original hosted image, shows the features a store holds.

<figure><img src="../../.gitbook/assets/gimg-dd2ce0c86341.png" alt=""><figcaption><p>Features.</p><p>Credit: <a href="https://lh3.googleusercontent.com/-Syb5MJEHTEHc12GTKNQN8bWnt83zFs_isY_CFMISCYQJPLnvt-XdV3B_ycaRziMns-z0crVA01PpZUHI3Hgw251xhnIh_LB88cQKMb9_MNUzmN68cxvBZ6lsEw8FGxzMDX-_9xo">copied from the original hosted image</a>.</p></figcaption></figure>

If you would rather not run the store yourself, there are managed products. [**Tecton.ai**](https://www.tecton.ai/) managed feature store is the first; the address now lands on Databricks, which pitches a unified platform for data, analytics, and AI. Its figure follows.

<figure><img src="../../.gitbook/assets/gimg-2ef7cc3daf21.png" alt=""><figcaption><p>Features.</p><p>Credit: <a href="https://lh3.googleusercontent.com/3NQTUG2PVOIJZbNBYNj-BxZv5A4POEf1KJ20f4nhet_gaxj4cAJXjXwld9ZG-RoEWnRe-DWfS_qe1PrSojfXcTtlJZYy4w6_njyBi9qgsnDr7jnnfqMDMG-8Ea31qWGn5toG4HwI">copied from the original hosted image</a>.</p></figcaption></figure>

The [**Iguazio feature store**](https://www.iguazio.com/feature-store/) is the second ML feature store product, with its own figure below.

<figure><img src="../../.gitbook/assets/gimg-3d5cd4bcc8e9.png" alt=""><figcaption><p>Features.</p><p>Credit: <a href="https://lh3.googleusercontent.com/RWd1x9OSefMVbp_X6JYdfIy_Kz9hM_x7Wtg0mvm3mWUt_hvvi6gWATMcMDDUJrv1jGYhXUlVtBnlI4oCPm0nXkDWxMrzUpD1gLUefWv0fczK3XGRCQqqN6iDQ5yc2-4MbT7U304r">copied from the original hosted image</a>.</p></figcaption></figure>

A store, managed or not, still does not make the whole infrastructure, which is the point of **Why feature store is not enough**. Almog Baku's "Effective AI Infrastructure or Why Feature Store Is Not Enough" argues that modern AI infrastructure could accelerate the ML lifecycle and ease the interaction between data scientists and engineers. Why feature store is not enough. [https://towardsdatascience.com/effective-ai-infrastructure-or-why-feature-store-is-not-enough-43bc2d803401](https://towardsdatascience.com/effective-ai-infrastructure-or-why-feature-store-is-not-enough-43bc2d803401)

## Deprecated links

{% hint style="warning" %}
These links and images no longer work. The original wording is kept here. A same-resource copy, when one was checked, is used above.
{% endhint %}
- **The importance of having one -** medium. This address no longer opens: https://towardsdatascience.com/the-importance-of-having-a-feature-store-e2a9cfa5619f
