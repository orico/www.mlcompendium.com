# MLOps Patterns

Production ML repeats the same shapes. The page gathers the pattern guides first, the catalogs, roadmaps, courses, and Metaflow reading, and then shows two end-to-end systems that use them.

## General patterns

A pattern only helps once you can see where it sits in a product's life, so the catalogs start with the lifecycle itself. [ML product lifecycle patterns](https://medium.com/data-science/understanding-ml-product-lifecycle-patterns-a39c18302452) is Antony Mayi's guide to understanding ML-product lifecycle patterns, and the figure below is its picture of those lifecycles.

<figure><img src="../../ops/.gitbook/assets/image (1) (1).png" alt=""><figcaption><p>ML product lifecycle patterns</p></figcaption></figure>

Inside that lifecycle, the design choices have names. The [ML design patterns book](https://github.com/GoogleCloudPlatform/ml-design-patterns) repo is the source code accompanying the O'Reilly book Machine Learning Design Patterns, and the two figures after it are pages from that book.

<figure><img src="../../ops/.gitbook/assets/image (2).png" alt=""><figcaption><p>ML design patterns book</p></figcaption></figure>

<figure><img src="../../ops/.gitbook/assets/image (1).png" alt=""><figcaption><p>ML design patterns book</p></figcaption></figure>

The book's patterns are about the model; the next catalog is about the system around it. [MLOps design patterns](https://github.com/mercari/ml-system-design-pattern/tree/master) is Mercari's set of system design patterns for machine learning, shown in the figure below.

<figure><img src="../../ops/.gitbook/assets/image.png" alt=""><figcaption><p>MLOps design patterns</p></figcaption></figure>

When a single catalog is not enough, [Awesome MLOps](https://github.com/visenger/awesome-mlops) is the curated list of references for MLOps. The Visenger figure, copied from the original hosted image, and the table of contents after it show how that list is organized.

<figure><img src="../../ops/.gitbook/assets/gimg-2a37b4dcacbf.png" alt=""><figcaption><p>Visenger</p><p>Credit: <a href="https://lh4.googleusercontent.com/6Dd5yQHT_iJxIGqiCSmHLs5m4nVb4by_ovEoBjrJTFcUoEvh7nmiNWpb84TJQcd5IWuSy5vElL6nFsXv5NkOKzo0Juc1ZVzX1jr3BWVgIrfhTIfGggSysNOZABG5-6h4vB8_kQ9q">copied from the original hosted image.</a></p></figcaption></figure>

<figure><img src="../../ops/.gitbook/assets/image (2) (1).png" alt=""><figcaption><p>Table of contents</p></figcaption></figure>

Lists describe the field as it is; roadmaps describe where it is going. The [MLOps Roadmap](https://github.com/cdfoundation/sig-mlops/blob/main/roadmap/2022/MLOpsRoadmap2022.md) is the 2022 roadmap of the CD Foundation's SIG MLOps. [Google's Practitioners Guide to MLOps: a framework for continuous delivery and automation of machine learning](https://cloud.google.com/resources/mlops-whitepaper) is written for ML leaders and enterprise architects who want to understand MLOps in theory and in practice. The [State of MLOps](https://ml-ops.org/content/state-of-mlops) page on ml-ops.org comes with the template shown below.

<figure><img src="../../ops/.gitbook/assets/image (3).png" alt=""><figcaption><p>Template</p></figcaption></figure>

The roadmaps become concrete in two hands-on pieces. [Easy MLOps with PyCaret and MLflow](https://medium.com/data-science/easy-mlops-with-pycaret-mlflow-7fbcbf1e38c6) is by Moez Ali. [Challenges and solutions by Iguazio](https://medium.com/data-science/ml-ops-challenges-solutions-and-future-trends-d2e59b74dc6b) is by Yaron Haviv, and the Iguazio figure below is credited to that article. The general-patterns figure after it is copied from the original hosted image.

<figure><img src="../../ops/.gitbook/assets/gimg-cad16f4b25b6.png" alt=""><figcaption><p>Iguazio</p><p>Credit: <a href="https://medium.com/data-science/ml-ops-challenges-solutions-and-future-trends-d2e59b74dc6b">Iguazio</a></p></figcaption></figure>

<figure><img src="../../ops/.gitbook/assets/gimg-32acb26e3d04.png" alt=""><figcaption><p>General patterns</p><p>Credit: <a href="https://lh3.googleusercontent.com/TqEy5NDYAnnuyM0o1j8XkKgl2KynL1Pfy6ZHG1LU7d0Ev6RtVXbCEcMFcakbPMlvYKJ41jmLDGIVazNyWA-wYEf1xKCbTzOFbJttpAp6nIWOJAvEdn1yP14NZBqXmP8b-LI80Y57">copied from the original hosted image.</a></p></figcaption></figure>

For a full course instead of articles, [Stanford CS329](https://stanford-cs329s.github.io/syllabus.html) is CS 329S: Machine Learning Systems Design. The course goes into how ML systems are built, and how to debug them, find a root cause, and monitor them.

One framework keeps coming up in these patterns, so the Metaflow reading on Medium follows, from a high-level review and the schema to docs on loading and storing data:

{% cards %}
{% card title="1" href="https://medium.com/bigdatarepublic/a-review-of-netflixs-metaflow-65c6956e168d" %}
A high-level review.
{% endcard %}

{% card title="2" href="https://medium.com/acing-ai/decoding-netflix-metaflow-2ad84b36199e" %}
The schema.
{% endcard %}

{% card title="3" href="https://medium.com/analytics-vidhya/metaflow-by-netflix-the-good-the-bad-and-the-ugly-b7fc6a833484" %}
{% endcard %}

{% card title="4" href="https://medium.com/data-science/learn-metaflow-in-10-mins-netflixs-python-r-framework-for-data-scientists-2ef124c716e4" %}
Amazing. (EJ) Vivek Pandey.
{% endcard %}

{% card title="5" href="https://medium.com/data-science/be-more-efficient-to-produce-machine-learning-pipeline-with-metaflow-db5f943ebbe7" %}
Extra. Jean-Michel D.
{% endcard %}

{% card title="6" href="https://docs.metaflow.org/metaflow/data" %}
Loading and storing data, in the docs.
{% endcard %}
{% endcards %}

A pipeline also has to be tuned and watched once it runs. HyperparameterHunter is the thread of [Hyperopt, MLflow, unit tests, concept drift, using Python and Kafka](https://medium.com/data-science/putting-ml-in-production-ii-logging-and-monitoring-algorithms-91f174044e4e), the logging-and-monitoring side of putting ML in production.

The same notes are in [Drift](mlops-monitoring-and-alerts.md#drift) and [ML Experiment Management](experiment-management.md).

## Patterns in practice

With the catalogs in hand, two systems show the patterns running end to end.

The same notes are in [MetaFlow](full-stack-and-ops.md#metaflow) and [Prefect](full-stack-and-ops.md#prefect).

The first is an MLOps end-to-end system, "[You dont need a bigger boat](https://github.com/jacopotagliabue/you-dont-need-a-bigger-boat)", using Metaflow, Snowflake, DBT, Prefect, Great Expectations, Weights & Biases, SageMaker, and Lambda. The second strips that down: [A simplistic end-to-end system](https://github.com/jacopotagliabue/post-modern-stack) joins the modern data stack with the modern ML stack using Snowflake, DBT, S3, CometML, Reclist, and SageMaker.

Both systems follow the lifecycle this page opened with, and Antony Mayi's guide is also at its original address: A Guide to Classifying Operational Lifecycles of ML-Driven Products with an Overview of their Notable Patterns, ML product lifecycle patterns. [https://towardsdatascience.com/understanding-ml-product-lifecycle-patterns-a39c18302452](https://towardsdatascience.com/understanding-ml-product-lifecycle-patterns-a39c18302452)

## Deprecated links

{% hint style="warning" %}
These links and images no longer work. The original wording is kept here. A same-resource copy, when one was checked, is used above.
{% endhint %}

- Easy mlops with pycaret and mlflow. This address no longer opens: https://towardsdatascience.com/easy-mlops-with-pycaret-mlflow-7fbcbf1e38c6
- Challenges and solutions by iguazio. This address no longer opens: https://towardsdatascience.com/ml-ops-challenges-solutions-and-future-trends-d2e59b74dc6b
- 4 (amazing). This address no longer opens: https://towardsdatascience.com/learn-metaflow-in-10-mins-netflixs-python-r-framework-for-data-scientists-2ef124c716e4
- 5 (extra). This address no longer opens: https://towardsdatascience.com/be-more-efficient-to-produce-machine-learning-pipeline-with-metaflow-db5f943ebbe7
- HyperparameterHunter, Hyperopt, mlflow, unit test, concept drifts, using python and kafka. This address no longer opens: https://towardsdatascience.com/putting-ml-in-production-ii-logging-and-monitoring-algorithms-91f174044e4e
