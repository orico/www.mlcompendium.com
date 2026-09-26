# MLOps Tools

An MLOps stack is many tools, not one product. This page walks CI, packages, Docker, Kubeflow, workflow runners, serving, and the services beside them.

## Continuous integration

CI is the first gate before packages and containers enter the stack.

The same notes are in [Continuous Integration](../devops/full-stack-and-ops/continuous-integration.md).

- Travis CI is the most simple and flexible ci/cd tool available today, by Sybre Waaijer. [Travis](https://travis-ci.org/)
- Deliver production-ready software at AI speed. [Circle CI](https://circleci.com/)
- GitHub Actions
 - [poetry black pytest](https://medium.com/@vanflymen/blazing-fast-ci-with-github-actions-poetry-black-and-pytest-9e74299dd4a5)

## Package repositories

With CI named, package repositories are where the built artifacts live.

- PyPI — public
- Gemfury is a hosted repository for your public and private packages, where they are safe and within reach. [Gemfury](https://gemfury.com/)

## Docker for data science

Packages need a runtime. Docker is how the data-science environment travels.

The same notes are in [Docker](../devops/full-stack-and-ops/docker.md) and [Jupyter](../../data/engineering/data-science-tools.md#jupyter).

- [What are Docker layers](https://medium.com/@jessgreb01/digging-into-docker-layers-c22f948ed612)
- [Install on Ubuntu](https://linuxconfig.org/how-to-install-docker-on-ubuntu-18-04-bionic-beaver)
- Selecting an Image — Docker Stacks documentation. [Many Jupyter Docker images (Spark too)](https://jupyter-docker-stacks.readthedocs.io/en/latest/using/selecting.html)
- [How to run Jupyter Docker 1](https://medium.com/@rahulvaish/jupyter-docker-badd38fd6b51)
- [2](https://medium.com/fundbox-engineering/overview-d3759e83969c)
- [Tell Docker to run on a mounted disk](https://stackoverflow.com/questions/32070113/how-do-i-change-the-default-docker-container-location)
- [Docker, Keras, k8s, Flask serving](https://medium.com/analytics-vidhya/deploy-your-first-deep-learning-model-on-kubernetes-with-python-keras-flask-and-docker-575dc07d9e76)
- Learn how to use Docker Compose to define and run multi-container applications with this detailed introduction to the tool. [Compose](https://docs.docker.com/compose/)
- Docker Compose is simply a tool that allows you to describe a collection of multiple containers that can interact via their own network in a very straightforward way.
- [Docker on Ubuntu, tutorial](https://medium.com/fundbox-engineering/overview-d3759e83969c)
- [Docker for data science](https://aoyilmaz.medium.com/docker-in-data-science-and-a-friendly-beginner-to-docker-186fafdfbdeb)
- [Using VS Code to debug containers](https://nirradi.medium.com/vsc-vs-pycharm-developing-inside-docker-containers-4892c83d30e4)

## Kubeflow for data science

Once containers exist, Kubeflow is ML on Kubernetes with pipelines and serving options.

The same notes are in [Kubeflow](../devops/full-stack-and-ops/kubernetes.md#kubeflow).

- Machine Learning on Kubernetes with Kubeflow, by Google Cloud Tech. [YouTube — the easy way](https://www.youtube.com/watch?v=P5wcE4IwKgQ)
- [intro](https://medium.com/@amina.alsherif/how-to-get-started-with-kubeflow-187792f3e99)
- Introducing Kubeflow - A Composable, Portable, Scalable ML Stack Built for Kubernetes. [intro2](https://kubernetes.io/blog/2017/12/introducing-kubeflow-composable/)
- [intro3](https://medium.com/better-programming/kubeflow-pipelines-with-gpus-1af6a74ec2a)
- There are many ways to serve a trained model in and outside of Kubeflow. [Really good detailed article, for example it supports many serving options such as Seldon](https://ubuntu.com/blog/ml-serving-models-with-kubeflow-on-ubuntu-part-1)
- [presentation](https://www.oliverwyman.com/content/dam/oliver-wyman/v2/events/2018/March/Google_London_Event/Public%20Introduction%20to%20Kubeflow.pdf)
- Tutorials:
 - Example for end-to-end machine learning on Kubernetes using Kubeflow and Seldon Core - kubeflow/example-seldon. [Official example](https://github.com/kubeflow/example-seldon)
 - Kubeflow End to End - GitHub Issue Summarization. [Step by step tut](http://web.archive.org/web/20190330195417/https://codelabs.developers.google.com/codelabs/cloud-kubeflow-e2e-gis/index.html?index=..%2F..index)
 - [really detailed tut](https://medium.com/data-science/how-to-create-and-deploy-a-kubeflow-machine-learning-pipeline-part-1-efea7a4b650f)
 - End-to-end Reusable ML Pipeline with Seldon and Kubeflow — seldon-core documentation. [KF + Seldon on ec2](http://web.archive.org/web/20240417175403/https://docs.seldon.io/projects/seldon-core/en/latest/examples/kubeflow_seldon_e2e_pipeline.html)

## Airflow

Kubeflow is not the only orchestrator. Airflow schedules the DAGs around the models.

The same notes are in [Airflow](../../data/processing/data-pipelines.md#airflow) and [Data Pipelines](../../data/processing/data-pipelines.md).

- [Airflow](https://airflow.apache.org/) is a platform created by the community to programmatically author, schedule and monitor workflows.
- [Airflow in 5 minutes](https://medium.com/swlh/apache-airflow-in-5-minutes-c005b4b11b26) Ashish Kumar
- [Airflow 2.0 tutorial](https://medium.com/apache-airflow/apache-airflow-2-0-tutorial-41329bbf7211) Tomasz Urbaszek
- [Simple ETL](https://adenilsoncastro.medium.com/apache-airflow-the-etl-02-f4ac25f4d9b4) Adnilson Castro
- [Airflow Scheduler & Webserver](https://medium.com/analytics-vidhya/manage-your-workflows-with-apache-airflow-e7b0e45544a8) Shritam Kumar Mund
- [Airflow for DS](https://medium.com/data-science/apache-airflow-for-data-science-how-to-write-your-first-dag-in-10-minutes-9d6e884def72) Dario Rade

## Prefect

Airflow is one runner. Prefect is another workflow option for ML.

The same notes are in [ML Architecture](ml-architecture.md) and [Patterns in practice](mlops.md#patterns-in-practice).

- Orchestrating Machine Learning Workflows with Prefect // Kevin Kho // MLOps Meetup #94, by AAIF Live. [ML workflows with Prefect](https://www.youtube.com/watch?v=SP6WqCRUkNc)

## MetaFlow

Beside Prefect, Metaflow is Netflix's framework for the same pipeline problem.

The same notes are in [ML Architecture](ml-architecture.md) and [Patterns in practice](mlops.md#patterns-in-practice).

- Metaflow: The ML Infrastructure at Netflix, by AICamp. [Intro](https://www.youtube.com/watch?v=JCbOI_1ZA5E)
- Metaflow Framework : Hello World Program #AI #ML, by AI Scientist. and [what is](https://www.youtube.com/watch?v=bxVAniteuQs)

- A step-by-step guide on installing and configuring each of the kubeflow components on your local machine. really detailed tut. [https://towardsdatascience.com/how-to-create-and-deploy-a-kubeflow-machine-learning-pipeline-part-1-efea7a4b650f](https://towardsdatascience.com/how-to-create-and-deploy-a-kubeflow-machine-learning-pipeline-part-1-efea7a4b650f)
- Learn to work with Bash and Python operators in Apache Airflow by coding an entire DAG. Airflow for DS. [https://towardsdatascience.com/apache-airflow-for-data-science-how-to-write-your-first-dag-in-10-minutes-9d6e884def72](https://towardsdatascience.com/apache-airflow-for-data-science-how-to-write-your-first-dag-in-10-minutes-9d6e884def72)
- Medium on DL as a service by Nir Orman. [https://towardsdatascience.com/serving-deep-learning-algorithms-as-a-service-6aa610368fde](https://towardsdatascience.com/serving-deep-learning-algorithms-as-a-service-6aa610368fde)
- Towards Data Science. Scaling ML on the cloud. [https://towardsdatascience.com/scalable-efficient-big-data-analytics-machine-learning-pipeline-architecture-on-cloud-4d59efc092b5](https://towardsdatascience.com/scalable-efficient-big-data-analytics-machine-learning-pipeline-architecture-on-cloud-4d59efc092b5)

## Seldon

Workflows still need a serving layer. Seldon puts models on Kubernetes.

The same notes are in [MLOps Deployment](mlops-deployment.md).

- Runs in k8s
- Seldon-core vs seldon-deploy (what are the differences?)
- [Serving graph, recipe file](https://becominghuman.ai/seldon-inference-graph-pipelined-model-serving-211c6b095f62)
- [Descriptive intro](https://medium.com/seldon-open-source-machine-learning/introducing-seldon-core-machine-learning-deployment-for-kubernetes-e10e94c19fd8)
- [Sales pitch intro](https://medium.com/seldon-open-source-machine-learning/introducing-seldon-deploy-c390d11af20c)

## Serving models

With Seldon named, these notes are the broader serving patterns and Dapr.

The same notes are in [MLOps Deployment](mlops-deployment.md).

- Machine learning system design patterns Data Science Leads IL Meeting - June 2020 Slides by Shai Yanovski // Explorium.ai. ML system design patterns [res](https://docs.google.com/presentation/d/1pSkklHkBySMnJNODshW8NZVpBSqOsbJBWeEq8RrS0M4/edit#slide=id.g81f938aa2b_0_47)
- System design patterns for machine learning. [git](https://github.com/mercari/ml-system-design-pattern)
- Seldon
- [Medium on DL as a service by Nir Orman](https://medium.com/data-science/serving-deep-learning-algorithms-as-a-service-6aa610368fde)
- [Scaling ML on the cloud](https://medium.com/data-science/scalable-efficient-big-data-analytics-machine-learning-pipeline-architecture-on-cloud-4d59efc092b5)
- [Dapr](https://github.com/dapr/dapr) is a portable, serverless, event-driven runtime that makes it easy for developers to build resilient, stateless and stateful microservices that run on the cloud and edge and embraces the diversity of languages and developer frameworks.

 Dapr codifies the best practices for building microservice applications into open, independent, building blocks that enable you to build portable applications with the language and framework of your choice. Each building block is independent and you can use one, some, or all of them in your application.

## FastAPI

Serving often needs an HTTP API. FastAPI is the typed Flask alternative here.

- FastAPI framework, high performance, easy to learn, fast to code, ready for production. [Flask on steroids with variable parameters](https://fastapi.tiangolo.com/alternatives/)

## Kafka for data science

APIs are not the only pipe. Kafka is the streaming bus for data science.

The same notes are in [Kafka](../devops/full-stack-and-ops/key-value-db.md#kafka).

- Kafka in a Nutshell. Kafka in a Nutshell. [What is, terminology, use cases](https://sookocheff.com/post/kafka/kafka-in-a-nutshell/)

## Redis for data science

Beside Kafka, Redis is the in-memory store those services lean on.

The same notes are in [Key Value DB](../devops/full-stack-and-ops/key-value-db.md).

- What is, vs [memcached](https://medium.com/@pankaj.itdeveloper/memcached-vs-redis-which-one-to-choose-d5177482dc42)
- [Redis cluster](https://medium.com/@inthujan/introduction-to-redis-redis-cluster-6c7760c8ebbc)
- [Redis plus spaCy](https://medium.com/data-science/spacy-redis-magic-60f25c21303d)
4. Note: Redis is a managed dictionary; its strength lies when you have a lot of data that needs to be queried and managed and you don’t want to hard-code it, for example.
5. [Long tutorial](https://realpython.com/python-redis/)

## Sentry

Once traffic flows, Sentry watches the Python stack for errors.

- [For Python](https://sentry.io/for/python/), your code is telling you more than what your logs let on. Sentry’s full stack monitoring gives you full visibility into your code, so you can catch issues before they become downtime.

## Statsd

Sentry is errors. StatsD is the gauges and buckets for metrics.

- Daemon for easy but powerful stats aggregation. [Statistics server, with gauges/buckets and flushing/sending ability](https://github.com/statsd/statsd/blob/master/examples/python_example.py)

## Visualization

Metrics need a picture. Plotly is the visualization note on this shelf.

- Jupyter notebook tutorial on how to install, run, and use Jupyter for interactive matplotlib plotting, data analysis, and publishing code. [How to use Plotly in Python](https://plot.ly/python/ipython-notebook-tutorial/)

Plotly for Jupyter Lab:

```
jupyter labextension install @jupyterlab/plotly-extension
```

## Tutorials

With the tools named, these tutorials combine Kubernetes, Seldon, and the rest.

The same notes are in [Tutorials](../devops/full-stack-and-ops/tutorials.md).

- [Kubernetes, sklearn, s2i, GCloud, Seldon random serving for A/B testing](https://medium.com/analytics-vidhya/manage-ml-deployments-like-a-boss-deploy-your-first-ab-test-with-sklearn-kubernetes-and-b10ae0819dfe)
- [Polyaxon — training, Argo package/deployment, Seldon serving](https://medium.com/analytics-vidhya/polyaxon-argo-and-seldon-for-model-training-package-and-deployment-in-kubernetes-fa089ba7d60b)

## Deprecated links

{% hint style="warning" %}
These links and images no longer work. The original wording is kept here. A same-resource copy, when one was checked, is used above.
{% endhint %}

- Containerize your ds environment using docker compose. This address no longer opens: https://towardsdatascience.com/containerize-your-whole-data-science-environment-or-anything-you-want-with-docker-compose-e962b8ce8ce5
- Step by step tut. This address no longer opens: https://codelabs.developers.google.com/codelabs/cloud-kubeflow-e2e-gis/index.html?index=..%2F..index#0
- endtoend tut. This address no longer opens: https://journal.arrikto.com/an-end-to-end-ml-pipeline-on-prem-notebooks-kubeflow-pipelines-on-the-new-minikf-ee618b7dc7de
- KF + Seldon on ec2. This address no longer opens: https://docs.seldon.io/projects/seldon-core/en/latest/examples/kubeflow_seldon_e2e_pipeline.html
- A better airflow? This address no longer opens: http://airflow
- Redis plus spacy. This address no longer opens: https://towardsdatascience.com/spacy-redis-magic-60f25c21303d
- Venn for python. This address no longer opens: http://ow-to-create-and-customize-venn-diagrams-in-python-263555527305
