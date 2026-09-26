# MLOps Tools

An MLOps stack is many tools, not one product. The page follows the order a change travels: CI and package repositories, Docker and Kubeflow, the workflow runners, serving, and then the services beside them, from Kafka and Redis to error tracking, metrics, and plots, ending with tutorials that combine them.

## Continuous integration

CI is the first gate before packages and containers enter the stack.

The same notes are in [Continuous Integration](../devops/full-stack-and-ops/continuous-integration.md).

The two hosted services here differ mostly in pitch. [Travis](https://travis-ci.org/) presents Travis CI as the most simple and flexible CI/CD tool for continuous integration and delivery. [Circle CI](https://circleci.com/) promises to deliver production-ready software at AI speed, validating, testing, and shipping every change with automation. GitHub Actions is the third option, and [poetry black pytest](https://medium.com/@vanflymen/blazing-fast-ci-with-github-actions-poetry-black-and-pytest-9e74299dd4a5) builds blazing fast CI with it, starting from a blank Django project.

## Package repositories

With CI named, package repositories are where the built artifacts live. PyPI is the public one. When packages must stay private, [Gemfury](https://gemfury.com/) is a hosted repository for your public and private packages, where they are safe and within reach.

## Docker for data science

Packages need a runtime. Docker is how the data-science environment travels.

The same notes are in [Docker](../devops/full-stack-and-ops/docker.md) and [Jupyter](../../data/engineering/data-science-tools.md#jupyter).

It helps to know what an image is made of before building one. [What are Docker layers](https://medium.com/@jessgreb01/digging-into-docker-layers-c22f948ed612) is Jessica digging into the contents of each layer that makes up an image. [Install on Ubuntu](https://linuxconfig.org/how-to-install-docker-on-ubuntu-18-04-bionic-beaver) covers Ubuntu 18.04 Bionic Beaver.

For data science the usual first container is a notebook. [Many Jupyter Docker images (Spark too)](https://jupyter-docker-stacks.readthedocs.io/en/latest/using/selecting.html) is the Docker Stacks guide to selecting an image. [How to run Jupyter Docker 1](https://medium.com/@rahulvaish/jupyter-docker-badd38fd6b51) is Rahul Vaish on running ready-to-go Jupyter notebooks with Docker, and [2](https://medium.com/fundbox-engineering/overview-d3759e83969c) runs a local Jupyter and JupyterLab environment the same way. When the default location runs out of space, [Tell Docker to run on a mounted disk](https://stackoverflow.com/questions/32070113/how-do-i-change-the-default-docker-container-location) is the Stack Overflow answer for moving images out of /var/lib/docker.

A container can also serve a model. [Docker, Keras, k8s, Flask serving](https://medium.com/analytics-vidhya/deploy-your-first-deep-learning-model-on-kubernetes-with-python-keras-flask-and-docker-575dc07d9e76) is Gus Cavanaugh's basic example: build a Keras model, serve it as a REST API with Flask, and deploy it with Docker and Kubernetes. Once there is more than one container, [Compose](https://docs.docker.com/compose/) is the introduction to defining and running multi-container applications. Docker Compose is simply a tool that allows you to describe a collection of multiple containers that can interact via their own network in a very straightforward way.

The same Fundbox post also works as a [Docker on Ubuntu, tutorial](https://medium.com/fundbox-engineering/overview-d3759e83969c), and [Docker for data science](https://aoyilmaz.medium.com/docker-in-data-science-and-a-friendly-beginner-to-docker-186fafdfbdeb) is the beginner-friendly overview for data scientists. A note on debugging inside containers from VS Code used to sit here and is kept at the end of the page.

## Kubeflow for data science

Once containers exist, Kubeflow is ML on Kubernetes with pipelines and serving options.

The same notes are in [Kubeflow](../devops/full-stack-and-ops/kubernetes.md#kubeflow).

The easy way in is a video: [YouTube — the easy way](https://www.youtube.com/watch?v=P5wcE4IwKgQ) is Google Cloud Tech's Machine Learning on Kubernetes with Kubeflow. The written intros follow. [intro](https://medium.com/@amina.alsherif/how-to-get-started-with-kubeflow-187792f3e99) is Amina's guide to getting started, imagining a training and inference stack that runs anywhere with little configuration. [intro2](https://kubernetes.io/blog/2017/12/introducing-kubeflow-composable/) is the Kubernetes blog post that introduced Kubeflow as a composable, portable, scalable ML stack built for Kubernetes. [intro3](https://medium.com/better-programming/kubeflow-pipelines-with-gpus-1af6a74ec2a) runs Kubeflow Pipelines with GPUs.

Serving is where Kubeflow branches. There are many ways to serve a trained model in and outside of Kubeflow, and the Ubuntu post on those alternatives is a [Really good detailed article, for example it supports many serving options such as Seldon](https://ubuntu.com/blog/ml-serving-models-with-kubeflow-on-ubuntu-part-1). The [presentation](https://www.oliverwyman.com/content/dam/oliver-wyman/v2/events/2018/March/Google_London_Event/Public%20Introduction%20to%20Kubeflow.pdf) is a public introduction to Kubeflow that walks model building, validation, serving, logging, monitoring, and roll-out.

Tutorials put the pieces together:

 - [Official example](https://github.com/kubeflow/example-seldon): end-to-end machine learning on Kubernetes using Kubeflow and Seldon Core.
 - [Step by step tut](http://web.archive.org/web/20190330195417/https://codelabs.developers.google.com/codelabs/cloud-kubeflow-e2e-gis/index.html?index=..%2F..index): the Kubeflow End to End GitHub Issue Summarization codelab.
 - [really detailed tut](https://medium.com/data-science/how-to-create-and-deploy-a-kubeflow-machine-learning-pipeline-part-1-efea7a4b650f): Lak Lakshmanan takes an existing real-world TensorFlow model and operationalizes it as a Kubeflow pipeline.
 - [KF + Seldon on ec2](http://web.archive.org/web/20240417175403/https://docs.seldon.io/projects/seldon-core/en/latest/examples/kubeflow_seldon_e2e_pipeline.html): the seldon-core documentation's end-to-end reusable ML pipeline with Seldon and Kubeflow.

## Airflow

Kubeflow is not the only orchestrator. Airflow schedules the DAGs around the models.

The same notes are in [Airflow](../../data/processing/data-pipelines.md#airflow) and [Data Pipelines](../../data/processing/data-pipelines.md).

[Airflow](https://airflow.apache.org/) is a platform created by the community to programmatically author, schedule and monitor workflows. [Airflow in 5 minutes](https://medium.com/swlh/apache-airflow-in-5-minutes-c005b4b11b26) by Ashish Kumar introduces it as an open-source tool for orchestrating complex workflows and data processing pipelines. [Airflow 2.0 tutorial](https://medium.com/apache-airflow/apache-airflow-2-0-tutorial-41329bbf7211) by Tomasz Urbaszek covers the many new features in 2.0. With the concepts of DAGs and operators in place, [Simple ETL](https://adenilsoncastro.medium.com/apache-airflow-the-etl-02-f4ac25f4d9b4) by Adnilson Castro pulls the latest news from the NewsAPI into an SQLite database. [Airflow Scheduler & Webserver](https://medium.com/analytics-vidhya/manage-your-workflows-with-apache-airflow-e7b0e45544a8) by Shritam Kumar Mund is about managing workflows with those two components, and [Airflow for DS](https://medium.com/data-science/apache-airflow-for-data-science-how-to-write-your-first-dag-in-10-minutes-9d6e884def72) by Dario Rade writes a first DAG in 10 minutes.

## Prefect

Airflow is one runner. Prefect is another workflow option for ML.

The same notes are in [ML Architecture](ml-architecture.md) and [Patterns in practice](mlops.md#patterns-in-practice).

[ML workflows with Prefect](https://www.youtube.com/watch?v=SP6WqCRUkNc) is Kevin Kho's MLOps Meetup #94 talk on orchestrating machine learning workflows with Prefect.

## MetaFlow

Beside Prefect, Metaflow is Netflix's framework for the same pipeline problem.

The same notes are in [ML Architecture](ml-architecture.md) and [Patterns in practice](mlops.md#patterns-in-practice).

[Intro](https://www.youtube.com/watch?v=JCbOI_1ZA5E) is AICamp's talk on Metaflow as the ML infrastructure at Netflix, and [what is](https://www.youtube.com/watch?v=bxVAniteuQs) is AI Scientist's Metaflow hello-world program.

Several of the runner and serving notes above also live at their original Towards Data Science addresses:

- A step-by-step guide on installing and configuring each of the kubeflow components on your local machine. really detailed tut. [https://towardsdatascience.com/how-to-create-and-deploy-a-kubeflow-machine-learning-pipeline-part-1-efea7a4b650f](https://towardsdatascience.com/how-to-create-and-deploy-a-kubeflow-machine-learning-pipeline-part-1-efea7a4b650f)
- Learn to work with Bash and Python operators in Apache Airflow by coding an entire DAG. Airflow for DS. [https://towardsdatascience.com/apache-airflow-for-data-science-how-to-write-your-first-dag-in-10-minutes-9d6e884def72](https://towardsdatascience.com/apache-airflow-for-data-science-how-to-write-your-first-dag-in-10-minutes-9d6e884def72)
- Medium on DL as a service by Nir Orman. [https://towardsdatascience.com/serving-deep-learning-algorithms-as-a-service-6aa610368fde](https://towardsdatascience.com/serving-deep-learning-algorithms-as-a-service-6aa610368fde)
- Towards Data Science. Scaling ML on the cloud. [https://towardsdatascience.com/scalable-efficient-big-data-analytics-machine-learning-pipeline-architecture-on-cloud-4d59efc092b5](https://towardsdatascience.com/scalable-efficient-big-data-analytics-machine-learning-pipeline-architecture-on-cloud-4d59efc092b5)

## Seldon

Workflows still need a serving layer. Seldon puts models on Kubernetes.

The same notes are in [MLOps Deployment](mlops-deployment.md).

Seldon runs in k8s, and the open question the author left is Seldon-core vs seldon-deploy (what are the differences?). The reading answers it from three sides. [Serving graph, recipe file](https://becominghuman.ai/seldon-inference-graph-pipelined-model-serving-211c6b095f62) is Shirish Hirekodi on the Seldon inference graph, serving models in a pipeline. [Descriptive intro](https://medium.com/seldon-open-source-machine-learning/introducing-seldon-core-machine-learning-deployment-for-kubernetes-e10e94c19fd8) is Alex Housley introducing Seldon Core, machine learning deployment for Kubernetes. [Sales pitch intro](https://medium.com/seldon-open-source-machine-learning/introducing-seldon-deploy-c390d11af20c) is Alex Housley again, introducing Seldon Deploy.

## Serving models

With Seldon named, the broader serving patterns come next, and Dapr closes them.

The same notes are in [MLOps Deployment](mlops-deployment.md).

The patterns have been catalogued twice. Shai Yanovski's slides from the Data Science Leads IL meeting, June 2020, at Explorium.ai are the ML system design patterns [res](https://docs.google.com/presentation/d/1pSkklHkBySMnJNODshW8NZVpBSqOsbJBWeEq8RrS0M4/edit#slide=id.g81f938aa2b_0_47). The [git](https://github.com/mercari/ml-system-design-pattern) repo is Mercari's system design patterns for machine learning. Seldon, from the section above, is one serving option among them. [Medium on DL as a service by Nir Orman](https://medium.com/data-science/serving-deep-learning-algorithms-as-a-service-6aa610368fde) is about running deep learning algorithms as a service, and [Scaling ML on the cloud](https://medium.com/data-science/scalable-efficient-big-data-analytics-machine-learning-pipeline-architecture-on-cloud-4d59efc092b5) is Satish Chandra Gupta's architecture for a high-throughput, low-latency big data pipeline on cloud.

The services around a model also need a runtime. [Dapr](https://github.com/dapr/dapr) is a portable, serverless, event-driven runtime that makes it easy for developers to build resilient, stateless and stateful microservices that run on the cloud and edge and embraces the diversity of languages and developer frameworks.

 Dapr codifies the best practices for building microservice applications into open, independent, building blocks that enable you to build portable applications with the language and framework of your choice. Each building block is independent and you can use one, some, or all of them in your application.

## FastAPI

Serving often needs an HTTP API. FastAPI is the typed Flask alternative here.

The FastAPI framework is high performance, easy to learn, fast to code, and ready for production, and its alternatives page is why it reads as [Flask on steroids with variable parameters](https://fastapi.tiangolo.com/alternatives/).

## Kafka for data science

APIs are not the only pipe. Kafka is the streaming bus for data science.

The same notes are in [Kafka](../devops/full-stack-and-ops/key-value-db.md#kafka).

Kafka is a messaging system, and messaging matters because moving data between systems is a hugely important piece of infrastructure. Kafka in a Nutshell covers [What is, terminology, use cases](https://sookocheff.com/post/kafka/kafka-in-a-nutshell/).

## Redis for data science

Beside Kafka, Redis is the in-memory store those services lean on.

The same notes are in [Key Value DB](../devops/full-stack-and-ops/key-value-db.md).

Start with what it is, vs [memcached](https://medium.com/@pankaj.itdeveloper/memcached-vs-redis-which-one-to-choose-d5177482dc42), a comparison of caching choices for application performance. [Redis cluster](https://medium.com/@inthujan/introduction-to-redis-redis-cluster-6c7760c8ebbc) is the introduction to scaling it out, and [Redis plus spaCy](https://medium.com/data-science/spacy-redis-magic-60f25c21303d) is the "Spacy + Redis = Magic" pairing for NLP. Note: Redis is a managed dictionary; its strength lies when you have a lot of data that needs to be queried and managed and you don’t want to hard-code it, for example. The [Long tutorial](https://realpython.com/python-redis/) is Real Python on how to use Redis with Python.

## Sentry

Once traffic flows, Sentry watches the Python stack for errors.

[For Python](https://sentry.io/for/python/), your code is telling you more than what your logs let on. Sentry’s full stack monitoring gives you full visibility into your code, so you can catch issues before they become downtime.

## Statsd

Sentry is errors. StatsD is the gauges and buckets for metrics.

StatsD is a daemon for easy but powerful stats aggregation, and its Python example shows the [Statistics server, with gauges/buckets and flushing/sending ability](https://github.com/statsd/statsd/blob/master/examples/python_example.py).

## Visualization

Metrics need a picture. Plotly is the visualization note on this shelf.

[How to use Plotly in Python](https://plot.ly/python/ipython-notebook-tutorial/) is the Jupyter notebook tutorial on how to install, run, and use Jupyter for interactive plotting, data analysis, and publishing code.

Plotly for Jupyter Lab:

```
jupyter labextension install @jupyterlab/plotly-extension
```

## Tutorials

With the tools named, these tutorials combine Kubernetes, Seldon, and the rest.

The same notes are in [Tutorials](../devops/full-stack-and-ops/tutorials.md).

When serving up endpoints is no longer enough and you need model management, Gus Cavanaugh's [Kubernetes, sklearn, s2i, GCloud, Seldon random serving for A/B testing](https://medium.com/analytics-vidhya/manage-ml-deployments-like-a-boss-deploy-your-first-ab-test-with-sklearn-kubernetes-and-b10ae0819dfe) deploys a first A/B test. Daniel Rodriguez's [Polyaxon — training, Argo package/deployment, Seldon serving](https://medium.com/analytics-vidhya/polyaxon-argo-and-seldon-for-model-training-package-and-deployment-in-kubernetes-fa089ba7d60b) treats model management as training many models with different data, parameters, features, and algorithms, then deploying the best one.

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
- Using VS Code to debug containers. This address no longer opens: https://nirradi.medium.com/vsc-vs-pycharm-developing-inside-docker-containers-4892c83d30e4
