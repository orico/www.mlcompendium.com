# Full-Stack & Ops

### CONTINUOUS INTEGRATION

1. [Travis](https://travis-ci.org)
2. [Circle CI](https://circleci.com)
3. [TeamCity](https://www.jetbrains.com/teamcity/)
4. [Jenkins](https://www.jenkins.io)
5. Github Actions
   1. [poetry black pytest](https://medium.com/@vanflymen/blazing-fast-ci-with-github-actions-poetry-black-and-pytest-9e74299dd4a5)

### PACKAGE REPOSITORIES

1. Pypi - public
2. [Gemfury](https://gemfury.com) - private

### DEVOPS / SRE&#x20;

* [Dev vs ops, vs devops vs sre - history and details by google.](https://www.youtube.com/watch?v=tEylFyxbDLE\&list=PLIivdWyY5sqJrKl7D2u-gmis8h9K66qoj\&index=2)
* [Sre vs devops](https://medium.com/hackernoon/sre-vs-devops-the-dilemma-f7054714525c)
* [Cloudops vs devops](https://victorops.com/blog/what-is-cloudops-vs-devops)
* [Itops vs devops](https://www.graylog.org/post/itops-vs-devops-what-is-the-difference)&#x20;
* [AIOps](https://www.appdynamics.com/what-is-ai-ops/) - “AIOps platforms utilize big data, modern machine learning and other advanced analytics technologies to directly and indirectly enhance IT operations (monitoring, automation and service desk) functions with proactive, personal and dynamic insight. AIOps platforms enable the concurrent use of multiple data sources, data collection methods, analytical (real-time and deep) technologies, and presentation technologies.
* [Definition](https://dzone.com/articles/dev-vs-ops-and-devops)

![](https://lh3.googleusercontent.com/-q3xPZ\_ASRimnV37VYLqPZxoKFSQPKSrkIQdnBHaxCPOkP9rTZT7t-6n98Zp4NKwG8QuuFNlk4omZv234Dx8QrBohcVzh7kLoiOwYmfHF5skCBKt6q8zRaHZrn2r481i3QXzr7hH)

### [Cap theorem](https://towardsdatascience.com/cap-theorem-and-distributed-database-management-systems-5c2be977950e)

* [Cap](https://www.confluent.io/blog/turning-the-database-inside-out-with-apache-samza/) 2015
* [Cap is changing](https://www.infoq.com/articles/cap-twelve-years-later-how-the-rules-have-changed/)

![](https://lh6.googleusercontent.com/cDV78UprJnSuEkoqVRRzg9K\_a8YlvYAQlJ\_YDj6CRMqypYp0BwFkHhErzcMtt8h0LWKd4cPk3ftCpRyLTMxLNNxCNJ6nAUZNoEh0umdNzsAdIt0IUMDBJT\_uvdWgD9UxHLpHisiS)

### Docker&#x20;

* [What are docker layers](https://medium.com/@jessgreb01/digging-into-docker-layers-c22f948ed612)?
* [Install on ubuntu](https://linuxconfig.org/how-to-install-docker-on-ubuntu-18-04-bionic-beaver)
* [Many jupyter docker images (spark too)](https://jupyter-docker-stacks.readthedocs.io/en/latest/using/selecting.html)
* [How to run jupyter docker 1](https://medium.com/@rahulvaish/jupyter-docker-badd38fd6b51), [2](https://medium.com/fundbox-engineering/overview-d3759e83969c)
* [Tell docker to run on a mounted disk](https://stackoverflow.com/questions/32070113/how-do-i-change-the-default-docker-container-location)
* [Docker, keras, k8s, flask serving](https://medium.com/analytics-vidhya/deploy-your-first-deep-learning-model-on-kubernetes-with-python-keras-flask-and-docker-575dc07d9e76)
* [Compose](https://docs.docker.com/compose/) - run multi coker applications.
* [Docker on ubuntu, tutorial](https://medium.com/fundbox-engineering/overview-d3759e83969c)
* [Containerize your ds environment using docker compose](https://towardsdatascience.com/containerize-your-whole-data-science-environment-or-anything-you-want-with-docker-compose-e962b8ce8ce5) - Docker-Compose is simply a tool that allows you to describe a collection of multiple containers that can interact via their own network in a very straight forward way,&#x20;
* [docker for data science](https://aoyilmaz.medium.com/docker-in-data-science-and-a-friendly-beginner-to-docker-186fafdfbdeb)

### Kubernetes

* For beginners:
  * [1](https://medium.com/containermind/a-beginners-guide-to-kubernetes-7e8ca56420b6), [2](https://medium.com/faun/kubernetes-basics-for-new-users-d57fdf85adba)\*, [3](https://medium.com/google-cloud/kubernetes-101-pods-nodes-containers-and-clusters-c1509e409e16)\*, [4](https://medium.com/swlh/kubernetes-in-a-nutshell-tutorial-for-beginners-caa442dfd6c0), [5](https://kubernetes.io/docs/tutorials/kubernetes-basics/)\*, 6,&#x20;
* Advanced&#x20;
  * [1](https://www.freecodecamp.org/news/learn-kubernetes-in-under-3-hours-a-detailed-guide-to-orchestrating-containers-114ff420e882/), 2, 3,

### Helm

* [Package manager for kubernetes](https://helm.sh)

### Kubeflow

* [Youtube - the easy way,](https://www.youtube.com/watch?v=P5wcE4IwKgQ) [intro](https://medium.com/@amina.alsherif/how-to-get-started-with-kubeflow-187792f3e99)\*, [intro2\*](https://kubernetes.io/blog/2017/12/introducing-kubeflow-composable/), [intro3](https://medium.com/better-programming/kubeflow-pipelines-with-gpus-1af6a74ec2a),
* [Really good detailed article, for example it supports many serving options such as seldon](https://ubuntu.com/blog/ml-serving-models-with-kubeflow-on-ubuntu-part-1)
* [presentation](https://www.oliverwyman.com/content/dam/oliver-wyman/v2/events/2018/March/Google\_London\_Event/Public%20Introduction%20to%20Kubeflow.pdf)
* Tutorials:
  * [Official example](https://github.com/kubeflow/example-seldon)
  * [Step by step tut](https://codelabs.developers.google.com/codelabs/cloud-kubeflow-e2e-gis/index.html?index=..%2F..index#0)\*
  * [endtoend tut](https://journal.arrikto.com/an-end-to-end-ml-pipeline-on-prem-notebooks-kubeflow-pipelines-on-the-new-minikf-ee618b7dc7de),&#x20;
  * [really detailed tut](https://towardsdatascience.com/how-to-create-and-deploy-a-kubeflow-machine-learning-pipeline-part-1-efea7a4b650f)
  * KF + [Seldon on ec2](https://docs.seldon.io/projects/seldon-core/en/latest/examples/kubeflow\_seldon\_e2e\_pipeline.html)

### MiniKF

* [Tutorial](https://journal.arrikto.com/an-end-to-end-ml-pipeline-on-prem-notebooks-kubeflow-pipelines-on-the-new-minikf-ee618b7dc7de), [youtube](https://www.youtube.com/watch?v=XZGHFktDSE0)
* [ROK - save snapshot of your env](https://journal.arrikto.com/arrikto-launches-rok-and-rok-registry-93d76eb0c3a2)

### S2i

* Builds docker images out of gits

### Terraform

1. [youtube course](https://www.youtube.com/watch?v=SLB\_c\_ayRMo)

### AirFlow

* [Airflow](https://airflow.apache.org) is a platform created by the community to programmatically author, schedule and monitor workflows.
* [Airflow in 5 minutes](https://medium.com/swlh/apache-airflow-in-5-minutes-c005b4b11b26) by Ashish Kumar
* [Airflow 2.0 tutorial](https://medium.com/apache-airflow/apache-airflow-2-0-tutorial-41329bbf7211) by Tomasz Urbaszek
* [Simple  ETL](https://adenilsoncastro.medium.com/apache-airflow-the-etl-02-f4ac25f4d9b4) by Adnilson Castro
* [Airflow Scheduler & Webserver](https://medium.com/analytics-vidhya/manage-your-workflows-with-apache-airflow-e7b0e45544a8) by Shritam Kumar Mund &#x20;

### Prefect

* [A better airflow ](http://airflow)?

### Seldon&#x20;

* Runs in k8s
* Seldon-core seldon-deploy (what are the differences?)
* [Serving graph, recipe file](https://becominghuman.ai/seldon-inference-graph-pipelined-model-serving-211c6b095f62)
* [Descriptive intro ](https://medium.com/seldon-open-source-machine-learning/introducing-seldon-core-machine-learning-deployment-for-kubernetes-e10e94c19fd8)
* [Sales pitch intro](https://medium.com/seldon-open-source-machine-learning/introducing-seldon-deploy-c390d11af20c)

### Tutorials&#x20;

* [Kubernetes, sklearn, s2i, gcloud, seldon random serving for ab testing](https://medium.com/analytics-vidhya/manage-ml-deployments-like-a-boss-deploy-your-first-ab-test-with-sklearn-kubernetes-and-b10ae0819dfe)
* [Polyaxon - training, argo-package/deployment , seldin -serving](https://medium.com/analytics-vidhya/polyaxon-argo-and-seldon-for-model-training-package-and-deployment-in-kubernetes-fa089ba7d60b)

### AWS Lambda

* [Comparison](https://www.bluematador.com/blog/serverless-in-aws-lambda-vs-fargate) against aws fargate

### rabbitMQ&#x20;

* [Producer broker, consumer - a tutorial on what is RMQ](https://www.cloudamqp.com/blog/2015-05-18-part1-rabbitmq-for-beginners-what-is-rabbitmq.html)
* [Part2.3](https://www.cloudamqp.com/blog/2015-05-21-part2-3-rabbitmq-for-beginners\_example-and-sample-code-python.html) - python code
* [Part 3](https://www.cloudamqp.com/blog/2015-05-27-part3-rabbitmq-for-beginners\_the-management-interface.html) -  managing
* [Part 4](https://www.cloudamqp.com/blog/2015-09-03-part4-rabbitmq-for-beginners-exchanges-routing-keys-bindings.html)

### [ActiveMQ ](http://activemq.apache.org)

\- Apache ActiveMQ™ is the most popular open source, multi-protocol, Java-based messaging serve

### Kafka&#x20;

* [Web](https://kafka.apache.org)
* [Medium - really good short intro](https://medium.com/hacking-talent/kafka-all-you-need-to-know-8c7251b49ad0)
* [Intro](https://medium.com/@jcbaey/what-is-apache-kafka-e9e73884e367), [Intro 2](https://medium.com/@patelharshali136/apache-kafka-tutorial-kafka-for-beginners-a58140cef84f),&#x20;
* [Kafka in a nutshell](https://medium.com/@aiven\_io/apache-kafka-in-a-nutshell-df10dfcc7dc) - But even these solutions came up short in some cases. For example, RabbitMQ stores messages in DRAM until the DRAM is completely consumed, at which point messages are written to disk, [severely impacting performance](https://blog.mavenhive.in/which-one-to-use-and-when-rabbitmq-vs-apache-kafka-7d5423301b58).

Also, the routing logic of AMQP can be fairly complicated as opposed to Apache Kafka. For instance, each consumer simply decides which messages to read in Kafka.

In addition to message routing simplicity, there are places where developers and DevOps staff prefer Apache Kafka for its high throughput, scalability, performance, and durability; although, developers still swear by all three systems for various reasons.

* [Apache Kafka Kafka](https://medium.com/develbyte/introduction-to-zookeeper-bcda7ef136cd) is a pub-sub messaging system. It uses Zookeeper to detect crashes, to implement topic discovery, and to maintain production and consumption state for topics.
* [Tutorial on putting a model in kafka and using zoo keeper](https://towardsdatascience.com/putting-ml-in-production-i-using-apache-kafka-in-python-ce06b3a395c8) with code.

### Zoo keeper

* Intro [1](https://medium.com/@rinu.gour123/role-of-apache-zookeeper-in-kafka-monitoring-configuration-c5bd1a7e4226), [2-usecases](https://medium.com/@bikas.katwal10/zookeeper-introduction-designing-a-distributed-system-using-zookeeper-and-java-7f1b108e236e), [3\*](https://medium.com/@ben2460/about-apache-zookeeper-distributed-lock-1a990315e05c), [4](https://www.tutorialspoint.com/zookeeper/zookeeper\_overview.htm)
* What is [1](https://medium.com/@gavindya/what-is-zookeeper-db8dfc30fc9b), [2](https://medium.com/rahasak/apache-zookeeper-31b2091657a8)
* It is a [service discovery in a nutshell, kafka is using it to allow discovery, registration etc of services. So that customers can subscribe and get their publication. ](https://www.quora.com/What-is-ZooKeeper)

### ELK

* [Elastic on ELK](https://www.elastic.co/what-is/elk-stack)
* [Logz.io on ELK](https://logz.io/learn/complete-guide-elk-stack)
* [Hackernoon intro](https://hackernoon.com/elastic-stack-a-brief-introduction-794bc7ff7d4f)

### Logz.io

* [Intro,](https://www.youtube.com/watch?v=LqJYeeTss9Q) [What is](https://www.youtube.com/watch?v=6VVig5tnTJE)

### Sentry

* [For python,](https://sentry.io/for/python/) Your code is telling you more than what your logs let on. Sentry’s full stack monitoring gives you full visibility into your code, so you can catch issues before they become downtime.

### Kafka for DS

1. [What is, terminology, use cases](https://sookocheff.com/post/kafka/kafka-in-a-nutshell/#:\~:text=Kafka%20topics%20are%20divided%20into,from%20a%20topic%20in%20parallel.)

### Redis for DS

1. What is, vs [memcached](https://medium.com/@pankaj.itdeveloper/memcached-vs-redis-which-one-to-choose-d5177482dc42)
2. [Redis cluster](https://medium.com/@inthujan/introduction-to-redis-redis-cluster-6c7760c8ebbc)
3. [Redis plus spacy](https://towardsdatascience.com/spacy-redis-magic-60f25c21303d)
4. Note: redis is a managed dictionary its strength lies when you have a lot of data that needs to be queries and managed and you don’t want to hard code it, for example.
5. [Long tutorial](https://realpython.com/python-redis/)

### Statsd

1. [Statistics server, with gauges/buckets and flushing/sending ability](https://github.com/statsd/statsd/blob/master/examples/python\_example.py)

### FastAPI

1. [Flask on steroids with variable parameters](https://fastapi.tiangolo.com/alternatives/)

### SnowFlake / Redshift

1. Snowflake [Intro and demo](https://www.youtube.com/watch?v=dUL8GO4ZK9s)
2. [The three pillars - snowflake](https://towardsdatascience.com/why-you-need-to-know-snowflake-as-a-data-scientist-d4e5a87c2f3d)
3. [Snowflake vs redshift on medium](https://towardsdatascience.com/redshift-or-snowflake-e0e3ea427dbc)
4. [SF vs RS](https://www.xplenty.com/blog/redshift-vs-snowflake/)
5. [RS vs BQ](https://www.xplenty.com/blog/redshift-vs-bigquery-comprehensive-guide/)
6. [SF vs BQ](https://www.xplenty.com/blog/snowflake-vs-bigquery/)
7. [Feature engineering in snowflake](https://towardsdatascience.com/feature-engineering-in-snowflake-1730a1b84e5b)

### Programming Concepts

[Dependency injection](https://www.freecodecamp.org/news/a-quick-intro-to-dependency-injection-what-it-is-and-when-to-use-it-7578c84fa88f/) - based on [SOLID](https://scotch.io/bar-talk/s-o-l-i-d-the-first-five-principles-of-object-oriented-design#toc-single-responsibility-principle) the class should do one thing, so we are letting other classes create 3rd party/class objects for us instead of doing it internally, either by init passing or by injecting in runtime.\


[SOLID](https://scotch.io/bar-talk/s-o-l-i-d-the-first-five-principles-of-object-oriented-design#toc-single-responsibility-principle) - the five principles of object oriented. \
\


### Visualization

1. [How to use plotly in python](https://plot.ly/python/ipython-notebook-tutorial/)

Plotly for jupyter lab “jupyter labextension install @jupyterlab/plotly-extension”

1. [Venn for python](http://ow-to-create-and-customize-venn-diagrams-in-python-263555527305)

### Serving Models

1. ML SYSTEM DESIGN PATTERNS, [res](https://docs.google.com/presentation/d/1pSkklHkBySMnJNODshW8NZVpBSqOsbJBWeEq8RrS0M4/edit#slide=id.g81f938aa2b\_0\_47), [git](https://github.com/mercari/ml-system-design-pattern)
2. Seldon
3. [Medium on DL as a service by Nir Orman](https://towardsdatascience.com/serving-deep-learning-algorithms-as-a-service-6aa610368fde)
4. [Scaling ML on the cloud](https://towardsdatascience.com/scalable-efficient-big-data-analytics-machine-learning-pipeline-architecture-on-cloud-4d59efc092b5)
5. [Dapr](https://github.com/dapr/dapr) is a portable, serverless, event-driven runtime that makes it easy for developers to build resilient, stateless and stateful microservices that run on the cloud and edge and embraces the diversity of languages and developer frameworks.

Dapr codifies the best practices for building microservice applications into open, independent, building blocks that enable you to build portable applications with the language and framework of your choice. Each building block is independent and you can use one, some, or all of them in your application.\


#### EXPERIMENT MANAGEMENT

1. [All the alternatives](https://blog.valohai.com/top-machine-learning-platforms)
2. Cnvrg.io -
   1. Manage - Easily navigate machine learning with dashboards, reproducible data science, dataset organization, experiment tracking and visualization, a model repository and more
   2. Build - Run and track experiments in hyperspeed with the freedom to use any compute environment, framework, programming language or tool - no configuration required
   3. Automate - Build more models and automate your machine learning from research to production using reusable components and drag-n-drop interface
3. Comet.ml - Comet lets you track code, experiments, and results on ML projects. It’s fast, simple, and free for open source projects.
4. Floyd - notebooks on the cloud, similar to colab / kaggle, etc. gpu costs 4$/h
5. [Trains - open source](https://heartbeat.fritz.ai/trains-all-aboard-ba92a728eb6d)
6. Missing link - RIP
7. Spark
   1. [Rdds vs datasets vs dataframes](https://databricks.com/blog/2016/07/14/a-tale-of-three-apache-spark-apis-rdds-dataframes-and-datasets.html)
   2. [What are Rdds?](https://www.quora.com/What-are-resilient-distributed-datasets-RDDs-How-do-they-help-Spark-with-its-awesome-speed)
   3. [keras , tf,  spark](https://medium.com/qubida-analytics-blog/build-a-deep-learning-image-classification-pipeline-with-spark-keras-and-tensorflow-3bf26fda15e6)
   4. [Repartition vs coalesce ](https://medium.com/@mrpowers/managing-spark-partitions-with-coalesce-and-repartition-4050c57ad5c4)
   5. [Best practices](https://www.bi4all.pt/en/news/en-blog/apache-spark-best-practices/)
8. Databricks
   1. [Koalas](https://github.com/databricks/koalas) - pandas API on Apache Spark
   2. [Intro to DB on spark](https://www.youtube.com/watch?v=DqihOzZl5jM\&list=PLTPXxbhUt-YV-CwJTiE36C-0le8wlFJ5G\&index=5), has some basic sklearn-like tool and other custom operations such as single-vector-based aggregator for using features as an input to a model
   3. [Pyspark.ml](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html)
   4. [Keras as a single node (no spark)](https://docs.databricks.com/applications/deep-learning/single-node-training/keras.html)
   5. [Horovod for distributed keras (and more)](https://docs.databricks.com/applications/deep-learning/distributed-training/mnist-tensorflow-keras.html)
   6. [Documentations](https://docs.databricks.com/index.html) (read me, has all libraries)
   7. [Medium tutorial](https://towardsdatascience.com/how-to-train-your-neural-networks-in-parallel-with-keras-and-apache-spark-ea8a3f48cae6), explains the 3 pros of DB with examples of using with native and non native algos
      1. Spark sql
      2. Mlflow
      3. Streaming
      4. SystemML DML using keras models.
   8. [systemML notebooks (didnt read)](http://systemml.apache.org/get-started.html#sample-notebook)
   9. [Sklearn notebook example](https://docs.databricks.com/\_static/notebooks/scikit-learn.html)
   10. [Utilizing spark nodes](https://databricks.com/blog/2016/02/08/auto-scaling-scikit-learn-with-apache-spark.html) for grid searching with sklearn
       1. from spark\_sklearn import GridSearchCV
   11. [How can we leverage](https://databricks-prod-cloudfront.cloud.databricks.com/public/13fe59d17777de29f8a2ffdf85f52925/5638528096339357/1867405/6918044996430578/latest.html) our existing experience with modeling libraries like [scikit-learn](http://scikit-learn.org/stable/index.html)? We'll explore three approaches that make use of existing libraries, but still benefit from the parallelism provided by Spark.

These approaches are:

* Grid Search
* Cross Validation
* Sampling (random, chronological subsets of data across clusters)

1. Github [spark-sklearn](https://github.com/databricks/spark-sklearn) (needs to be compared to what spark has internally)
   1. [Ref:](https://mapr.com/blog/predicting-airbnb-listing-prices-scikit-learn-and-apache-spark/) It's worth pausing here to note that the architecture of this approach is different than that used by MLlib in Spark. Using spark-sklearn, we're simply distributing the cross-validation run of each model (with a specific combination of hyperparameters) across each Spark executor. Spark MLlib, on the other hand, will distribute the internals of the actual learning algorithms across the cluster.
   2. The main advantage of spark-sklearn is that it enables leveraging the very rich set of [machine learning](https://mapr.com/ebook/machine-learning-logistics/) algorithms in scikit-learn. These algorithms do not run natively on a cluster (although they can be parallelized on a single machine) and by adding Spark, we can unlock a lot more horsepower than could ordinarily be used.
   3. Using [spark-sklearn](https://github.com/databricks/spark-sklearn) is a straightforward way to throw more CPU at any machine learning problem you might have. We used the package to reduce the time spent searching and reduce the error for our estimator
2. [Airbnb example using spark and sklearn,cross\_val& grid search comparison vs joblib](https://mapr.com/blog/predicting-airbnb-listing-prices-scikit-learn-and-apache-spark/)
3. [Sklearn example 2, tfidf, ](http://cdn2.hubspot.net/hubfs/438089/notebooks/ML/scikit-learn/demo\_-\_1\_-\_sklearn.html)
4. [Tracking experiments](https://docs.databricks.com/applications/mlflow/tracking.html)
   1. [example](https://docs.databricks.com/applications/mlflow/tracking-examples.html#train-a-scikit-learn-model-and-save-in-scikit-learn-format)
5. [Saving loading deployment](https://docs.databricks.com/applications/mlflow/models.html#examples)
   1. [Aws sagemaker](https://docs.databricks.com/applications/mlflow/model-examples.html#scikit-learn-model-deployment-on-sagemaker)
   2. [Medium](https://towardsdatascience.com/a-different-way-to-deploy-a-python-model-over-spark-2da4d625f73e) and sklearn random trees
6. [How to productionalize your model using db spark 2.0 on youtube](https://databricks.com/session/how-to-productionize-your-machine-learning-models-using-apache-spark-mllib-2-x)

### API GATEWAY

1. [what is](https://www.javatpoint.com/introduction-to-api-gateways)

### NGINX

1. [NGINX](https://www.nginx.com/resources/glossary/nginx/) is open source software for web serving, reverse proxying, caching, load balancing, media streaming, and more. It started out as a web server designed for maximum performance and stability. In addition to its HTTP server capabilities, NGINX can also function as a proxy server for email (IMAP, POP3, and SMTP) and a reverse proxy and load balancer for HTTP, TCP, and UDP servers.
2. Cloudflare on [what is a reverse proxy](https://www.cloudflare.com/learning/cdn/glossary/reverse-proxy/)
3. "[One advantage of using NGINX as an API gateway ](https://www.nginx.com/blog/deploying-nginx-plus-as-an-api-gateway-part-1/#:\~:text=One%20advantage%20of%20using%20NGINX,deploy%20a%20separate%20API%20gateway.)is that it can perform that role while simultaneously acting as a reverse proxy, load balancer, and web server for existing HTTP traffic. If NGINX is already part of your application delivery stack then it is generally unnecessary to deploy a separate API gateway"

