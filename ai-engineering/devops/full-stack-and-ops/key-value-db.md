# Key Value DB

Services pass messages and state to each other, and something has to carry that traffic reliably. The page starts with RabbitMQ as the classic message broker, sets ActiveMQ beside it, then moves to Kafka and the reasons people choose it, ksqlDB for querying Kafka streams, and ZooKeeper, the coordination service Kafka relies on.

The same notes are in [Redis for data science](../../mlops/full-stack-and-ops.md#redis-for-data-science).

## RabbitMQ

The first broker is RabbitMQ, and its own definition says what a broker does:

> RabbitMQ is an open-source message-broker software that originally implemented the Advanced Message Queuing Protocol and has since been extended with a plug-in architecture to support Streaming Text Oriented Messaging Protocol, MQ Telemetry Transport, and other protocols.

CloudAMQP's RabbitMQ for beginners series teaches it in order. [Producer broker, consumer — a tutorial on what is RMQ](https://www.cloudamqp.com/blog/2015-05-18-part1-rabbitmq-for-beginners-what-is-rabbitmq.html) is part 1, which explains what RabbitMQ and message queuing are and the important concepts. [Part2.3](https://www.cloudamqp.com/blog/2015-05-21-part2-3-rabbitmq-for-beginners_example-and-sample-code-python.html) is the tutorial with example source code for Python and the client library Pika. [Part 3](https://www.cloudamqp.com/blog/2015-05-27-part3-rabbitmq-for-beginners_the-management-interface.html) outlines the management interface, where queues can be created, deleted, and listed from the browser. [Part 4](https://www.cloudamqp.com/blog/2015-09-03-part4-rabbitmq-for-beginners-exchanges-routing-keys-bindings.html) covers the different types of exchanges and when you should use each.

## [ActiveMQ](http://activemq.apache.org/)

Beside RabbitMQ sits the Java option. Apache ActiveMQ™ is the most popular open source, multi-protocol, Java-based messaging server.

## Kafka

Brokers differ, and Kafka is built around a different idea: a log of events rather than a queue that empties.

The same notes are in [Implementation](../../../data/engineering/data-contract.md#implementation) and [Kafka for data science](../../mlops/full-stack-and-ops.md#kafka-for-data-science).

Apache Kafka is a distributed event store and stream-processing platform. Confluent, an IBM Company, has two course trailers to start with: [Kafka 101 YouTube](https://www.youtube.com/watch?v=j4bqyAMMb7o&list=PLa7VYi0yPIH0KbnJQcMv5N9iW8HkZHztH) is Apache Kafka 101 for Beginners, and [event sourcing & event storage](https://www.youtube.com/watch?v=p_sSRwpBkgs&list=PLa7VYi0yPIH1TXGUoSUqXgPMD2SQXEXxj&index=1) is the Event Sourcing and Event Storage with Apache Kafka® course. The project's own site is the Apache Kafka [Web](https://kafka.apache.org/) page.

The written intros explain the model behind it. [Medium — really good short intro](https://medium.com/hacking-talent/kafka-all-you-need-to-know-8c7251b49ad0) is Maria Valcam's "Kafka: All you need to know": a publish/subscribe system where producers write records that one or more consumers read, and records cannot be deleted or modified once sent, the distributed commit log. [Intro](https://medium.com/@jcbaey/what-is-apache-kafka-e9e73884e367) starts from the need to send and receive notifications in micro-services and high-availability infrastructures, and covers typical use cases, what Kafka is not, and getting-started links. [Intro 2](https://medium.com/@patelharshali136/apache-kafka-tutorial-kafka-for-beginners-a58140cef84f) is a beginners' tutorial on what Kafka is, its history, its architecture, components, and partitions.

Why pick Kafka over RabbitMQ is the question the next note answers. [Kafka in a nutshell](https://medium.com/@aiven_io/apache-kafka-in-a-nutshell-df10dfcc7dc) — But even these solutions came up short in some cases. For example, RabbitMQ stores messages in DRAM until the DRAM is completely consumed, at which point messages are written to disk, severely impacting performance.

Also, the routing logic of AMQP can be fairly complicated as opposed to Apache Kafka. For instance, each consumer simply decides which messages to read in Kafka.

In addition to message routing simplicity, there are places where developers and DevOps staff prefer Apache Kafka for its high throughput, scalability, performance, and durability; although, developers still swear by all three systems for various reasons.

Kafka does not coordinate itself, which is where ZooKeeper first appears. [Apache Kafka](https://medium.com/develbyte/introduction-to-zookeeper-bcda7ef136cd) is a pub-sub messaging system. It uses ZooKeeper to detect crashes, to implement topic discovery, and to maintain production and consumption state for topics. To see it serve a model, the [Tutorial on putting a model in Kafka and using ZooKeeper](https://medium.com/data-science/putting-ml-in-production-i-using-apache-kafka-in-python-ce06b3a395c8) with code is Javier Rodriguez Zaurin's "Putting ML in production I: using Apache Kafka in Python."

## KSQLDB

Once events live in Kafka, the next step is asking questions of them without writing a consumer. KSQLDB 101 uses Kafka Streams to run queries over Kafka, and the course is on [youtube](https://www.youtube.com/watch?v=oThQzCXjuk4&list=PLa7VYi0yPIH3ulxsOf5g43_QiB-HOg5_Y).

## ZooKeeper

Kafka already depends on coordination, so the last piece is ZooKeeper, the discovery and lock service underneath. Intro [1](https://medium.com/@rinu.gour123/role-of-apache-zookeeper-in-kafka-monitoring-configuration-c5bd1a7e4226) is Rinu Gour on the role of ZooKeeper in Kafka, for monitoring and configuration. [2 — use cases](https://medium.com/@bikas.katwal10/zookeeper-introduction-designing-a-distributed-system-using-zookeeper-and-java-7f1b108e236e) is Bikas Katwal's tutorial with a practical example: a small distributed system of replicated Spring Boot servers that uses ZooKeeper for cluster state and leader election. [3](https://medium.com/@ben2460/about-apache-zookeeper-distributed-lock-1a990315e05c) is Ben Yaakobi on ZooKeeper and cluster synchronization, a centralized service for configuration, naming, distributed synchronization, and group services that is basically a filesystem of ZNodes. [4](https://www.tutorialspoint.com/zookeeper/zookeeper_overview.htm) is the TutorialsPoint overview: ZooKeeper is a distributed co-ordination service to manage large set of hosts, solving a complicated coordination problem with a simple architecture and API.

For what ZooKeeper is at its core, What is [1](https://medium.com/@gavindya/what-is-zookeeper-db8dfc30fc9b) starts from the challenges of distributed applications, data inconsistency and lack of consensus, and describes the centralized service ZooKeeper provides for configuration and synchronization. [2](https://medium.com/rahasak/apache-zookeeper-31b2091657a8) starts from distributed systems as nodes that coordinate their actions by message passing, and the coordination they have to perform together. In short, it is a [service discovery in a nutshell; Kafka is using it to allow discovery, registration etc of services. So that customers can subscribe and get their publication.](https://www.quora.com/What-is-ZooKeeper)

## Deprecated links

{% hint style="warning" %}
These links and images no longer work. The original wording is kept here. A same-resource copy, when one was checked, is used above.
{% endhint %}

- severely impacting performance. This address no longer opens: https://blog.mavenhive.in/which-one-to-use-and-when-rabbitmq-vs-apache-kafka-7d5423301b58
- Tutorial on putting a model in Kafka and using zoo keeper. This address no longer opens: https://towardsdatascience.com/putting-ml-in-production-i-using-apache-kafka-in-python-ce06b3a395c8
