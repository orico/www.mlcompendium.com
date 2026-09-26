# AI Engineering

The model is now a service, and the service has to be built, deployed, and monitored.
After this chapter the reader can place feature stores, experiment tracking, deployment, and monitoring on the path from a trained model to a running system, and feature stores continue [Data Processing](../data/processing-intro.md), where raw values were scaled, transformed, imputed, and annotated into the shape a learner expects.

# Ops

The pages above are about the method: what it is and how it works. The pages in this half are about how you build, ship, govern, and run the system.

That work splits into five groups. [MLOps](mlops/mlops-intro.md) is the gap between ML in a notebook and ML in production. [DataOps](../data/engineering/sql.md) starts from asking a warehouse a question in SQL. [DevOps](devops/devops-strategy.md) opens with the strategy that keeps delivery speed from stalling. [DevSecOps](devsecops/tbd.md) puts security inside the delivery loop, not after it. [Architecture](architecture/problems.md) begins with the signs that a system which looks distributed still behaves like one monolith.

A makeover of these pages was done, and [How it was done](../appendix/ops-makeover.md) records the process.

## Model platform

The model platform comes first, because everything else in this half serves a model that has to run. [Definitions](definitions.md) separates the Ops names that get mixed: DataOps, MLOps, GitOps, DevOps, and DevSecOps. [MLOps Intro](mlops/mlops-intro.md) lays out the building blocks and reading list for getting from notebook to production. [MLOps Patterns](mlops/mlops.md) collects the shapes production ML keeps repeating, then two end-to-end systems that use them. [MLOps Teams](mlops/mlops-teams.md) asks who owns that work: machine learning engineers, or a separate MLOps role.

With the owners named, the pages follow the model's path. [Feature Stores & Feature Pipelines](mlops/feature-stores-and-feature-pipelines.md) is the pipeline that builds features and the store that serves the same features to training and serving. [ML Experiment Management](mlops/experiment-management.md) keeps track of runs that multiply faster than anyone can remember them. [Model Formats](mlops/model-formats.md) is how a trained model leaves the notebook in a form another application can read. [MLOps Deployment](mlops/mlops-deployment.md) puts that model somewhere that can serve it. [MLOps Tools](mlops/full-stack-and-ops.md) walks the many tools of the stack, from CI and Docker to Kubeflow, workflow runners, and serving. [ML Model Monitoring & Alerts](mlops/mlops-monitoring-and-alerts.md) keeps a shipped model from going blind. [AI As Data](mlops/ai-as-data.md) treats models as AI tables you query with SQL, and [ML Architecture](mlops/ml-architecture.md) closes the group with end-to-end architecture examples.

## Architecture and security

Once the model runs, the system around it has to scale, stay software, and stay secure. [System Design](architecture/system-design.md) is system design interviews, scalability reading, and practice for a service that has to scale. [Development Concepts](architecture/development-concepts.md) is design patterns, dependency injection, and SOLID, because ML code still has to be software. [Problems](architecture/problems.md) lists the coupling symptoms of a distributed monolith. On the security side, [DevSecOps Definitions](devsecops/tbd.md) gives three definitions of DevSecOps, [DevSecOps Tools](devsecops/tools.md) points at Atlassian's list of the toolchain, and [DevSecOps Concepts](devsecops/concepts.md) is the simplified information-security vocabulary to have clear before the tools.

## DevOps

The last group is the delivery machinery that every service above depends on. [DevOps Strategy](devops/devops-strategy.md) points at eight strategies and how to put them in place, and [DevOps Tools](devops/full-stack-and-ops/README.md) is the section that holds the tools. [Tutorials](devops/full-stack-and-ops/tutorials.md) are two end-to-end walkthroughs: A/B serving, and a training-and-serving stack. [Continuous Integration](devops/full-stack-and-ops/continuous-integration.md) is how code lands continuously. [Docker](devops/full-stack-and-ops/docker.md) makes a data-science environment run the same way everywhere, and [Kubernetes](devops/full-stack-and-ops/kubernetes.md) orchestrates those containers. [Cloud Objects](devops/full-stack-and-ops/cloud-objects.md) compares AWS Lambda with Fargate for serverless compute. [Key Value DB](devops/full-stack-and-ops/key-value-db.md) covers RabbitMQ, ActiveMQ, Kafka, ksqlDB, and ZooKeeper for the messages and state services pass. [API Gateway](devops/full-stack-and-ops/api-gateway.md) is the single front door for services. [Infrastructure As code](devops/full-stack-and-ops/infrastructure-as-code.md) turns cloud resources into code with Terraform. [Logs](devops/full-stack-and-ops/logs.md) is the trail production systems leave, [ELK](devops/full-stack-and-ops/elk.md) is the stack to search and visualize it, and [SLO](devops/full-stack-and-ops/slo.md) turns uptime promises into measurable objectives with Sloth.
