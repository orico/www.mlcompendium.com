# Kubernetes

Containers need an orchestrator. This page is Kubernetes, then Helm, Kubeflow, MiniKF, and the courses that teach those pieces.

The same notes are in [DevOps Courses](docker.md).

## Kubernetes

These notes are beginner and advanced Kubernetes guides before Helm packages anything.

- For beginners:
 - [1](https://medium.com/containermind/a-beginners-guide-to-kubernetes-7e8ca56420b6)
 - [2](https://medium.com/faun/kubernetes-basics-for-new-users-d57fdf85adba)
 - [3](https://medium.com/google-cloud/kubernetes-101-pods-nodes-containers-and-clusters-c1509e409e16)
 - [4](https://medium.com/swlh/kubernetes-in-a-nutshell-tutorial-for-beginners-caa442dfd6c0)
 - Learn Kubernetes Basics. Learn Kubernetes Basics. [5](https://kubernetes.io/docs/tutorials/kubernetes-basics/)
- Advanced
 - Learn Kubernetes in Under 3 Hours: A Detailed Guide to Orchestrating Containers. [1](https://www.freecodecamp.org/news/learn-kubernetes-in-under-3-hours-a-detailed-guide-to-orchestrating-containers-114ff420e882/)
- Kubetools - Curated List of Kubernetes Tools. [A list with all the tools](https://collabnix.github.io/kubetools/)

## Helm

With the cluster named, Helm is the package manager for Kubernetes.

- Helm. Helm. [Package manager for Kubernetes](https://helm.sh/)

## Kubeflow

Helm installs charts. Kubeflow is the ML stack that runs on that cluster.

The same notes are in [Kubeflow for data science](../../mlops/full-stack-and-ops.md#kubeflow-for-data-science) and [MLOps Intro](../../mlops/mlops-intro.md).

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

## MiniKF

Kubeflow on a laptop is MiniKF, with the on-prem pipeline notes beside it.

- Kubeflow Pipelines on-prem with MiniKF, by Arrikto. [youtube](https://www.youtube.com/watch?v=XZGHFktDSE0)

- A step-by-step guide on installing and configuring each of the kubeflow components on your local machine. really detailed tut. [https://towardsdatascience.com/how-to-create-and-deploy-a-kubeflow-machine-learning-pipeline-part-1-efea7a4b650f](https://towardsdatascience.com/how-to-create-and-deploy-a-kubeflow-machine-learning-pipeline-part-1-efea7a4b650f)


## Kubernetes courses

After MiniKF, these courses cover Kubernetes, Helm, minikube, and Bitnami sealed secrets.

- Kubernetes Crash Course: Learn the Basics and Build a Microservice Application, by KodeKloud. [Kubernetes crash course](https://www.youtube.com/watch?v=XuSQU5Grv1g)


3. Helm
 - What is Helm? | Helm Concepts Explained | KodeKloud. [What is Helm](https://www.youtube.com/watch?v=kJscDZfHXrQ)
 - How to Create Helm Charts - The Ultimate Guide, by DevOps Journey. [Helm Charts](https://www.youtube.com/watch?v=jUYNS90nq8U)
- minikube start. minikube start. [What is minikube](https://minikube.sigs.k8s.io/docs/start/?arch=%2Fmacos%2Fx86-64%2Fstable%2Fbinary+download)
- Bitnami Sealed Secrets - How To Store Kubernetes Secrets In Git Repositories, by DevOps & AI Toolkit. [Bitnami](https://www.youtube.com/watch?v=xd2QoV6GJlc)

## Deprecated links

{% hint style="warning" %}
These links and images no longer work. The original wording is kept here. A same-resource copy, when one was checked, is used above.
{% endhint %}

- Step by step tut. This address no longer opens: https://codelabs.developers.google.com/codelabs/cloud-kubeflow-e2e-gis/index.html?index=..%2F..index#0
- endtoend tut. This address no longer opens: https://journal.arrikto.com/an-end-to-end-ml-pipeline-on-prem-notebooks-kubeflow-pipelines-on-the-new-minikf-ee618b7dc7de
- KF + Seldon on ec2. This address no longer opens: https://docs.seldon.io/projects/seldon-core/en/latest/examples/kubeflow_seldon_e2e_pipeline.html
- Tutorial (MiniKF). This address no longer opens: https://journal.arrikto.com/an-end-to-end-ml-pipeline-on-prem-notebooks-kubeflow-pipelines-on-the-new-minikf-ee618b7dc7de
- ROK - save snapshot of your env. This address no longer opens: https://journal.arrikto.com/arrikto-launches-rok-and-rok-registry-93d76eb0c3a2
