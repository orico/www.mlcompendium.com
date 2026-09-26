# Kubernetes

Containers need an orchestrator once there are more than a few of them. The page starts with Kubernetes itself, then Helm for packaging, Kubeflow for the ML stack on top, MiniKF for running that stack locally, and ends with the courses that teach those pieces.

The same notes are in [DevOps Courses](docker.md).

## Kubernetes

Kubernetes has a steep learning curve, so the guides run from beginner to advanced. For beginners, [1](https://medium.com/containermind/a-beginners-guide-to-kubernetes-7e8ca56420b6) is Imesh Gunaratne's beginner's guide. [2](https://medium.com/faun/kubernetes-basics-for-new-users-d57fdf85adba) is Ellen Mei's basics for new users, a synthesis of the concepts she found while reading through lots of documentation for a system that manages containerized applications at scale. [3](https://medium.com/google-cloud/kubernetes-101-pods-nodes-containers-and-clusters-c1509e409e16) is Daniel Sanche's Kubernetes 101 on pods, nodes, containers, and clusters, written for newcomers who find the official documentation overwhelming. [4](https://medium.com/swlh/kubernetes-in-a-nutshell-tutorial-for-beginners-caa442dfd6c0) is Katarzyna Dusza's nutshell tutorial for those who have never played with this container orchestrator. [5](https://kubernetes.io/docs/tutorials/kubernetes-basics/) is the official Learn Kubernetes Basics, modules that walk through deploying a containerized application on a cluster, scaling the deployment, and updating the application.

For the advanced step, [1](https://www.freecodecamp.org/news/learn-kubernetes-in-under-3-hours-a-detailed-guide-to-orchestrating-containers-114ff420e882/) is Learn Kubernetes in Under 3 Hours, a detailed guide to orchestrating containers. Beyond the core system, [A list with all the tools](https://collabnix.github.io/kubetools/) is Kubetools, a curated list of Kubernetes tools.

## Helm

With the cluster running, applications still have to be installed on it repeatably. [Package manager for Kubernetes](https://helm.sh/) is Helm, the package manager for Kubernetes.

## Kubeflow

Helm installs charts. Kubeflow is the ML stack that runs on that cluster, from training to serving.

The same notes are in [Kubeflow for data science](../../mlops/full-stack-and-ops.md#kubeflow-for-data-science) and [MLOps Intro](../../mlops/mlops-intro.md).

The introductions come first. [YouTube — the easy way](https://www.youtube.com/watch?v=P5wcE4IwKgQ) is Google Cloud Tech's Machine Learning on Kubernetes with Kubeflow. [intro](https://medium.com/@amina.alsherif/how-to-get-started-with-kubeflow-187792f3e99) is Amina's how-to-get-started, which imagines running a training and inference stack wherever, with little or no configuration from one piece of hardware to the next. [intro2](https://kubernetes.io/blog/2017/12/introducing-kubeflow-composable/) is the Kubernetes blog post introducing Kubeflow as a composable, portable, scalable ML stack built for Kubernetes, at a time when machine learning was one of the fastest growing workloads on the platform. [intro3](https://medium.com/better-programming/kubeflow-pipelines-with-gpus-1af6a74ec2a) is Kubeflow Pipelines With GPUs.

Serving is where the options multiply. [Really good detailed article, for example it supports many serving options such as Seldon](https://ubuntu.com/blog/ml-serving-models-with-kubeflow-on-ubuntu-part-1) is Ubuntu's part 1 on serving models with Kubeflow, starting from the point that there are many ways to serve a trained model in and outside of Kubeflow, and what to consider for each. The [presentation](https://www.oliverwyman.com/content/dam/oliver-wyman/v2/events/2018/March/Google_London_Event/Public%20Introduction%20to%20Kubeflow.pdf) is a public introduction to Kubeflow that lays out the pieces around the model: data splitting, trainer, model validation, serving, logging, monitoring, roll-out, and portability.

The tutorials then put those pieces together end to end. The [Official example](https://github.com/kubeflow/example-seldon) is kubeflow/example-seldon, end-to-end machine learning on Kubernetes using Kubeflow and Seldon Core. [Step by step tut](http://web.archive.org/web/20190330195417/https://codelabs.developers.google.com/codelabs/cloud-kubeflow-e2e-gis/index.html?index=..%2F..index) is the archived codelab Kubeflow End to End - GitHub Issue Summarization. The [really detailed tut](https://medium.com/data-science/how-to-create-and-deploy-a-kubeflow-machine-learning-pipeline-part-1-efea7a4b650f) is Lak Lakshmanan's part 1 on taking an existing real-world TensorFlow model and turning it into a Kubeflow pipeline. [KF + Seldon on ec2](http://web.archive.org/web/20240417175403/https://docs.seldon.io/projects/seldon-core/en/latest/examples/kubeflow_seldon_e2e_pipeline.html) is the archived seldon-core documentation page, End-to-end Reusable ML Pipeline with Seldon and Kubeflow.

## MiniKF

A full cluster is heavy for learning, so Kubeflow on a laptop is MiniKF, with the on-prem pipeline notes beside it. The [youtube](https://www.youtube.com/watch?v=XZGHFktDSE0) video is Arrikto's Kubeflow Pipelines on-prem with MiniKF.

The local install has its own guide, a step-by-step guide on installing and configuring each of the kubeflow components on your local machine, and it is a really detailed tut: [https://towardsdatascience.com/how-to-create-and-deploy-a-kubeflow-machine-learning-pipeline-part-1-efea7a4b650f](https://towardsdatascience.com/how-to-create-and-deploy-a-kubeflow-machine-learning-pipeline-part-1-efea7a4b650f)


## Kubernetes courses

After MiniKF, the courses go back over the whole path in video form: Kubernetes, Helm, minikube, and Bitnami sealed secrets. [Kubernetes crash course](https://www.youtube.com/watch?v=XuSQU5Grv1g) is KodeKloud's crash course on the basics, building a microservice application.

Helm gets two videos of its own. [What is Helm](https://www.youtube.com/watch?v=kJscDZfHXrQ) is KodeKloud's explanation of Helm concepts, and [Helm Charts](https://www.youtube.com/watch?v=jUYNS90nq8U) is DevOps Journey's ultimate guide to creating Helm charts. To practice without a cloud cluster, [What is minikube](https://minikube.sigs.k8s.io/docs/start/?arch=%2Fmacos%2Fx86-64%2Fstable%2Fbinary+download) is the minikube start page: local Kubernetes that makes it easy to learn and develop, a single command away once Docker or a virtual machine environment is available. Finally, [Bitnami](https://www.youtube.com/watch?v=xd2QoV6GJlc) is DevOps & AI Toolkit on Bitnami Sealed Secrets, how to store Kubernetes secrets in git repositories.

## Deprecated links

{% hint style="warning" %}
These links and images no longer work. The original wording is kept here. A same-resource copy, when one was checked, is used above.
{% endhint %}

- Step by step tut. This address no longer opens: https://codelabs.developers.google.com/codelabs/cloud-kubeflow-e2e-gis/index.html?index=..%2F..index#0
- endtoend tut. This address no longer opens: https://journal.arrikto.com/an-end-to-end-ml-pipeline-on-prem-notebooks-kubeflow-pipelines-on-the-new-minikf-ee618b7dc7de
- KF + Seldon on ec2. This address no longer opens: https://docs.seldon.io/projects/seldon-core/en/latest/examples/kubeflow_seldon_e2e_pipeline.html
- Tutorial (MiniKF). This address no longer opens: https://journal.arrikto.com/an-end-to-end-ml-pipeline-on-prem-notebooks-kubeflow-pipelines-on-the-new-minikf-ee618b7dc7de
- ROK - save snapshot of your env. This address no longer opens: https://journal.arrikto.com/arrikto-launches-rok-and-rok-registry-93d76eb0c3a2
