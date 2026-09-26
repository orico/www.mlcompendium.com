# Docker

A data-science environment has to run the same way everywhere, and Docker is how it gets packaged. The page goes from what an image is made of, to installing Docker and running Jupyter in it, to Compose for several containers, then Source-to-Image and a full beginner course.

The same notes are in [DevOps Courses](docker.md), [Docker for data science](../../mlops/full-stack-and-ops.md#docker-for-data-science), and [Jupyter](../../../data/engineering/data-science-tools.md#jupyter).

An image is a stack of layers, so that is the first thing to understand. [What are Docker layers](https://medium.com/@jessgreb01/digging-into-docker-layers-c22f948ed612) is Jessica's dig into those layers, written while trying to view the contents of each layer that made up an image. With that picture, the next step is getting Docker onto a machine: [Install on Ubuntu](https://linuxconfig.org/how-to-install-docker-on-ubuntu-18-04-bionic-beaver) covers installing Docker on Ubuntu 18.04 Bionic Beaver.

For data science the usual first container is a notebook. [Many Jupyter Docker images (Spark too)](https://jupyter-docker-stacks.readthedocs.io/en/latest/using/selecting.html) is the Selecting an Image page of the Docker Stacks documentation. [How to run Jupyter Docker 1](https://medium.com/@rahulvaish/jupyter-docker-badd38fd6b51) is Rahul Vaish's "Jupyter + Docker!", on how Docker lets you run ready-to-go Jupyter notebooks, and [2](https://medium.com/fundbox-engineering/overview-d3759e83969c) is the Fundbox post on running a local Jupyter and JupyterLab environment with Docker. Images take disk space, and [Tell Docker to run on a mounted disk](https://stackoverflow.com/questions/32070113/how-do-i-change-the-default-docker-container-location) is the Stack Overflow question on moving the default location, /var/lib/docker, to a bigger mounted drive. Once a notebook runs, a model can run too: [Docker, Keras, k8s, Flask serving](https://medium.com/analytics-vidhya/deploy-your-first-deep-learning-model-on-kubernetes-with-python-keras-flask-and-docker-575dc07d9e76) is Gus Cavanaugh's basic example of building a Keras model, serving it as a REST API with Flask, and deploying it with Docker and Kubernetes.

A real environment is rarely one container. [Compose](https://docs.docker.com/compose/) is the Docker documentation's introduction to using Docker Compose to define and run multi-container applications. The same Fundbox post appears again here as [Docker on Ubuntu, tutorial](https://medium.com/fundbox-engineering/overview-d3759e83969c). Docker Compose is simply a tool that allows you to describe a collection of multiple containers that can interact via their own network in a very straightforward way. [Docker for data science](https://aoyilmaz.medium.com/docker-in-data-science-and-a-friendly-beginner-to-docker-186fafdfbdeb) brings those pieces back to the data-science environment this page started with. A post on using VS Code to debug containers used to sit here; that address no longer opens and is kept at the end of the page.

## S2i

Writing Dockerfiles by hand is not the only way to get an image. Source-to-Image, S2I, builds Docker images out of gits, straight from the repository.


## Docker course

After those separate pieces, one course ties them together from the start. [Docker in 2 hours](https://www.youtube.com/watch?v=fqMOX6JJhGo&list=RDQMjGhJ6Dhkx4Q&start_radio=1) is freeCodeCamp.org's Docker Tutorial for Beginners, a full DevOps course on how to run applications in containers.

## Deprecated links

{% hint style="warning" %}
These links and images no longer work. The original wording is kept here. A same-resource copy, when one was checked, is used above.
{% endhint %}

- Containerize your ds environment using docker compose. This address no longer opens: https://towardsdatascience.com/containerize-your-whole-data-science-environment-or-anything-you-want-with-docker-compose-e962b8ce8ce5
- Using VS Code to debug containers. This address no longer opens: https://nirradi.medium.com/vsc-vs-pycharm-developing-inside-docker-containers-4892c83d30e4
