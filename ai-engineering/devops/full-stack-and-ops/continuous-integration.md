# Continuous Integration

Code has to land continuously, which means every change is built and tested automatically. The page starts with the CI products and a GitHub Actions example, then moves to the Argo courses that carry the same delivery path into Kubernetes.

The same notes are in [Continuous integration](../../mlops/full-stack-and-ops.md#continuous-integration), [MLOps Course](../../mlops/mlops-course.md), and [Model Testing](../../../evals/data-and-model-tests.md#model-testing).

The first choice is which CI service runs those builds. [Travis](https://travis-ci.org/) presents itself as the most simple and flexible CI/CD tool for continuous integration and continuous delivery. [Circle CI](https://circleci.com/) promises to deliver production-ready software at AI speed, validating, testing, and shipping every change with automation. [TeamCity](https://www.jetbrains.com/teamcity/) is the CI/CD solution by JetBrains. [Jenkins](https://www.jenkins.io/) is the open source automation server that developers use to reliably build, test, and deploy their software.

GitHub Actions is the option that lives next to the code, and [poetry black pytest](https://medium.com/@vanflymen/blazing-fast-ci-with-github-actions-poetry-black-and-pytest-9e74299dd4a5) is a blazing fast CI example with GitHub Actions, Poetry, Black, and Pytest, built on a new Django project from scratch.


## Argo courses

CI ends where deployment begins, and on Kubernetes Argo covers that next stretch. The five Argo videos are all by DevOps & AI Toolkit and build on each other. [Events](https://www.youtube.com/watch?v=sUPkGChvD54&list=PLyicRj904Z9_dGuNs6AN5Khljjn9ssbQ6) is Argo Events, the event-based dependency manager for Kubernetes. [Workflows & Pipelines](https://www.youtube.com/watch?v=UMaivwrAyTA) is Argo Workflows and Pipelines for CI/CD, machine learning, and other Kubernetes workflows. [Argo CD](https://www.youtube.com/watch?v=vpWQeoaiRM4) applies GitOps principles to manage a production environment in Kubernetes. [Rollouts](https://www.youtube.com/watch?v=84Ky0aPbHvY) is Argo Rollouts, canary deployments made easy in Kubernetes. [How to harness all of the above together](https://www.youtube.com/watch?v=XNXJtxkUKeY) is "Automation of Everything", combining Argo Events, Workflows & Pipelines, CD, and Rollouts.
