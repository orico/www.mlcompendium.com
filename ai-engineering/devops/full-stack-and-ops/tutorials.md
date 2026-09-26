# Tutorials

Serving a model as an endpoint is only the first step; managing many models in production is the real job. This page holds two end-to-end tutorials for that, first A/B serving on Kubernetes and then a full training-and-serving stack.

The same notes are in [Tutorials](../../mlops/full-stack-and-ops.md#tutorials).

The first tutorial is for the moment when simply serving up endpoints is not good enough any more. [Kubernetes, scikit-learn, s2i, gcloud, and Seldon: random serving for A/B testing](https://medium.com/analytics-vidhya/manage-ml-deployments-like-a-boss-deploy-your-first-ab-test-with-sklearn-kubernetes-and-b10ae0819dfe) is Gus Cavanaugh's walkthrough of deploying a first A/B test with scikit-learn and Kubernetes, the start of model management.

The same notes are in [A/B Testing](../../../decision-intelligence/a-b-testing.md).

The second tutorial widens model management from one deployment to the whole loop of training many models and shipping the best one. [Polyaxon for training, an Argo package and deployment, and Seldon for serving](https://medium.com/analytics-vidhya/polyaxon-argo-and-seldon-for-model-training-package-and-deployment-in-kubernetes-fa089ba7d60b) is Daniel Rodriguez's version of that loop in Kubernetes, where model management means tooling and pipelines for data scientists to develop, deploy, measure, and improve models.
