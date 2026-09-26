# MLOps Deployment

Training is not shipping. The page is about putting a model into a place that can serve it, and it starts with Lightning for PyTorch.

The same notes are in [Seldon](full-stack-and-ops.md#seldon) and [Serving models](full-stack-and-ops.md#serving-models).

## Lightning

With the deployment problem named, Lightning is a framework for training and deploying PyTorch models without living in YAML and Kubernetes.

[Lightning](https://lightning.ai/): a framework to train and deploy PyTorch models. Focus on the science, not the engineering. The deployment piece is Lightning Apps: a framework to build composable, reactive ML workflows. Bye YAML and Kubernetes, hello Python. The [Examples](https://lightning.ai/docs/app/stable/#build-self-contained-components) in the Lightning AI docs, from the PyTorch Lightning creators, show how to build self-contained components; the docs now present Lightning as an AI cloud to build, train, and deploy your own models.
