# Evals

A model is not a result until it has a score, a split, and a baseline.
After this chapter the reader can choose a metric, a validation scheme, and a test that would catch a broken dataset, and calibration, explanation, and fairness wait in Responsible AI.

The score comes first. [Evaluation Metrics](evaluation-metrics.md) collects the supervised and unsupervised measures, from accuracy and precision/recall through ROC, F1, and perplexity, with a reality check on the metric-learning literature. A score is only as honest as the data under it, so [Datasets Reliability & Correctness](validation.md) is about why we shouldn't trust models: dataset ablation, shortcut cues, and behavioral testing that ask whether a dataset actually requires the reasoning you think it does.

Once the score and the data can be trusted, the model has to keep earning them. [Training Strategies](training-strategies.md) is the framework for a continuous training strategy, with periodic and performance-based retraining driven by data changes, window size, and what to retrain. [Hyper Parameter Optimization](hyper-parameter-optimization.md) is searching hyperparameters with Hyperopt, HyperparameterHunter, and a comparison of optimizers, without losing track of experiments.

The last two pages turn evaluation into something a pipeline runs and a field compares. [Data & Model Tests](data-and-model-tests.md) validates data in ML pipelines and unit tests models, from Great Expectations-style data checks to pytest and mocks. [Benchmarking](benchmarking.md) collects benchmarks across datasets, algorithms, hardware, cloud providers, and platforms, including NLP and multi-task learning leaderboards and notes on scaling networks.

All of it rests on the split being real. [Data Scientists — Stop Using A Random Seed](https://cohenori.medium.com/data-scientists-stop-using-a-random-seed-fd8f2b83ed50) (April 2022) is why a score needs a real split.
