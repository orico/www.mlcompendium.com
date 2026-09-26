# Training Strategies

A model that scored well once still has to be retrained as the data moves, and the question is when and how. The page answers it with one framework for a continuous training strategy: periodic and performance-based retraining driven by data changes, window size, and what to retrain.

The same notes are in [Drift](../ai-engineering/mlops/mlops-monitoring-and-alerts.md#drift), [Incremental Learning](../decision-intelligence/incremental-learning.md), and [Online Learning](../problem-framing/online-learning.md).

The **(amazing) Framework for a successful training strategy** is [https://towardsdatascience.com/framework-for-a-successful-continuous-training-strategy-8c83d17bb9dc](https://towardsdatascience.com/framework-for-a-successful-continuous-training-strategy-8c83d17bb9dc). It lays out the choices in order: periodic, performance based, driven by data changes, dynamic window size, dynamic data selection, what to retrain and the level.
