# Hyper Parameter Optimization

Once a model has a metric, the next question is which settings get the best score without losing track of the experiments that produced it. The page starts with Hyperopt and its search algorithms, moves to HyperparameterHunter, which records every run, and ends with a package that compares optimizers.

The same notes are in [Drift](../ai-engineering/mlops/mlops-monitoring-and-alerts.md#drift), [HYPER PARAM GRID SEARCHES](../deep-learning/deep-neural-nets.md#hyper-param-grid-searches), and [Hyper param optimization](../deep-learning/meta-learning.md#hyper-param-optimization).

[**Using HyperOpt**](http://hyperopt.github.io/hyperopt/) is the documentation for Hyperopt, Distributed Asynchronous Hyper-parameter Optimization. It offers **Random Search** and **Tree of Parzen Estimators (TPE)**. Hyperopt has been designed to accommodate Bayesian optimization algorithms based on Gaussian processes and regression trees, but these are not currently implemented. All algorithms can be run either serially, or in parallel by communicating via [**MongoDB**](http://www.mongodb.org/).

Search is only half of the job; the other half is keeping the results, which is why this page overlaps with Mlflow, Hyperparameterhunter, hyperopt, concept drift, and unit tests. The overview that tied those together no longer opens and is kept at the end of the page. The same notes are in [ML Experiment Management](../ai-engineering/mlops/experiment-management.md).

For the search itself, [**Hyperopt**](http://hyperopt.github.io/hyperopt/) is the tool for hyperparameter search. [**HyperparameterHunter**](https://github.com/HunterMcGushion/hyperparameter_hunter) adds the record keeping: easy hyperparameter optimization and automatic result saving across machine learning algorithms and libraries. It provides a wrapper for machine learning algorithms that saves all the important data. Simplify the experimentation and hyperparameter tuning process by letting HyperparameterHunter do the hard work of recording, organizing, and learning from your tests, all while using the same libraries you already do. Don't let any of your experiments go to waste, and start doing hyperparameter optimization the way it was meant to be.

The two are not free to swap. An implementation and comparison found HH slower than HO due to usage of skopt; that write-up is also kept at the end of the page. To compare more than two, [**HumpDay**](https://github.com/microprediction/humpday) is a package that compares optimization algorithms and ranks them.

## Deprecated links

{% hint style="warning" %}
These links and images no longer work. The original wording is kept here. A same-resource copy, when one was checked, is used above.
{% endhint %}

- Mlflow, Hyperparameterhunter, hyperopt, concept drift, unit tests. This address no longer opens: https://towardsdatascience.com/putting-ml-in-production-ii-logging-and-monitoring-algorithms-91f174044e4e
- Implementation and comparison **- HH slower than HO due to usage of skopt.** This address no longer opens: https://towardsdatascience.com/putting-ml-in-production-ii-logging-and-monitoring-algorithms-91f174044e4e
