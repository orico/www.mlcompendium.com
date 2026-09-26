# Hyper Parameter Optimization

This page points at Hyperopt, HyperparameterHunter, and a comparison of optimizers. It is for searching hyperparameters without losing track of experiments.

The same notes are in [Drift](../ai-engineering/mlops/mlops-monitoring-and-alerts.md#drift), [HYPER PARAM GRID SEARCHES](../deep-learning/deep-neural-nets.md#hyper-param-grid-searches), and [Hyper param optimization](../deep-learning/meta-learning.md#hyper-param-optimization).

- Documentation for Hyperopt, Distributed Asynchronous Hyper-parameter Optimization. [**Using HyperOpt**](http://hyperopt.github.io/hyperopt/)

 **Random Search**

 **Tree of Parzen Estimators (TPE)**

 **Hyperopt has been designed to accommodate Bayesian optimization algorithms based on Gaussian processes and regression trees, but these are not currently implemented.**

 **All algorithms can be run either serially, or in parallel by communicating via** [**MongoDB**](http://www.mongodb.org/)**.**

 **Mlflow, Hyperparameterhunter, hyperopt, concept drift, unit tests.**

The same notes are in [ML Experiment Management](../ai-engineering/mlops/experiment-management.md).

 [**Hyperopt**](http://hyperopt.github.io/hyperopt/) **for hyperparameter search**
- Easy hyperparameter optimization and automatic result saving across machine learning algorithms and libraries - HunterMcGushion/hyperparameter_hunter. [**HyperparameterHunter**](https://github.com/HunterMcGushion/hyperparameter_hunter)

 **provides a wrapper for machine learning algorithms that saves all the important data. Simplify the experimentation and hyperparameter tuning process by letting HyperparameterHunter do the hard work of recording, organizing, and learning from your tests, all while using the same libraries you already do. Don't let any of your experiments go to waste, and start doing hyperparameter optimization the way it was meant to be.**
3. **Implementation and comparison - HH slower than HO due to usage of skopt.**
4. [**HumpDay**](https://github.com/microprediction/humpday) **- a package that compares optimization algorithms and ranks them**

## Deprecated links

{% hint style="warning" %}
These links and images no longer work. The original wording is kept here. A same-resource copy, when one was checked, is used above.
{% endhint %}

- Mlflow, Hyperparameterhunter, hyperopt, concept drift, unit tests. This address no longer opens: https://towardsdatascience.com/putting-ml-in-production-ii-logging-and-monitoring-algorithms-91f174044e4e
- Implementation and comparison **- HH slower than HO due to usage of skopt.** This address no longer opens: https://towardsdatascience.com/putting-ml-in-production-ii-logging-and-monitoring-algorithms-91f174044e4e
