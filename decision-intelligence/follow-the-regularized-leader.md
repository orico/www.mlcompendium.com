# Follow the regularized leader

Learning online means deciding every round and paying a loss each time; Follow the Regularized Leader (FTRL) is an algorithm built for that setting. The page goes from the McMahan paper, to intuitive explanations, to the Keras optimizer that implements it, with a deprecated address kept at the end.

The same notes are in [Incremental Learning](incremental-learning.md), [Online Learning](../problem-framing/online-learning.md), and [Regularization](../predictive-ml/regularization.md).

The source is [FTRL the paper](https://research.google.com/pubs/archive/41159.pdf) by McMahan et al., case studies from a deployed ad click-through rate (CTR) prediction system, a massive-scale learning problem, built on an FTRL-Proximal online learning algorithm. For the intuition behind it, [FTRL](https://www.quora.com/What-is-an-intuitive-explanation-of-Follow-the-Regularized-Leader-FTRL-algorithm) by Nicolo Compolongo answers what an intuitive explanation of the algorithm is:

> The “Follow the Regularized Leader” algorithm stems from the online learning setting, where the learning process is sequential. In this setting, an online player makes a decision in every round and suffers a loss.

A longer version of that intuition is [A thorough Medium article](https://medium.com/@dhirajreddy13/factorization-machines-and-follow-the-regression-leader-for-dummies-7657652dce69) by Dhiraj Reddy, Factorization Machines and Follow The Regularized Leader for dummies, which starts from ad click prediction: the model learns from the user's click or no click at time T and updates its weights to decide which ad to show at time T+1. To use it rather than derive it, [Keras on FTRL](https://keras.io/api/optimizers/ftrl/) is the Keras Team's documentation for the Ftrl optimizer, which describes it this way:

> "Follow The Regularized Leader" (FTRL) is an optimization algorithm developed at Google for click-through rate prediction in the early 2010s. It is most suitable for shallow models with large and sparse feature spaces. The algorithm is described by [McMahan et al., 2013](https://research.google.com/pubs/archive/41159.pdf). The Keras version has support for both online L2 regularization (the L2 regularization described in the paper above) and shrinkage-type L2 regularization (which is the addition of an L2 penalty to the loss function).

## Deprecated links

{% hint style="warning" %}
These links and images no longer work. The original wording is kept here. A same-resource copy, when one was checked, is used above.
{% endhint %}

- FTRL the paper by McMahan et al. This address no longer opens: https://static.googleusercontent.com/media/research.google.com/en/pubs/archive/41159.pdf
