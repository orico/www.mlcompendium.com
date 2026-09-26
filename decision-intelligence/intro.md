# Decision Intelligence

This chapter is how an action is chosen and how you know it worked for the business. It starts with experiments you plan and test, moves to decisions that keep being made as data arrives, and ends with causality when you can only observe.
After it, the reader can design a test, a bandit, or a sequential decision, and run tracking lives in [Experiment management](../ai-engineering/mlops/experiment-management.md).

## Experimentation

The first way to know an action worked is to plan a test so you learn from fewer runs. [Design Of Experiments](design-of-experiments.md) is where that planning starts, with introductions and tutorials in Python, R, and Matlab. [DOE Tools](doe-tools.md) is the tooling rather than the theory: generators and notebooks such as doePy and PyDoe. [Factorial Design](factorial-design.md) is the factorial experiment as one DoE layout. Once the runs are planned, [Hypothesis Testing](hypothesis-testing.md) is the statistical language behind the decision they feed. [A/B Testing](a-b-testing.md) applies that language to online controlled experiments: whether a product change moved the metric that matters. [Multi Armed Bandits](multi-armed-bandits.md) keeps allocating trials while learning which arm pays off, and [Contextual Bandits](contextual-bandits.md) chooses actions using side information, not only arm history.

## Sequential decisions

A bandit already decides again at every step, and the next pages make that the whole setting. [Active Learning Algorithms](active-learning-algorithms.md) is active-style learning when labels arrive in large streams, starting with the Passive Aggressive classifier. [Incremental Learning](incremental-learning.md) extends a model as new input arrives instead of training only once. [Follow the regularized leader](follow-the-regularized-leader.md) is FTRL, an online learning algorithm. [Reinforcement Learning](reinforcement-learning.md) is agents that learn by interacting for reward.

## Causality

When you cannot run the test at all, the comparison has to come from observation. [Propensity Score Matching](propensity-score-matching.md) is propensity score matching and propensity modeling for observational comparisons. [Root Cause Effects (RCE/RCA)](root-cause-effects-rce-rca.md) is root-cause analysis: finding which faults or graph structure explain observed incidents.
