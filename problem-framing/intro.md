# Problem Framing

The same table is a different problem under a different label regime, and there is no reinforcement-setting page here. The chapter first names the task, then walks the learning settings from weak and scarce labels to streams, few-shot data, and the choice of a model family.
After this chapter the reader can name the task and the learning setting, and reinforcement learning is in [Reinforcement Learning](../decision-intelligence/reinforcement-learning.md).

## Tasks

The first step is to see where a problem sits among machine learning setups. The [Overview](types-of-machine-learning.md) maps how those setups differ by how much supervision and feedback they use, so you can place a problem before you pick a method. When one example carries more than one label, [Multi Label Classification](multi-label-classification.md) covers the main ways to assign several labels per example, with problem-transformation recipes, metrics, and tooling.

## Learning settings

Once the task is named, the label regime decides the setting. [Weakly Supervised](weakly-supervised.md) is for labels that are incomplete, inaccurate, or inexact rather than fully supervised. [Label Algorithms](label-algorithms.md) handles unbalanced labels, label propagation and spreading, and label noise, for when labels are scarce, noisy, or need to be spread across a graph of related examples. [Semi Supervised](semi-supervised.md) is training with lots of unlabeled data and only a little labeled data. When labels are expensive, [Active Learning](active-learning.md) chooses which examples a human labels next, with uncertainty and diversity sampling and query-by-committee. [Online Learning](online-learning.md) is for models that update as labeled examples arrive in a stream, including non-stationary settings. [N-Shot Learning](n-shot-learning.md) covers zero-, one-, and few-shot learning, adapting with only a handful of labeled examples per class. With the task and the setting known, [Model Families](model-families.md) maps the problem shape to an algorithm family.
