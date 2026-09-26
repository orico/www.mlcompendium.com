# Multi Label Classification

Some examples need more than one label, and a plain multiclass classifier cannot say that. The page starts with what multilabel classification is, then the methods and metrics, the problem-transformation recipes, an EfficientNet case, and the tooling.

The same notes are in [MULTI LABEL/OUTPUT](../deep-learning/deep-neural-nets.md#multi-labeloutput).

## What multilabel classification is

The starting point is the definition and its contrast with multiclass, taken from the mlr tutorial on multilabel classification. (what is?) [Multilabel classification is a classification problem where multiple target labels can be assigned to each observation instead of only one like in multiclass classification.](https://mlr.mlr-org.com/articles/tutorial/multilabel.html)

Two different approaches exist for multilabel classification. Problem transformation methods try to transform the multilabel classification into binary or multiclass classification problems. Algorithm adaptation methods adapt multiclass algorithms so they can be applied directly to the problem. I.e., the two approaches are to use a classifier that does multi label, or to use any classifier with a wrapper that compares each two labels.

## Methods and metrics

Once the two approaches are named, the next question is how each method works and how to score a prediction that is a set of labels. The same notes are in [Precision Recall ROC AUC](../evals/evaluation-metrics.md#precision--recall--roc--auc).

Jesse Read's two-part "Multi-label Classification" talk from Universidad Carlos III de Madrid is the great [PDF](https://users.ics.aalto.fi/jesse/talks/Multilabel-Part01.pdf) that explains about multi label classification and especially metrics, [part 2 here](https://users.ics.aalto.fi/jesse/talks/Multilabel-Part02.pdf) continues it. For the full catalogue, [An awesome Paper](https://www.researchgate.net/publication/273859036_Multi-Label_Classification_An_Overview) explains all of these methods in detail!

## Problem transformation methods

The paper's problem-transformation recipes are the concrete form of the first approach, and they are a genuine list, PT1 to PT6:

PT1: for each sample select one label, remove all others.

PT2: remove every sample which has multi labels.

PT3: for every combo of labels create a single-label, i.e. A&B, A&C etc.

PT4: (most common) create L datasets, for each label learn a binary representation, i.e., is it there or not.

PT5: duplicate each sample with only one of its labels

PT6: read the paper

There are other approaches for doing it within algorithms, they rely on the ideas PT3/4/5/6 implemented in the algorithms, or other tricks. They also introduce Label cardinality and label density.

## EfficientNet for multilabel

The recipes meet a real model in GumGum's threat-detection work, where brand safety for advertisers means scanning a publisher's inventory for threats that are often not the salient object in the image. [Efficient net](https://medium.com/gumgum-tech/multi-label-classification-for-threat-detection-part-1-60318b90ce11) is part 1 of that series, and [part 2](https://medium.com/gumgum-tech/multi-label-image-classifier-for-threat-detection-with-fp16-inference-part-2-40fe0f9a93b3), by Aditya Ramesh, runs exploratory data analysis on the multilabel dataset and FP16 inference. EfficientNet is based on a network derived from a neural architecture search and novel compound scaling method is applied to iteratively build more complex network which achieves state of the art accuracy on multiclass classification tasks. Compound scaling refers to increasing the network dimensions in all three scaling formats using a novel strategy.

## Tools

After the model, the work is in the libraries. Multi label confusion matrices with sklearn are the scikit-learn side, and the [Scikit multilearn package](http://scikit.ml/index.html), hosted at scikit.ml, is the scikit-multilearn side.

An awesome Paper that explains all of these methods in detail, also available here! [http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.104.9401&rep=rep1&type=pdf](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.104.9401&rep=rep1&type=pdf)

## Deprecated links

{% hint style="warning" %}
These links and images no longer work. The original wording is kept here. A same-resource copy, when one was checked, is used above.
{% endhint %}

- (what is?) Multilabel classification is a classification problem where multiple target labels can be assigned to each observation instead of only one like in multiclass classification. This address no longer opens: https://mlr-org.github.io/mlr-tutorial/devel/html/multilabel/index.html
