# Distribution

A dataset always sits under some probability shape, and the work starts by naming that shape before comparing two of them. The page moves from distribution types, through the Gaussian that so many tests assume, into comparing distributions and then distance methods between them.
The same notes are in [Distribution Transformation](distribution-transformation.md) and [Probability](probability.md).

## Types

Naming a shape starts with what a distribution is: (What are?) probabilities in a distribution always add up to 1. MachineLearningMastery's "A Gentle Introduction to Statistical Data Distributions" gives [More distribution explanations](https://machinelearningmastery.com/statistical-data-distributions/) for readers who want the definitions in prose before the pictures.

The fastest overview is a crib sheet. A very good explanation (Cloudera crib sheet) is the Cloudera Blog post on common probability distributions, the data scientist's crib sheet: [https://blog.cloudera.com/blog/2015/12/common-probability-distributions-the-data-scientists-crib-sheet/](https://blog.cloudera.com/blog/2015/12/common-probability-distributions-the-data-scientists-crib-sheet/)

<figure><img src="../.gitbook/assets/gimg-1b1a1d76129b.png" alt=""><figcaption><p>Common probability distributions crib sheet.</p></figcaption></figure>

When the crib sheet is too terse, "Statistical Distributions" is [A very wordy explanation](http://people.stern.nyu.edu/adamodar/New_Home_Page/StatFile/statdistns.htm), and its second figure is below.

<figure><img src="../.gitbook/assets/gimg-3e477c0f7117.png" alt=""><figcaption><p>Figure 2 from the wordy distribution explanation.</p></figcaption></figure>

One type gets its own note: Poisson and Poisson process, explained on Towards Data Science at [https://towardsdatascience.com/the-poisson-distribution-and-poisson-process-explained-4e2cb17d459](https://towardsdatascience.com/the-poisson-distribution-and-poisson-process-explained-4e2cb17d459)


## Gaussian / Normal Distribution

Among all those types, one shape is assumed far more often than the rest, and the folk rule says it plainly: “ if you collect data and it is not normal, “you need to collect more data”. The source of that line no longer opens and is kept at the end of the page.

For why the one-sample t-test assumes normality, the Stack Exchange question has [Beautiful graphs](https://stats.stackexchange.com/questions/116550/why-do-we-have-to-assume-normality-for-a-one-sample-t-test); it starts from the central limit theorem, under which the sampling distribution of the sample means is normal, so the standard error and a 95% confidence interval can be computed to accept or reject the null hypothesis.

The Quora answer on why we use the normal distribution sums it up in one sentence: of all continuous distributions over the real line with a given variance, the normal uniquely maximizes differential entropy. In plainer terms, [The normal distribution is popular for two reasons:](https://www.quora.com/Why-do-we-use-the-normal-distribution-The-normal-is-an-approximation-Why-dont-we-use-a-simpler-distribution-with-simpler-numbers-to-memorize-If-it-is-an-approximation-does-it-have-to-be-so-specific)

1. It is the most common distribution in nature (as distributions go)
2. An enormous number of statistical relationships become clear and tractable if one assumes the normal.

Sure, nothing in real life exactly matches the Normal. But it is uncanny how many things come close.

This is partly due to the Central Limit Theorem, which says that if you average enough unrelated things, you eventually get the Normal.

The Normal distribution in statistics is a special world in which the math is straightforward and all the parts fit together in a way that is easy to understand and interpret. It may not exactly match the real world, but it is close enough that this one simplifying assumption allows you to predict lots of things, and the predictions are often pretty reasonable. It is statistically convenient, and it is represented by basic statistics: the average, and the variance (or standard deviation) - the average of what's left when you take away the average, but to the power of 2.

That convenience has a price in testing. In a statistical test, you need the data to be normal to guarantee that your p-values are accurate with your given sample size.

If the data are not normal, your sample size may or may not be adequate, and it may be difficult for you to know which is true.


## Comparing distributions

Once the shape of one sample is known, the next question is whether two samples share it. The classic test is Kolmogorov–Smirnov, and the Wikipedia entry carries the author's warning in its link: [Kolmogorov Smirnov not good for categoricals.](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test) The general question, how to check two unknown distributions for equality and which statistical tests exist for it, is the Math Stack Exchange thread on [Comparing two](https://math.stackexchange.com/questions/159940/comparing-distribution-of-two-data-sets). For the intuition behind describing and comparing distributions, there is [Khan academy](https://www.khanacademy.org/math/ap-statistics/quantitative-data-ap/describing-comparing-distributions/v/comparing-distributions). To compare them [Visually](https://www.stat.auckland.ac.nz/~ihaka/787/lectures-distrib.pdf), the lecture notes linked there are the reference.

The earlier Gaussian section warned about data that are not normal, and the Quora question on which test quantifies similarity between two distributions covers exactly the case [When they are not normal](https://www.quora.com/Which-statistical-test-to-use-to-quantify-the-similarity-between-two-distributions-when-they-are-not-normal). In machine learning the two samples are often train and test, and Shikhar Gupta's "How (dis)similar are my train and test data?" is the [Using train / test trick](https://medium.com/data-science/how-dis-similar-are-my-train-and-test-data-56af3923de9b): comparing one mix of apples and oranges with another mix whose distribution is different. Finally, when the goal is to name the shape of real data rather than compare two samples, there is [Code for Identifying distribution type and params, based on best fit.](https://stackoverflow.com/questions/37487830/how-to-find-probability-distribution-and-parameters-for-real-data-python-3), a Stack Overflow question on finding the distribution type and parameters that a skewed, all-positive target most closely resembles.


## Comparing distributions (distance methods)

A test says whether two distributions differ; a distance says by how much, which is what drift monitoring needs. Categorical data can be transformed to a histogram i.e., #class / total and then measured for distance between two histograms’, e.g., train and production. Using earth mover distance [python](https://jeremykun.com/2018/03/05/earthmover-distance/) [git wrapper to c](https://github.com/pdinges/python-emd), linear programming, so its slow. The python link is the Earthmover Distance post, which starts from computing distance between points with uncertain locations; the git wrapper to c is a Python wrapper for Yossi Rubner's implementation of the earth mover's distance (EMD).

The same notes are in [Drift](../ai-engineering/mlops/mlops-monitoring-and-alerts.md#drift).

For the idea itself, [Earth movers](https://medium.com/data-science/earth-movers-distance-68fff0363ef2) is the Medium explanation of earth mover's distance, and the [EMD paper](http://infolab.stanford.edu/pub/cstr/reports/cs/tr/99/1620/CS-TR-99-1620.ch4.pdf) is the paper behind it. Also check KL DIVERGENCE in the information theory section.

The same notes are in [Cross entropy, relative ent, KL-D, JS-D, soft max](information-theory.md#cross-entropy-relative-ent-kl-d-js-d-soft-max).

Distances between distributions can also drive learning, not only monitoring. [Bengio](https://arxiv.org/abs/1901.10912) et al, transfer objective for learning to disentangle casual mechanisms - We propose to meta-learn causal structures based on how fast a learner adapts to new distributions arising from sparse distributional changes


## Deprecated links

{% hint style="warning" %}
These links and images no longer work. The original wording is kept here. A same-resource copy, when one was checked, is used above.
{% endhint %}

- “ if you collect data and it is not normal, “you need to collect more data”. This address no longer opens: https://www.isixsigma.com/topic/normal-distributions-why-does-it-matter/
- Using train / test trick. This address no longer opens: https://towardsdatascience.com/how-dis-similar-are-my-train-and-test-data-56af3923de9b
- Earth movers. This address no longer opens: https://towardsdatascience.com/earth-movers-distance-68fff0363ef2
