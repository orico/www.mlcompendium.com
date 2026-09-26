# Probability & Statistics

Probability asks what happens next under a random process; statistics asks what process would explain what already happened. The page starts from that difference, then walks introductions to probability and statistics, more statistics, sampling and resampling, wiki topics, and recommended courses.
The same notes are in [Probability](probability.md).

The best single starting point is a [Coursera course](https://www.youtube.com/watch?v=WkOinijQmPU&list=PLpl-gQkQivXiBmGyzLrUjzsblmQsLtkzJ&index=1) on probabilities - for data science, actually quite good in explaining a lot of the basic tools,prob, conditional, distributions, sampling, CI, hypothesis, etc. A great resource for proba/bayes/b-networks/etc (adam bali) used to sit beside it; that address no longer opens and is kept at the end of the page.

### Difference between

Before any tool, it helps to see that the two fields invert the same uncertainty problem. The Stack Exchange question [Difference between](https://stats.stackexchange.com/questions/665/whats-the-difference-between-probability-and-statistics) asks what separates probability from statistics and why they are studied together. I.e, Probability deals with predicting the likelihood of future events, while statistics involves the analysis of the frequency of past events. The problems considered by probability and statistics are inverse to each other.

In probability theory we consider some underlying process which has some randomness or uncertainty modeled by random variables, and we figure out what happens.

 => Underlying process + randomness and random variables -> what happens next?

In statistics we observe something that has happened, and try to figure out what underlying process would explain those observations.

 => observe what happened -> what is the underlying process?

Finally, probability theory is mainly concerned with the deductive part, statistics with the inductive part of modeling processes with uncertainty.

### Introduction to Probability

The deductive side comes first, and Math is Fun walks it one small page at a time, starting from the observation that life is full of random events. [Types of events](https://www.mathsisfun.com/data/probability-events-types.html) names them, [Independent events](https://www.mathsisfun.com/data/probability-events-independent.html) covers events that do not affect each other, and [Conditional proba](https://www.mathsisfun.com/data/probability-events-conditional.html) is how to handle dependent events. Calculating probabilities can be hard, sometimes we add them, sometimes we multiply them, and often it is hard to figure out what to do, which is why [Proba tree diagrams](https://www.mathsisfun.com/data/probability-tree-diagrams.html) come next. [Mutually exclusive events](https://www.mathsisfun.com/data/probability-events-mutually-exclusive.html) are the ones that cannot happen at the same time, like turning left and turning right, or heads and tails.

Counting comes before most of those calculations. In English we use the word combination loosely, without thinking if the order of things is important, and [Combination and permutations](https://www.mathsisfun.com/combinatorics/combinations-permutations.html) makes that distinction exact. [Bayes](https://www.mathsisfun.com/data/bayes-theorem.html) is the Math is Fun page on Bayes' theorem. [Least squares regresssion](https://www.mathsisfun.com/data/least-squares-regression.html) It works by making the total of the square of the errors as small as possible (that is why it is called "least squares".

The sequence ends with random variables. A Random Variable is a set of possible values from a random experiment, such as heads and tails given the values 0 and 1. [Random variables](https://www.mathsisfun.com/data/random-variables.html) introduces them, [Continuous random variables](https://www.mathsisfun.com/data/random-variables-continuous.html) extends them past discrete outcomes, and [Random vars mean, std, variance](https://www.mathsisfun.com/data/random-variables-mean-variance.html) gives their mean, variance, and standard deviation.

### Introduction to statistics

With random variables in hand, the inductive side starts from data itself. Data is a collection of facts and numbers, and the Math is Fun [Table of](https://www.mathsisfun.com/data/index.html#stats) contents for using and handling data is the index for everything below.

The first statistics are centers. The Median is the middle of a sorted list of numbers ([Median](https://www.mathsisfun.com/median.html)), and the mode is the number which appears most often ([Mode](https://www.mathsisfun.com/mode.html)). The mean has several forms: the [Weighted mean](https://www.mathsisfun.com/data/weighted-mean.html); the [Geometric mean](https://www.mathsisfun.com/numbers/geometric-mean.html), a special type of average where we multiply the numbers together and then take a square root (for two numbers), cube root, and so on; and the [Harmonic mean](https://www.mathsisfun.com/numbers/harmonic-mean.html), the reciprocal of the average of the reciprocals.

Centers then need spread and position. Percentile is the value below which a percentage of data falls ([Percentiles](https://www.mathsisfun.com/data/percentiles.html)). [Mean deviation](https://www.mathsisfun.com/data/mean-deviation.html) is how far, on average, all values are from the middle. When two sets of data are strongly linked together we say they have a High Correlation ([Correlation](https://www.mathsisfun.com/data/correlation.html)). Deviation means how far from the normal, and the [Standard deviation](https://www.mathsisfun.com/data/standard-deviation.html) is a measure of how spread out numbers are; the [formula](https://www.mathsisfun.com/data/standard-deviation-formulas.html) page gives the calculation, deviation meaning how far from the average.

Spread leads to shape. Data can be distributed (spread out) in different ways, and in many cases it tends to sit around a central value, which is the [Standard normal distribution](https://www.mathsisfun.com/data/standard-normal-distribution.html). Data can be skewed, meaning it tends to have a long tail on one side or the other, and [Skewness of distribution](https://www.mathsisfun.com/data/skewness.html) explains why it is called negative skew.

Shape then supports inference. A confidence interval is a range of values we are fairly sure our true value lies in: [Confidence intervals (using std)](https://www.mathsisfun.com/data/confidence-interval.html). Accuracy is how close a measured value is to the true value and precision is how close the measurements are to each other, the [Accuracy vs precision (accurate vs hitting closely or density)](https://www.mathsisfun.com/accuracy-precision.html) page. [Probability](https://www.mathsisfun.com/data/probability.html) is how likely something is to happen, and the complement of an event is all outcomes that are NOT the event ([Probability complement](https://www.mathsisfun.com/data/probability-complement.html)). Is that a true difference, or just a fluke? The [Chi-square test, p_value, independent, dependent, significance](https://www.mathsisfun.com/data/chi-square-test.html) page answers that by telling a real relationship from random chance.

Two terms get confused often enough to need their own links. [Variation vs variance](https://stats.stackexchange.com/questions/88348/is-variation-the-same-as-variance) - a private case - is the Stack Exchange question on whether variation and variance are the same term. [Std vs variance](https://www.investopedia.com/ask/answers/021215/what-difference-between-standard-deviation-and-variance.asp) - std is in the same metric as the mean, is the root of variance., allows outliers to influence, will not result in samples cancelling each other without the square root in the formula.

### More on Statistics

After the introductions, the same concepts come back in plain English as a Data Science Central series, in three parts. The first is [25 concepts](https://www.datasciencecentral.com/profiles/blogs/25-statistical-concepts-explained-in-simple-english-part-2), then (part 2) [29 more concepts](https://www.datasciencecentral.com/profiles/blogs/29-statistical-concepts-explained-in-simple-english-part-1), then (part1) & [part 3](https://www.datasciencecentral.com/profiles/blogs/29-statistical-concepts-explained-in-simple-english-part-2?fbclid=IwAR0VQFeBaJsm3ouEf7sV5WAupE1cI3PXhzWe9-lUYkZ_XCCF72_3r8w5hrI).

### STATISTICAL SAMPLING AND RESAMPLING

Every statistic above is computed on a sample, so the next question is how the sample was drawn and how wrong it can be.

The same notes are in [TRAIN / TEST / CROSS VALIDATION](datasets.md#train--test--cross-validation).

MachineLearningMastery's "A Gentle Introduction to Statistical Sampling and Resampling" answers [What is? Method for sampling/resampling, and sampling errors explained.](https://machinelearningmastery.com/statistical-sampling-and-resampling/)

Another great course on probability, distribution types, conditional, joint, chain, etc. [http://legacydirs.umiacs.umd.edu/~jbg/teaching/INST_414/](http://legacydirs.umiacs.umd.edu/~jbg/teaching/INST_414/)

### Wiki

Sampling from more than one variable at once needs the joint vocabulary, and the Wikipedia entries give the definitions. [Marginal probability](https://en.wikipedia.org/wiki/Marginal_distribution) is the marginal distribution entry, [Joint probability](https://en.wikipedia.org/wiki/Joint_probability_distribution) is the joint probability distribution entry, and [Conditional probability](https://en.wikipedia.org/wiki/Probability) points at the main Probability entry. The chain rule has two meanings worth separating: the probability [Chain rule](https://en.wikipedia.org/wiki/Chain_rule_(probability) on Wikipedia, and derivatives using the chain rule, on [khan](https://www.khanacademy.org/math/ap-calculus-ab/ab-differentiation-2-new/ab-3-1a/v/chain-rule-introduction).

### Recommended Courses

The shelf closes with courses and one friendly article. Another great course on probability, distribution types, conditional, joint, chain, etc. is the course listed under sampling above. [Kahn](https://www.khanacademy.org/math/precalculus/prob-comb) is the probability course on the same site as the khan chain-rule video. Brandon Rohrer's "How Bayes Theorem works" is [A really good intro](https://www.youtube.com/watch?v=5NMxiOGL39M). [What are confidence intervals?](https://medium.com/data-science/a-very-friendly-introduction-to-confidence-intervals-9add126e714) is a very friendly introduction that discusses only the general idea, without too much fancy statistics terms, and with python.

(another angle) [The main difference between probability and statistics has to do with knowledge](https://www.thoughtco.com/probability-vs-statistics-3126368): what are the known facts? Inherent in both probability and statistics is a [population](https://www.thoughtco.com/what-is-a-population-in-statistics-3126308), every individual we are interested in studying, and a sample, consisting of the individuals that are selected from the population. In probability, we would start with us knowing everything about the composition of a population, and then would ask, “What is the likelihood that a selection, or sample, from the population, has certain characteristics?” In statistics, we have no knowledge about the types of socks in the drawer; we infer properties about the population on the basis of a random sample.

Some [calculations](https://www.mathsisfun.com/data/probability.html) to get you into probability:

- Finding out the probability of an event
- Of two consecutive events (multiplication)
- Of several events (sum)
- Etc..

## Deprecated links

{% hint style="warning" %}
These links and images no longer work. The original wording is kept here. A same-resource copy, when one was checked, is used above.
{% endhint %}

- A great resource for proba/bayes/b-networks/etc (adam bali). This address no longer opens: https://metacademy.org/graphs/concepts/bayesian_networks#focus=i9mo2e09&mode=learn
- What are confidence intervals? This address no longer opens: https://towardsdatascience.com/a-very-friendly-introduction-to-confidence-intervals-9add126e714
