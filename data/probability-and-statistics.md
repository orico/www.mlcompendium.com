# Probability & Statistics

Probability asks what happens next under a random process; statistics asks what process would explain what already happened.
This page starts from that difference, then walks introductions to probability and statistics, more statistics, sampling and resampling, wiki topics, and recommended courses.
The same notes are in [Probability](probability.md).

[Coursera course](https://www.youtube.com/watch?v=WkOinijQmPU&list=PLpl-gQkQivXiBmGyzLrUjzsblmQsLtkzJ&index=1) on probabilities - for data science, actually quite good in explaining a lot of the basic tools,prob, conditional, distributions, sampling, CI, hypothesis, etc.
- A great resource for proba/bayes/b-networks/etc (adam bali)

### Difference between

This section states how probability and statistics invert the same uncertainty problem.

- [Difference between](https://stats.stackexchange.com/questions/665/whats-the-difference-between-probability-and-statistics)
- I.e, Probability deals with predicting the likelihood of future events, while statistics involves the analysis of the frequency of past events.
- The problems considered by probability and statistics are inverse to each other.
- In probability theory we consider some underlying process which has some randomness or uncertainty modeled by random variables, and we figure out what happens.

 => Underlying process + randomness and random variables -> what happens next?

- In statistics we observe something that has happened, and try to figure out what underlying process would explain those observations.

 => observe what happened -> what is the underlying process?

- Finally, probability theory is mainly concerned with the deductive part, statistics with the inductive part of modeling processes with uncertainty

### Introduction to Probability

After the difference, this section links Math is Fun probability topics from events through random variables.

- Probability: Types of Events. Probability: Types of Events. [Types of events](https://www.mathsisfun.com/data/probability-events-types.html)
- Probability: Independent Events. Probability: Independent Events. [Independent events](https://www.mathsisfun.com/data/probability-events-independent.html)
- Conditional Probability. Conditional Probability. [Conditional proba](https://www.mathsisfun.com/data/probability-events-conditional.html)
- Calculating probabilities can be hard, sometimes we add them, sometimes we multiply them, and often it is hard to figure out what to do ... [Proba tree diagrams](https://www.mathsisfun.com/data/probability-tree-diagrams.html)
- The page covers mutually Exclusive Events. [Mutually exclusive events](https://www.mathsisfun.com/data/probability-events-mutually-exclusive.html)
- In English we use the word combination loosely, without thinking if the order of things is important. [Combination and permutations](https://www.mathsisfun.com/combinatorics/combinations-permutations.html)
- The page covers bayes. The page covers bayes. [Bayes](https://www.mathsisfun.com/data/bayes-theorem.html)
8. [Least squares regresssion](https://www.mathsisfun.com/data/least-squares-regression.html) It works by making the total of the square of the errors as small as possible (that is why it is called "least squares"
- A Random Variable is a set of possible values from a random experiment. [Random variables](https://www.mathsisfun.com/data/random-variables.html)
- A Random Variable is a set of possible values from a random experiment. [Continuous random variables](https://www.mathsisfun.com/data/random-variables-continuous.html)
- A Random Variable is a set of possible values from a random experiment. [Random vars mean, std, variance](https://www.mathsisfun.com/data/random-variables-mean-variance.html)

### Introduction to statistics

Beside probability, this section lists Math is Fun pages on descriptive statistics and related basics.

- Data is a collection of facts and numbers. [Table of](https://www.mathsisfun.com/data/index.html#stats)
- The Median is the middle of a sorted list of numbers. [Median](https://www.mathsisfun.com/median.html)
- The mode is the number which appears most often. [Mode](https://www.mathsisfun.com/mode.html)
- Math explained in easy language, plus puzzles, games, quizzes, worksheets and a forum. [Weighted mean](https://www.mathsisfun.com/data/weighted-mean.html)
- The Geometric Mean is a special type of average where we multiply the numbers together and then take a square root (for two numbers), cube root... [Geometric mean](https://www.mathsisfun.com/numbers/geometric-mean.html)
- The harmonic mean is: the reciprocal of the average of the reciprocals. [Harmonic mean](https://www.mathsisfun.com/numbers/harmonic-mean.html)
- Percentile is the value below which a percentage of data falls. [Percentiles](https://www.mathsisfun.com/data/percentiles.html)
- How far, on average, all values are from the middle. [Mean deviation](https://www.mathsisfun.com/data/mean-deviation.html)
- When two sets of data are strongly linked together we say they have a High Correlation. [Correlation](https://www.mathsisfun.com/data/correlation.html)
- Deviation means how far from the normal. [Standard deviation](https://www.mathsisfun.com/data/standard-deviation.html)
- Deviation means how far from the average. [formula](https://www.mathsisfun.com/data/standard-deviation-formulas.html)
- Data can be distributed (spread out) in different ways. [Standard normal distribution](https://www.mathsisfun.com/data/standard-normal-distribution.html)
- Data can be skewed, meaning it tends to have a long tail on one side or the other: Why is it called negative skew? [Skewness of distribution](https://www.mathsisfun.com/data/skewness.html)
- Confidence Intervals. Confidence Intervals. [Confidence intervals (using std)](https://www.mathsisfun.com/data/confidence-interval.html)
- Accuracy and Precision. Accuracy and Precision. [Accuracy vs precision (accurate vs hitting closely or density)](https://www.mathsisfun.com/accuracy-precision.html)
- The page covers probability. The page covers probability. [Probability](https://www.mathsisfun.com/data/probability.html)
- Complement of an Event: All outcomes that are NOT the event. [Probability complement](https://www.mathsisfun.com/data/probability-complement.html)
- Is that a true difference, or just a fluke? [Chi-square test, p_value, independent, dependent, significance](https://www.mathsisfun.com/data/chi-square-test.html)
18. [Variation vs variance](https://stats.stackexchange.com/questions/88348/is-variation-the-same-as-variance) - a private case
19. [Std vs variance](https://www.investopedia.com/ask/answers/021215/what-difference-between-standard-deviation-and-variance.asp) - std is in the same metric as the mean, is the root of variance., allows outliers to influence, will not result in samples cancelling each other without the square root in the formula.

### More on Statistics

After the introductions, this section points at Data Science Central article series on statistical concepts.

- TechTarget - Global Network of Information Technology Websites and Contributors. [25 concepts](https://www.datasciencecentral.com/profiles/blogs/25-statistical-concepts-explained-in-simple-english-part-2)
- TechTarget - Global Network of Information Technology Websites and Contributors. (part 2) [29 more concepts](https://www.datasciencecentral.com/profiles/blogs/29-statistical-concepts-explained-in-simple-english-part-1)
- TechTarget - Global Network of Information Technology Websites and Contributors. (part1) & [part 3](https://www.datasciencecentral.com/profiles/blogs/29-statistical-concepts-explained-in-simple-english-part-2?fbclid=IwAR0VQFeBaJsm3ouEf7sV5WAupE1cI3PXhzWe9-lUYkZ_XCCF72_3r8w5hrI)

### STATISTICAL SAMPLING AND RESAMPLING

With the concepts named, this section links an overview of sampling, resampling, and sampling error.

The same notes are in [TRAIN / TEST / CROSS VALIDATION](datasets.md#train--test--cross-validation).

- A Gentle Introduction to Statistical Sampling and Resampling - MachineLearningMastery.com. [What is? Method for sampling/resampling, and sampling errors explained.](https://machinelearningmastery.com/statistical-sampling-and-resampling/)

- Another great course on probability, distribution types, conditional, joint, chain, etc. [http://legacydirs.umiacs.umd.edu/~jbg/teaching/INST_414/](http://legacydirs.umiacs.umd.edu/~jbg/teaching/INST_414/)

### Wiki

After sampling, this section lists Wikipedia entries on marginal, joint, and conditional probability.

- Marginal distribution. Marginal distribution - Wikipedia. [Marginal probability](https://en.wikipedia.org/wiki/Marginal_distribution)
- Joint probability distribution. Joint probability distribution - Wikipedia. [Joint probability](https://en.wikipedia.org/wiki/Joint_probability_distribution)
- Probability. Probability - Wikipedia. [Conditional probability](https://en.wikipedia.org/wiki/Probability)
- Chain rule (probability. Chain rule (probability - Wikipedia. [Chain rule](https://en.wikipedia.org/wiki/Chain_rule_(probability)
- ) - derivatives using the chain rule, on. [khan](https://www.khanacademy.org/math/ap-calculus-ab/ab-differentiation-2-new/ab-3-1a/v/chain-rule-introduction)

### Recommended Courses

Closing the shelf, this section recommends probability courses and a friendly confidence-interval article.

- Another great course on probability, distribution types, conditional, joint, chain, etc.
- Client Challenge. Client Challenge. [Kahn](https://www.khanacademy.org/math/precalculus/prob-comb)
- How Bayes Theorem works, by Brandon Rohrer. [A really good intro](https://www.youtube.com/watch?v=5NMxiOGL39M)
- [What are confidence intervals?](https://medium.com/data-science/a-very-friendly-introduction-to-confidence-intervals-9add126e714)

(another angle) [The main difference between probability and statistics has to do with knowledge](https://www.thoughtco.com/probability-vs-statistics-3126368)

- what are the known facts? Inherent in both probability and statistics is a [population](https://www.thoughtco.com/what-is-a-population-in-statistics-3126308),
- every individual we are interested in studying, and a sample, consisting of the individuals that are selected from the population.
- in probability: would start with us knowing everything about the composition of a population, and then would ask, “What is the likelihood that a selection, or sample, from the population, has certain characteristics?”
- In statistics: we have no knowledge about the types of socks in the drawer. we infer properties about the population on the basis of a random sample.

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
