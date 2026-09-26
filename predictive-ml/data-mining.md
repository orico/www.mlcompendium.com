# Data Mining

Before any label exists, a transaction table can already say which items tend to show up together. This page answers that with association rules: first the slides, terms, and textbook chapter that define the problem, then Apriori and its Python implementations, then FP-Growth, and finally how to judge a rule with support, confidence, and lift.


### ASSOCIATION RULES

Association analysis starts from market basket data, where each row is one transaction. The same notes are in [Log Parsing / Templatization](templatization.md).

The [Association rules slides](https://www.slideshare.net/wanaezwani/apriori-and-eclat-algorithm-in-association-rule-mining) walk through the Apriori and Eclat algorithms in association rule mining. The [Terms](https://www.kdnuggets.com/2016/04/association-rules-apriori-algorithm-tutorial.html) are the KDnuggets tutorial on association rules and the Apriori algorithm. The [Paper — basic concepts and algo](https://www-users.cs.umn.edu/~kumar001/dmbook/ch5_association_analysis.pdf) is the textbook chapter on association analysis, which starts from the customer purchase data that grocery checkouts collect every day and the market basket transactions table built from it.

A four-part Knoldus series used to follow the same arc, and its original blog links are in Deprecated links. In order, it covered:

1. Apriori
2. Association rules
3. Fp-growth
4. Fp-tree construction

**APRIORI**

Apriori is the first algorithm to learn, because it finds frequent item sets by growing them one item at a time. The [Apyori tut](https://stackabuse.com/association-rule-mining-via-apriori-algorithm-in-python/) frames association rule mining as a technique to identify underlying relations between different items, using a supermarket as the example, and the matching library is [git](https://github.com/ymoch/apyori), a simple implementation of Apriori in Python. When that is too slow, [Efficient apriori](https://github.com/tommyod/Efficient-Apriori) is an efficient Python implementation of the same algorithm.

For the idea behind it, MachineLearningMastery's market basket analysis with association rule learning covers [One of the best known association rules algorithm](https://machinelearningmastery.com/market-basket-analysis-with-association-rule-learning/). [A very good visual example of a transaction DB with the apriori algorithm step by step](http://www.lessons2all.com/Apriori.php) is Vishwanath Pai's Lessons2all example of how to find a frequent item set in a transaction database, with a detailed solution. A Python 3.0 code walkthrough used to sit here; that address now points elsewhere and is kept at the end of the page.

In practice, [Mlxtnd](http://rasbt.github.io/mlxtend/api_subpackages/mlxtend.frequent_patterns/) is Sebastian Raschka's mlxtend frequent patterns module, and the GeeksforGeeks [tutorial](https://www.geeksforgeeks.org/implementing-apriori-algorithm-in-python/) implements Apriori in Python. The mlxtend module exposes four pieces:

 1. Apriori
 2. Rules
 3. pgrowth
 4. fpmax

**FP Growth**

Apriori rescans the data for every candidate set, so FP-Growth builds a tree once and grows patterns from it. Well Academy's video on data mining FP-growth shows [How to construct the fp-tree](https://www.youtube.com/watch?v=gq6nKbye648) with an FP-tree example. The rest of the author's list of sources:

2. The same example, but with a graph that shows that lower support cost less for fp-growth in terms of calc time.
3. Coursera video.
4. Another clip video
5. [How to validate these algorithms](https://stackoverflow.com/questions/32843093/how-to-validate-association-rules) — probably the best way is confidence/support/lift

Validation comes back to those three measures. It depends on your task. But usually you want all [three to be high.](https://stats.stackexchange.com/questions/229523/association-rules-support-confidence-and-lift)

- high support: should apply to a large amount of cases
- high confidence: should be correct often
- high lift: indicates it is not just a coincidence

Once both algorithms are clear, the last question is whether they give different answers:

1. [Difference between apriori and fp-growth](https://www.quora.com/What-is-the-difference-between-FPgrowth-and-Apriori-algorithms-in-terms-of-results) explains that both are frequent pattern mining algorithms, but they differ in how they represent, generate, and output frequent itemsets, which leads to practical differences in results, performance, and usability.

## Deprecated links

{% hint style="warning" %}
These links and images no longer work. The original wording is kept here. A same-resource copy, when one was checked, is used above.
{% endhint %}

- Apriori. This address no longer opens: https://blog.knoldus.com/machinex-why-no-one-uses-apriori-algorithm-for-association-rule-learning/
- Association rules. This address no longer opens: https://blog.knoldus.com/machinex-two-parts-of-association-rule-learning/
- Fp-growth. This address no longer opens: https://blog.knoldus.com/machinex-frequent-itemset-generation-with-the-fp-growth-algorithm/
- Fp-tree construction. This address no longer opens: https://blog.knoldus.com/machinex-understanding-fp-tree-construction/
- Coursera video. This address no longer opens: https://www.coursera.org/learn/data-patterns/lecture/ugqCs/2-5-fpgrowth-a-pattern-growth-approach
- Python 3.0 code. This address now points to an unrelated site: http://adataanalyst.com/machine-learning/apriori-algorithm-python-3-0/
