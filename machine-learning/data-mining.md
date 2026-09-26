# Data Mining

This page is about association rules: Apriori, FP-Growth, and how to read support, confidence, and lift.

### ASSOCIATION RULES

This section lists slides, terms, and a textbook chapter on association analysis.

1. [Association rules slides](https://www.slideshare.net/wanaezwani/apriori-and-eclat-algorithm-in-association-rule-mining) — apriori, eclat, fp growth — pretty complete
2. [Terms](https://www.kdnuggets.com/2016/04/association-rules-apriori-algorithm-tutorial.html) — lift, confidence
3. [Paper — basic concepts and algo](https://www-users.cs.umn.edu/~kumar001/dmbook/ch5_association_analysis.pdf)

Knoldus (original blog links are in Deprecated links):

1. Apriori
2. Association rules
3. Fp-growth
4. Fp-tree construction

**APRIORI**

1. [Apyori tut](https://stackabuse.com/association-rule-mining-via-apriori-algorithm-in-python/) [git](https://github.com/ymoch/apyori)
2. [Efficient apriori](https://github.com/tommyod/Efficient-Apriori)
3. [One of the best known association rules algorithm](https://machinelearningmastery.com/market-basket-analysis-with-association-rule-learning/) — apriori in weka
4. [A very good visual example of a transaction DB with the apriori algorithm step by step](http://www.lessons2all.com/Apriori.php)
5. [Python 3.0 code](http://adataanalyst.com/machine-learning/apriori-algorithm-python-3-0/)
6. [Mlxtnd](http://rasbt.github.io/mlxtend/api_subpackages/mlxtend.frequent_patterns/) [tutorial](https://www.geeksforgeeks.org/implementing-apriori-algorithm-in-python/)
 1. Apriori
 2. Rules
 3. pgrowth
 4. fpmax

**FP Growth**

1. [How to construct the fp-tree](https://www.youtube.com/watch?v=gq6nKbye648)
2. The same example, but with a graph that shows that lower support cost less for fp-growth in terms of calc time.
3. Coursera video.
4. Another clip video
5. [How to validate these algorithms](https://stackoverflow.com/questions/32843093/how-to-validate-association-rules) — probably the best way is confidence/support/lift

It depends on your task. But usually you want all [three to be high.](https://stats.stackexchange.com/questions/229523/association-rules-support-confidence-and-lift)

- high support: should apply to a large amount of cases
- high confidence: should be correct often
- high lift: indicates it is not just a coincidence

1. [Difference between apriori and fp-growth](https://www.quora.com/What-is-the-difference-between-FPgrowth-and-Apriori-algorithms-in-terms-of-results)

## Deprecated links

{% hint style="warning" %}
These links and images no longer work. The original wording is kept here. A same-resource copy, when one was checked, is used above.
{% endhint %}

- Apriori. This address no longer opens: https://blog.knoldus.com/machinex-why-no-one-uses-apriori-algorithm-for-association-rule-learning/
- Association rules. This address no longer opens: https://blog.knoldus.com/machinex-two-parts-of-association-rule-learning/
- Fp-growth. This address no longer opens: https://blog.knoldus.com/machinex-frequent-itemset-generation-with-the-fp-growth-algorithm/
- Fp-tree construction. This address no longer opens: https://blog.knoldus.com/machinex-understanding-fp-tree-construction/
- Coursera video. This address no longer opens: https://www.coursera.org/learn/data-patterns/lecture/ugqCs/2-5-fpgrowth-a-pattern-growth-approach
