# Recommender Systems

This page collects collaborative filtering, content-based, and matrix-factorization notes, then evaluation and tools.

1. [Beginner guide](https://www.analyticsvidhya.com/blog/2015/08/beginners-guide-learn-content-based-recommender-systems/) vidhya
2. [Real python on CF](https://realpython.com/build-recommendation-engine-collaborative-filtering/#steps-involved-in-collaborative-filtering)
3. [Intro to, using item-item or user-item](https://www.ethanrosenthal.com/2015/11/02/intro-to-collaborative-filtering/), validating using imdb data, git
4. Tfidf cosine similarity; [countvec cosine](https://www.datacamp.com/community/tutorials/recommender-systems-python)
5. Various implementations of CF; a serious review of algorithms
6. [Collaborative filtering, SVD](https://hackernoon.com/introduction-to-recommender-system-part-1-collaborative-filtering-singular-value-decomposition-44c9659c5e75)
7. [Part1,](https://hackernoon.com/introduction-to-recommender-system-part-1-collaborative-filtering-singular-value-decomposition-44c9659c5e75) Spotlight, item2vec, Neural nets for Recommender systems
8. [A general tutorial, has a nice intro](https://www.datacamp.com/community/tutorials/recommender-systems-python)
9. Medium on Movies
   1. Part 1 matrix factorization in movies, users vs movies
   2. Part 2 using collaborative filtering using open ai
   3. Part 3 using col-filtering with neural nets
10. Medium series on collaborative filtering and embeddings Part 1, part 2; [git](https://github.com/shik3519/collaborative-filtering)
11. [Movie recommender systems](https://www.kaggle.com/rounakbanik/movie-recommender-systems) on kaggle
    1. [On git](https://github.com/jaypatel00174/Movie-Recommendation)
12. Matrix factorization
13. [Collaborative filtering with binary countvec data, item-item, didnt work well on another domain](https://medium.com/radon-dev/item-item-collaborative-filtering-with-binary-or-unary-data-e8f0b465b2c3)
14. Netflix competition, matrix factorization over classical algorithms, a survey paper
15. Movie similarity based on genre
16. [Similar entities, matrix multiplication](https://medium.com/wbaa/https-medium-com-ingwbaa-boosting-selection-of-the-most-similar-entities-in-large-scale-datasets-450b3242e618) high sparsity
17. [Euclidean distance with high sparse data](https://stats.stackexchange.com/questions/117354/euclidean-distance-with-sparse-and-high-dimension-data)
18. Excel & fastai, [git](https://github.com/shik3519/collaborative-filtering/blob/master/cf-scratch-movielens/collaborative%20filtering%20from%20scratch.ipynb)
19. [CF for movie recommendation](https://medium.com/@wwwbbb8510/python-implementation-of-baseline-item-based-collaborative-filtering-2ba7c8960590)
20. [Comparison item vs user cf](https://medium.com/@wwwbbb8510/comparison-of-user-based-and-item-based-collaborative-filtering-f58a1c8a3f1d)
21. [build a recommendation engine with collaborative filtering](https://realpython.com/build-recommendation-engine-collaborative-filtering/)

## Evaluating Recommender Systems

This section lists evaluation methods, business choice notes, and Microsoft’s accuracy-metrics survey.

1. An exhaustive list of methods to evaluate
2. [Choosing the best for your business](https://medium.com/recombee-blog/evaluating-recommender-systems-choosing-the-best-one-for-your-business-c688ab781a35)
3. [Evaluating](https://medium.com/the-owl/evaluating-recommender-systems-749570354976)
4. [survey of accuracy eval metrics for RS by Microsoft](https://www.jmlr.org/papers/volume10/gunawardana09a/gunawardana09a.pdf)
5. [Building a validation framework](https://medium.com/moosend-engineering-data-science/building-a-validation-framework-for-recommender-systems-a-quest-ec173a24b56f)
6. Evaluation Metrics for RS
7. [offline vs online validation](https://www.quora.com/How-do-I-validate-my-recommendation-system-without-prior-user-interaction-data)
8. [Evaluating RS](https://tzin.bgu.ac.il/~shanigu/Publications/EvaluationMetrics.17.pdf)

## TOOLS

This section lists Surprise, Grover Prince’s repo, and python-recsys.

1. [Surprise](https://github.com/NicolasHug/Surprise), [docs](https://surprise.readthedocs.io/en/stable/FAQ.html#how-to-get-the-top-n-recommendations-for-each-user),
2. [Grover prince](https://github.com/groverpr/Machine-Learning), related article
3. [Recsys](https://github.com/ocelma/python-recsys) git

## Deprecated links

{% hint style="warning" %}
These links and images no longer work. The original wording is kept here. A same-resource copy, when one was checked, is used above.
{% endhint %}

- Tfidf cosine similarity. This address no longer opens: https://towardsdatascience.com/recommender-engine-under-the-hood-7869d5eab072
- Various implementations of CF. This address no longer opens: https://towardsdatascience.com/various-implementations-of-collaborative-filtering-100385c6dfe0
- Spotlight, item2vec, Neural nets for Recommender systems. This address no longer opens: https://towardsdatascience.com/introduction-to-recommender-system-part-2-adoption-of-neural-network-831972c4cbf7
- Part 1 matrix factorization in movies, users vs movies. This address no longer opens: https://towardsdatascience.com/fast-ai-season-1-episode-5-1-movie-recommendation-using-fastai-a53ed8e41269
- Part 2 using collaborative filtering. This address no longer opens: https://towardsdatascience.com/fast-ai-season-1-episode-5-2-collaborative-filtering-from-scratch-1877640f514a
- Part 3 using col-filtering with neural nets. This address no longer opens: https://towardsdatascience.com/fast-ai-season-1-episode-5-3-collaborative-filtering-using-neural-network-48e49d7f9b36
- Medium series on collaborative filtering and embeddings Part 1. This address no longer opens: https://towardsdatascience.com/collaborative-filtering-and-embeddings-part-1-63b00b9739ce
- part 2. This address no longer opens: https://towardsdatascience.com/collaborative-filtering-and-embeddings-part-2-919da17ecefb
- Matrix factorization. This address no longer opens: https://towardsdatascience.com/paper-summary-matrix-factorization-techniques-for-recommender-systems-82d1a7ace74
- Netflix competition, matrix factorization over classical algorithms, a survey paper. This address no longer opens: https://towardsdatascience.com/paper-summary-matrix-factorization-techniques-for-recommender-systems-82d1a7ace74
- Movie similarity based on genre. This address no longer opens: https://towardsdatascience.com/content-based-recommender-systems-28a1dbd858f5
- An exhaustive list of methods to evaluate. This address no longer opens: https://towardsdatascience.com/an-exhaustive-list-of-methods-to-evaluate-recommender-systems-a70c05e121de
- Evaluation Metrics for RS. This address no longer opens: https://towardsdatascience.com/evaluation-metrics-for-recommender-systems-df56c6611093
- related article. This address no longer opens: https://towardsdatascience.com/various-implementations-of-collaborative-filtering-100385c6dfe0
