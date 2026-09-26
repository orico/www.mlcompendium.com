### **FEATURE ENGINEERING**

This section links guides on encoding, text, and embedding features.

The same notes are in [Auto Feature Engineering](../deep-learning/meta-learning.md#auto-feature-engineering), [Box Cox](distribution-transformation.md#box-cox), and [Feature engineering](engineering/lakes-and-warehouses.md#feature-engineering).

1. [**Vidhya on FE, anomalies, engineering, imputing**](https://www.analyticsvidhya.com/blog/2016/01/guide-data-exploration/?utm_source=outlierdetectionpyod&utm_medium=blog)
2. [**Many types of FE**](https://medium.com/data-science/understanding-feature-engineering-part-1-continuous-numeric-data-da4e47099a7b)**, including log and box cox transform - a very useful explanation.**
3. [**Categorical Data**](https://medium.com/data-science/understanding-feature-engineering-part-2-categorical-data-f54324193e63)
4. [**Dummy variables and feature hashing**](https://medium.com/data-science/understanding-feature-engineering-part-2-categorical-data-f54324193e63) **- hashing is really cool.**
5. [**Text data**](https://medium.com/data-science/understanding-feature-engineering-part-3-traditional-methods-for-text-data-f6f7d70acd41) **- unigrams, bag of words, N-grams (2,3,..), tfidf matrix, cosine\_similarity(tfidf) ontop of a tfidf matrix, unsupervised hierarchical clustering with similarity measures on top of (cosine\_similarity), LDA for topic modelling in sklearn - pretty awesome, Kmeans(lda),.**
6. [**Deep learning data for FE**](https://medium.com/data-science/understanding-feature-engineering-part-4-deep-learning-methods-for-text-data-96c44370bbfa)  **-** [**Word embedding using keras, continuous BOW - CBOW, SKIPGRAM, word2vec - really good.**](https://medium.com/data-science/understanding-feature-engineering-part-4-deep-learning-methods-for-text-data-96c44370bbfa)
7. [**Topic Modelling**](http://chdoig.github.io/pygotham-topic-modeling/#/) **- a fantastic slide show about topic modelling using LDA etc.**

The same notes are in [LDA (Latent Dirichlet Allocation)](../language-ai/topics-modeling.md#lda-latent-dirichlet-allocation).

8. **Dipanjan on feature engineering** [**1**](https://medium.com/data-science/understanding-feature-engineering-part-1-continuous-numeric-data-da4e47099a7b) **- cont numeric** [ **2 -**](https://medium.com/data-science/understanding-feature-engineering-part-2-categorical-data-f54324193e63) **categorical** [**3**](https://medium.com/data-science/understanding-feature-engineering-part-3-traditional-methods-for-text-data-f6f7d70acd41) **- traditional methods**
9. [**Target encoding git**](https://pypi.org/project/target_encoding/)
10. [**Category encoding git**](https://pypi.org/project/category-encoders/)

### **REPRESENTATION LEARNING**

This subsection links a representation learning paper.

1. [**paper**](https://arxiv.org/abs/1807.03748?utm_campaign=The%20Batch&utm_source=hs_email&utm_medium=email&utm_content=83602348&_hsenc=p2ANqtz-8DcgUdDF--k3tWOmhM51lm28wHerZxlXKoGNU6hIu2P4Fj-RuEuciKtbWZZdWmvBg7KGeI44FWUrmpHdZIbAM-pUicgg&_hsmi=83602348)

### **SIMILARITY**

This section covers vector and text similarity measures.

The same notes are in [Recommender Systems](../ai-product/recommender-systems.md) and [String Matching](../language-ai/string-matching.md).

1. [**Cosine similarity tutorial**](http://blog.christianperone.com/2013/09/machine-learning-cosine-similarity-for-vector-space-models-part-iii/)
   - [**Cosine vs dot product**](https://datascience.stackexchange.com/questions/744/cosine-similarity-versus-dot-product-as-distance-metrics)
   - [**Cosine vs dot product 2**](https://blog.christianperone.com/2013/09/machine-learning-cosine-similarity-for-vector-space-models-part-iii/)
   - [**Fast cosine similarity**](https://stackoverflow.com/questions/51425300/python-fast-cosine-distance-with-cython) **implementation**
2. **Edit distance similarity**
3. [**Diff lib similarity and soundex**](https://datascience.stackexchange.com/questions/12575/similarity-between-two-words)
4. [**Soft cosine and cosine**](https://www.machinelearningplus.com/nlp/gensim-tutorial/)
5. [**Pearson also used to detect similar vectors**](https://machinelearningmastery.com/how-to-use-correlation-to-understand-the-relationship-between-variables/)



### Distance

This section lists common distance metrics and geo tools.

The same notes are in [Dynamic Time Warping (DTW)](../predictive-ml/time-series-search.md#dynamic-time-warping-dtw).

1. [Mastery on distance formulas](https://machinelearningmastery.com/distance-measures-for-machine-learning/)
   1. Role of Distance Measures
   2. Hamming Distance
   3. Euclidean Distance
   4. Manhattan Distance (Taxicab or City Block)
   5. Minkowski Distance
2. Cosine distance = 1 - cosine similarity
3. [Haversine](https://kanoki.org/2019/12/27/how-to-calculate-distance-in-python-and-pandas-using-scipy-spatial-and-distance-functions/) distance

#### Distance Tools

This subsection points to GeoPandas for geographic distance.

1. [GeoPandas](https://geopandas.org/en/stable/index.html)

### **FEATURE IMPUTING**

This section links imputation libraries and exploration guides.

1. [**Vidhya on FE, anomalies, engineering, imputing**](https://www.analyticsvidhya.com/blog/2016/01/guide-data-exploration/?utm_source=outlierdetectionpyod&utm_medium=blog)
2. [**Fancy impute**](https://pypi.org/project/fancyimpute/)

###

TF-IDF notes from this page are on the [TF-IDF](../language-ai/tf-idf.md) page.
