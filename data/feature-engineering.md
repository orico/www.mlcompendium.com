# Feature Engineering

Raw columns are rarely what a model needs, so this page is the work of building features, filling gaps, and measuring how alike two rows are.
It moves from feature engineering and imputing, through representation learning, into similarity, distance, and the distance tools.
### **FEATURE ENGINEERING**

This section is the feature-engineering guides on encoding, text, and embedding features.

The same notes are in [Auto Feature Engineering](../deep-learning/meta-learning.md#auto-feature-engineering), [Box Cox](distribution-transformation.md#box-cox), and [Feature engineering](engineering/lakes-and-warehouses.md#feature-engineering).

- This tutorial on data exploration comprises missing value imputation, outliers, feature engineering, and variable creation, by Sunil Ray. [**Vidhya on FE, anomalies, engineering, imputing**](https://www.analyticsvidhya.com/blog/2016/01/guide-data-exploration/?utm_source=outlierdetectionpyod&utm_medium=blog)
2. [**Many types of FE**](https://medium.com/data-science/understanding-feature-engineering-part-1-continuous-numeric-data-da4e47099a7b)**, including log and box cox transform - a very useful explanation.**
- [**Categorical Data**](https://medium.com/data-science/understanding-feature-engineering-part-2-categorical-data-f54324193e63) hashing really cool
- [**Dummy variables and feature hashing**](https://medium.com/data-science/understanding-feature-engineering-part-2-categorical-data-f54324193e63)
5. [**Text data**](https://medium.com/data-science/understanding-feature-engineering-part-3-traditional-methods-for-text-data-f6f7d70acd41) **- unigrams, bag of words, N-grams (2,3,..), tfidf matrix, cosine\_similarity(tfidf) ontop of a tfidf matrix, unsupervised hierarchical clustering with similarity measures on top of (cosine\_similarity), LDA for topic modelling in sklearn - pretty awesome, Kmeans(lda),.** Dipanjan feature engineering cont numeric categorical traditional methods
- [**Deep learning data for FE**](https://medium.com/data-science/understanding-feature-engineering-part-4-deep-learning-methods-for-text-data-96c44370bbfa)
- **-** [**Word embedding using keras, continuous BOW - CBOW, SKIPGRAM, word2vec - really good.**](https://medium.com/data-science/understanding-feature-engineering-part-4-deep-learning-methods-for-text-data-96c44370bbfa)
7. [**Topic Modelling**](http://chdoig.github.io/pygotham-topic-modeling/#/) **- a fantastic slide show about topic modelling using LDA etc.**

The same notes are in [LDA (Latent Dirichlet Allocation)](../language-ai/topics-modeling.md#lda-latent-dirichlet-allocation).

- **Dipanjan on feature engineering** [**1**](https://medium.com/data-science/understanding-feature-engineering-part-1-continuous-numeric-data-da4e47099a7b)
- **- cont numeric** [ **2 -**](https://medium.com/data-science/understanding-feature-engineering-part-2-categorical-data-f54324193e63)
- **categorical** [**3**](https://medium.com/data-science/understanding-feature-engineering-part-3-traditional-methods-for-text-data-f6f7d70acd41)
- Client Challenge. Client Challenge. [**Target encoding git**](https://pypi.org/project/target_encoding/)
- Client Challenge. Client Challenge. [**Category encoding git**](https://pypi.org/project/category-encoders/)

### **FEATURE IMPUTING**

After the engineering guides, this section links imputation libraries and exploration guides.

- This tutorial on data exploration comprises missing value imputation, outliers, feature engineering, and variable creation, by Sunil Ray. [**Vidhya on FE, anomalies, engineering, imputing**](https://www.analyticsvidhya.com/blog/2016/01/guide-data-exploration/?utm_source=outlierdetectionpyod&utm_medium=blog)
- Client Challenge. Client Challenge. [**Fancy impute**](https://pypi.org/project/fancyimpute/)
### **REPRESENTATION LEARNING**

With imputing named, this subsection links a representation learning paper.

- Representation Learning with Contrastive Predictive Coding. [**paper**](https://arxiv.org/abs/1807.03748?utm_campaign=The%20Batch&utm_source=hs_email&utm_medium=email&utm_content=83602348&_hsenc=p2ANqtz-8DcgUdDF--k3tWOmhM51lm28wHerZxlXKoGNU6hIu2P4Fj-RuEuciKtbWZZdWmvBg7KGeI44FWUrmpHdZIbAM-pUicgg&_hsmi=83602348)

### **SIMILARITY**

After representation, this section covers vector and text similarity measures.

The same notes are in [Recommender Systems](../ai-product/recommender-systems.md) and [String Matching](../language-ai/string-matching.md).

- * It has been a long time since I wrote the TF-IDF tutorial (Part I and Part II) and as I promissed, here is the continuation of the tutorial, by Christian S. Perone. [**Cosine similarity tutorial**](http://blog.christianperone.com/2013/09/machine-learning-cosine-similarity-for-vector-space-models-part-iii/)
 - [**Cosine vs dot product**](https://datascience.stackexchange.com/questions/744/cosine-similarity-versus-dot-product-as-distance-metrics)
 - * It has been a long time since I wrote the TF-IDF tutorial (Part I and Part II) and as I promissed, here is the continuation of the tutorial, by Christian S. Perone. [**Cosine vs dot product 2**](https://blog.christianperone.com/2013/09/machine-learning-cosine-similarity-for-vector-space-models-part-iii/)
 - [**Fast cosine similarity**](https://stackoverflow.com/questions/51425300/python-fast-cosine-distance-with-cython) **implementation**
2. **Edit distance similarity**
3. [**Diff lib similarity and soundex**](https://datascience.stackexchange.com/questions/12575/similarity-between-two-words)
- Gensim is billed as a Natural Language Processing package that does 'Topic Modeling for Humans', by Selva Prabhakaran. [**Soft cosine and cosine**](https://www.machinelearningplus.com/nlp/gensim-tutorial/)
- How to Calculate Correlation Between Variables in Python - MachineLearningMastery.com. [**Pearson also used to detect similar vectors**](https://machinelearningmastery.com/how-to-use-correlation-to-understand-the-relationship-between-variables/)



### Distance

From similarity, this section lists common distance metrics and geo tools, including Distance Tools.

The same notes are in [Dynamic Time Warping (DTW)](../predictive-ml/time-series-search.md#dynamic-time-warping-dtw).

- 4 Distance Measures for Machine Learning - MachineLearningMastery.com. [Mastery on distance formulas](https://machinelearningmastery.com/distance-measures-for-machine-learning/)
 1. Role of Distance Measures
 2. Hamming Distance
 3. Euclidean Distance
 4. Manhattan Distance (Taxicab or City Block)
 5. Minkowski Distance
2. Cosine distance = 1 - cosine similarity
- Working with Geo data is really fun and exciting especially when you clean up all the data and loaded it to a dataframe or to an array, by Your Name. [Haversine](https://kanoki.org/2019/12/27/how-to-calculate-distance-in-python-and-pandas-using-scipy-spatial-and-distance-functions/)

#### Distance Tools

This subsection points to GeoPandas for geographic distance.

- GeoPandas 1.1.4 — GeoPandas 1.1.4+0.g91ec4af.dirty documentation. [GeoPandas](https://geopandas.org/en/stable/index.html)

###

TF-IDF notes from this page are on the [TF-IDF](../language-ai/tf-idf.md) page.
