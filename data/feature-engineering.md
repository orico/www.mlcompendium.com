# Feature Engineering

Raw columns are rarely what a model needs, so this page is the work of building features, filling gaps, and measuring how alike two rows are.
It moves from feature engineering and imputing, through representation learning, into similarity, distance, and the distance tools.
### **FEATURE ENGINEERING**

Building a feature starts with looking at the data, and then with one guide per data type: numbers, categories, and text. The same notes are in [Auto Feature Engineering](../deep-learning/meta-learning.md#auto-feature-engineering), [Box Cox](distribution-transformation.md#box-cox), and [Feature engineering](engineering/lakes-and-warehouses.md#feature-engineering).

Sunil Ray's comprehensive guide to data exploration is the starting point, a tutorial that comprises missing value imputation, outliers, feature engineering, and variable creation: [**Vidhya on FE, anomalies, engineering, imputing**](https://www.analyticsvidhya.com/blog/2016/01/guide-data-exploration/?utm_source=outlierdetectionpyod&utm_medium=blog).

From there, Dipanjan's Understanding Feature Engineering series takes one data type per part. Part 1, Continuous Numeric Data, starts from the idea that data makes the world go round and is [**Many types of FE**](https://medium.com/data-science/understanding-feature-engineering-part-1-continuous-numeric-data-da4e47099a7b)**, including log and box cox transform - a very useful explanation.** Part 2 moves from continuous numbers to categorical data, the discrete kind of structured data: [**Categorical Data**](https://medium.com/data-science/understanding-feature-engineering-part-2-categorical-data-f54324193e63), where the hashing is really cool, and the same article is the [**Dummy variables and feature hashing**](https://medium.com/data-science/understanding-feature-engineering-part-2-categorical-data-f54324193e63) reference. Part 3, Traditional Methods for Text Data, is [**Text data**](https://medium.com/data-science/understanding-feature-engineering-part-3-traditional-methods-for-text-data-f6f7d70acd41) **- unigrams, bag of words, N-grams (2,3,..), tfidf matrix, cosine\_similarity(tfidf) ontop of a tfidf matrix, unsupervised hierarchical clustering with similarity measures on top of (cosine\_similarity), LDA for topic modelling in sklearn - pretty awesome, Kmeans(lda),.** That is Dipanjan feature engineering across cont numeric, categorical, and traditional methods.

Part 4 is the hands-on, intuitive approach to deep learning methods for text data (Word2Vec, GloVe and FastText), for turning noisy, unstructured text into structured, vectorized formats any machine learning algorithm can use. It is both [**Deep learning data for FE**](https://medium.com/data-science/understanding-feature-engineering-part-4-deep-learning-methods-for-text-data-96c44370bbfa) and [**Word embedding using keras, continuous BOW - CBOW, SKIPGRAM, word2vec - really good.**](https://medium.com/data-science/understanding-feature-engineering-part-4-deep-learning-methods-for-text-data-96c44370bbfa) Christine Doig's PyGotham 2015 Introduction to Topic Modeling in Python is [**Topic Modelling**](http://chdoig.github.io/pygotham-topic-modeling/#/) **- a fantastic slide show about topic modelling using LDA etc.**

The same notes are in [LDA (Latent Dirichlet Allocation)](../language-ai/topics-modeling.md#lda-latent-dirichlet-allocation).

For quick reference, **Dipanjan on feature engineering** is parts [**1**](https://medium.com/data-science/understanding-feature-engineering-part-1-continuous-numeric-data-da4e47099a7b) **- cont numeric**, [ **2 -**](https://medium.com/data-science/understanding-feature-engineering-part-2-categorical-data-f54324193e63) **categorical**, and [**3**](https://medium.com/data-science/understanding-feature-engineering-part-3-traditional-methods-for-text-data-f6f7d70acd41) for text. Categories also need encoders beyond dummies, and the two Python packages for that are [**Target encoding git**](https://pypi.org/project/target_encoding/) and [**Category encoding git**](https://pypi.org/project/category-encoders/).

### **FEATURE IMPUTING**

A built feature is only usable if its gaps are filled. The same data exploration guide by Sunil Ray covers missing value imputation alongside outliers and variable creation: [**Vidhya on FE, anomalies, engineering, imputing**](https://www.analyticsvidhya.com/blog/2016/01/guide-data-exploration/?utm_source=outlierdetectionpyod&utm_medium=blog). The library for doing it is the Python package [**Fancy impute**](https://pypi.org/project/fancyimpute/).
### **REPRESENTATION LEARNING**

Hand-built and imputed features have an alternative: learning the representation itself. Representation Learning with Contrastive Predictive Coding is the [**paper**](https://arxiv.org/abs/1807.03748?utm_campaign=The%20Batch&utm_source=hs_email&utm_medium=email&utm_content=83602348&_hsenc=p2ANqtz-8DcgUdDF--k3tWOmhM51lm28wHerZxlXKoGNU6hIu2P4Fj-RuEuciKtbWZZdWmvBg7KGeI44FWUrmpHdZIbAM-pUicgg&_hsmi=83602348) that proposes a universal unsupervised approach to extract useful representations from high-dimensional data, since unsupervised learning has not seen the adoption supervised learning has.

### **SIMILARITY**

Once rows are vectors, whether hand-built or learned, the next question is how alike two of them are. The same notes are in [Recommender Systems](../ai-product/recommender-systems.md) and [String Matching](../language-ai/string-matching.md).

Cosine similarity is the default measure. Christian S. Perone's Cosine Similarity for Vector Space Models (Part III) is the continuation of his TF-IDF tutorial (Part I and Part II), and it is the [**Cosine similarity tutorial**](http://blog.christianperone.com/2013/09/machine-learning-cosine-similarity-for-vector-space-models-part-iii/). The Stack Exchange question [**Cosine vs dot product**](https://datascience.stackexchange.com/questions/744/cosine-similarity-versus-dot-product-as-distance-metrics) notes that cosine similarity is just the dot product scaled by the product of the magnitudes, and asks when each makes the better distance metric; the same Perone tutorial on its https address is [**Cosine vs dot product 2**](https://blog.christianperone.com/2013/09/machine-learning-cosine-similarity-for-vector-space-models-part-iii/). When speed matters, [**Fast cosine similarity**](https://stackoverflow.com/questions/51425300/python-fast-cosine-distance-with-cython) **implementation** is the Stack Overflow thread on speeding up scipy's cosine distance with numpy and cython.

Words and strings need their own measures. Edit distance similarity is one. [**Diff lib similarity and soundex**](https://datascience.stackexchange.com/questions/12575/similarity-between-two-words) is the question about a Python library to identify the similarity between two words or sentences, for comparing audio-to-text output against known words such as a person or company name. Selva Prabhakaran's Gensim tutorial, for the package billed as 'Topic Modeling for Humans' that also works with word vector models such as Word2Vec and FastText, covers [**Soft cosine and cosine**](https://www.machinelearningplus.com/nlp/gensim-tutorial/). Correlation is a similarity measure too: MachineLearningMastery's How to Calculate Correlation Between Variables in Python is the [**Pearson also used to detect similar vectors**](https://machinelearningmastery.com/how-to-use-correlation-to-understand-the-relationship-between-variables/) link.



### Distance

Similarity has a mirror image in distance, and some models need the distance form. The same notes are in [Dynamic Time Warping (DTW)](../predictive-ml/time-series-search.md#dynamic-time-warping-dtw).

MachineLearningMastery's 4 Distance Measures for Machine Learning is the [Mastery on distance formulas](https://machinelearningmastery.com/distance-measures-for-machine-learning/) link. It covers the role of distance measures, then Hamming Distance, Euclidean Distance, Manhattan Distance (Taxicab or City Block), and Minkowski Distance. The bridge back to the section above is that cosine distance = 1 - cosine similarity. Geo data needs one more measure: working with Geo data is fun once the data is clean and loaded to a dataframe or an array, and the real work starts when you have to find distances between two coordinates or cities and generate a distance matrix, which is where [Haversine](https://kanoki.org/2019/12/27/how-to-calculate-distance-in-python-and-pandas-using-scipy-spatial-and-distance-functions/) distance with Scipy spatial and pandas comes in.

#### Distance Tools

For geographic distance at scale, the tool is [GeoPandas](https://geopandas.org/en/stable/index.html), whose GeoPandas 1.1.4 documentation is the reference.

###

TF-IDF notes from this page are on the [TF-IDF](../language-ai/tf-idf.md) page.
