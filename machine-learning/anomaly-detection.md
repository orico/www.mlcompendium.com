# Anomaly Detection

**“whether a new observation belongs to the same distribution as existing observations (it is an inlier), or should be considered as different (it is an outlier).**&#x20;

**=> Often, this ability is used to clean real data sets**

**Two important distinctions must be made:**

| **novelty detection:** |                                                                                                                                                              |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
|                        | **The training data is not polluted by outliers, and we are interested in detecting anomalies in new observations.**                                         |
| **outlier detection:** |                                                                                                                                                              |
|                        | <p><strong>The training data contains outliers, and we need to fit the central mode of the training data, ignoring the deviant observations</strong><br></p> |

1. [Using IQR for AD and why IQR difference is 2.7 sigma](https://towardsdatascience.com/why-1-5-in-iqr-method-of-outlier-detection-5d07fdc82097)
2. [**Medium**](https://towardsdatascience.com/anomaly-detection-for-dummies-15f148e559c1) **- good**
3. [**kdnuggets**](https://www.kdnuggets.com/2017/04/datascience-introduction-anomaly-detection.html)
4. **Index for** [**Z-score and other moving averages.** ](https://turi.com/learn/userguide/anomaly\_detection/moving\_zscore.html)
5. [**A survey**](https://d1wqtxts1xzle7.cloudfront.net/49916547/Mohiuddin\_Survey\_financial\_2015.pdf?1477591055=\&response-content-disposition=inline%3B+filename%3DA\_survey\_of\_anomaly\_detection\_techniques.pdf\&Expires=1594649751\&Signature=U\~N32meGWYyIIQz1zRYC4s2tCb7e5ut28GIBC3GSG4250UjhgTMQwEIB63zwPKtS5JyKew7RWVog8gytIhc3GSSfTwsRM7lqyghuDgbds-QMp3mNyVw2bYNztnoOWncHG8rhtkwUK1EbWcYeLKvqARnJoAS177C8r1GAhfKp14GgJzHpmnsoSkB6AowJ68nauf2VyA1b\~w1m\~UfSNoWtjbL59clAqHn7nfqw5PGBuLHSSSxCa5PX09mADy4VzuOySzYjIviRwOlgT1eQrART0KqozqVSiGKM3SeapuI3K5tSERVPPSTnpupp--WJyYCNzzvPrdjB121P2XU7fq73wQ\_\_\&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA)
6. [**A great tutorial**](https://www.analyticsvidhya.com/blog/2019/02/outlier-detection-python-pyod/?utm\_source=facebook.com\&utm\_medium=social\&fbclid=IwAR33KDnGMf5zp491WmhTsCFtinBDUp5RaVnoC4Cfxcc5rfo2yHreMo3M\_M4) **about AD using 20 algos in a** [**single python package**](https://github.com/yzhao062/pyod)**.**
7. [**Mastery on classifying rare events using lstm-autoencoder**](https://machinelearningmastery.com/lstm-model-architecture-for-rare-event-time-series-forecasting/)
8. **A** [**comparison**](http://scikit-learn.org/stable/modules/outlier\_detection.html#outlier-detection) **of One-class SVM versus Elliptic Envelope versus Isolation Forest versus LOF in sklearn. (The examples below illustrate how the performance of the** [**covariance.EllipticEnvelope**](http://scikit-learn.org/stable/modules/generated/sklearn.covariance.EllipticEnvelope.html#sklearn.covariance.EllipticEnvelope) **degrades as the data is less and less unimodal. The** [**svm.OneClassSVM**](http://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html#sklearn.svm.OneClassSVM) **works better on data with multiple modes and** [**ensemble.IsolationForest**](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html#sklearn.ensemble.IsolationForest) **and**[**neighbors.LocalOutlierFactor**](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html#sklearn.neighbors.LocalOutlierFactor) **perform well in every cases.)**
9. [**Using Autoencoders**](https://shiring.github.io/machine\_learning/2017/05/01/fraud) **- the information is there, but its all over the place.**
10. **Twitter anomaly -**
11. **Microsoft anomaly - a well documented black box, i cant find a description of the algorithm, just hints to what they sort of did**
    1. [**up/down trend, dynamic range, tips and dips**](https://blogs.technet.microsoft.com/machinelearning/2014/11/05/anomaly-detection-using-machine-learning-to-detect-abnormalities-in-time-series-data/)
    2. [**Api here**](https://docs.microsoft.com/en-us/azure/machine-learning/team-data-science-process/apps-anomaly-detection-api)&#x20;
12. **STL and** [**LSTM for anomaly prediction**](https://github.com/omri374/moda/blob/master/moda/example/lstm/LSTM\_AD.ipynb) **by microsoft**
    1. [**Medium on AD**](https://towardsdatascience.com/machine-learning-for-anomaly-detection-and-condition-monitoring-d4614e7de770)
    2. [**Medium on AD using mahalanobis, AE and**](https://towardsdatascience.com/how-to-use-machine-learning-for-anomaly-detection-and-condition-monitoring-6742f82900d7)&#x20;

### **OUTLIER DETECTION**

1. [**Alibi Detect**](https://github.com/SeldonIO/alibi-detect) **is an open source Python library focused on outlier, adversarial and drift detection. The package aims to cover both online and offline detectors for tabular data, text, images and time series. The outlier detection methods should allow the user to identify global, contextual and collective outliers.**\
   ![](https://lh4.googleusercontent.com/QonFzFq66lICpFO\_ZMwHOOVbf414oWxdIoV1CibK2OD5jlaRTgQGrs1cgitF2vv3HE0NitUn5XILiZRs3GRIGnDtBWbJEhcppaAhlxjThvS3\_dBgyfkBoM1dKlFEgUk1Vy3yeVyc)\
   \

2. [**Pyod**](https://pyod.readthedocs.io/en/latest/pyod.html)

![](https://lh5.googleusercontent.com/ZKkwCMKak5EBt4hGR2NMnx\_XLmc8UBkLb5-AlD83QnhpVddGHadQGajp0eutz-lo7WTK9cZdPwe6YWg4LeEgxbR5FtdxzAJ\_KtE3JiXMnDfkzElJznOJQt\_sqslltPkKPP3i-uv2)

![](https://lh3.googleusercontent.com/Shm9hSKFYXqN9ab4dYa92zlsTfBle5z\_iTtLSobJPpjWyo53-vNtDI7DTL-h32mCX8lea-AxGXF9UxlY\_9BhFn21UlduhYz74X8X92JxiMqSymRW4JgrFoaJMy6sizWbBEi7zM2N)![](https://lh6.googleusercontent.com/kmW2KZFP6OY0xth2NwTXwrMajzeXG6LY1PQpAkejy-hVmR32eauIwI2REmzahEBKRIAkooaDcwq4OXBs\_I-nacg4ncZljKg9WTA2RDX3PJdM6oHUxC6O\_fukyh6SEwnnvZQPsSvB)

1. [**Anomaly detection resources**](https://github.com/yzhao062/anomaly-detection-resources) **(great)**
2. [**Novelty and outlier detection inm sklearn**](https://scikit-learn.org/stable/modules/outlier\_detection.html)
3. [**SUOD**](https://github.com/yzhao062/suod) **(Scalable Unsupervised Outlier Detection) is an acceleration framework for large-scale unsupervised outlier detector training and prediction. Notably, anomaly detection is often formulated as an unsupervised problem since the ground truth is expensive to acquire. To compensate for the unstable nature of unsupervised algorithms, practitioners often build a large number of models for further combination and analysis, e.g., taking the average or majority vote. However, this poses scalability challenges in high-dimensional, large datasets, especially for proximity-base models operating in Euclidean space.**

![](https://lh4.googleusercontent.com/lTrANgbDggSvC5zIKxuzzSKYYgMNJX7yN9Vni3FTWj7kKSpBuxhc2vvE2Oy\_diF4uEalUovH3sVeIdmAfBtsTFKPL3vgzMfnX50\_8yUVENyV1uMx6fRO4gKLjGAfhnZy38dAE6\_y)

**SUOD is therefore proposed to address the challenge at three complementary levels: random projection (data level), pseudo-supervised approximation (model level), and balanced parallel scheduling (system level). As mentioned, the key focus is to accelerate the training and prediction when a large number of anomaly detectors are presented, while preserving the prediction capacity. Since its inception in Jan 2019, SUOD has been successfully used in various academic researches and industry applications, include PyOD** [**\[2\]**](https://github.com/yzhao062/suod#zhao2019pyod) **and** [**IQVIA**](https://www.iqvia.com/) **medical claim analysis. It could be especially useful for outlier ensembles that rely on a large number of base estimators.**\


1. [**Skyline**](https://github.com/earthgecko/skyline)
2. [**Scikit-lego**](https://scikit-lego.readthedocs.io/en/latest/outliers.html)
   1. ![](https://lh3.googleusercontent.com/unjrP1o3wqwUvv\_J0WeX\_9BZw8qrq9ToBVjSAHc1bWxOo3idh6CSLsVPTKSNovXve0-IOG5vaL5yqn4sg0a6OfvSM\_X5t41wK-P\_NFHjOzmmJyHKsv8I6se62OZtyildGKI5ZlrV)
   2. ![](https://lh5.googleusercontent.com/bafZPqSAbvczD3CE2yIPsPlTaYZ5qSAMdz4l7WqeuhQK-XjONBQDP0-tTYXjFcnMPlvljiMr1\_fvMlAFCLRtATsI3mcaXjxbcjcSD97OxVzVR41qecC1BZo9DKdYag7e97g2Jirk)

![](https://lh5.googleusercontent.com/9bBkl9p2YSeKumH3C2nwIpGdQvBYqt63JHtQsfJfS2wJqRJBWcLyHpZ1yuFEHh4tFdcUAc9dm-ihYYIa\_h9Doa\_AZpv273V0T5kEpGRfigyNXtRmR2XQWYQAVc9VFaQ-r6LPuA1-)

1. ![](https://lh6.googleusercontent.com/FJ\_1DRIuNjz3FY\_9d1QGeFb4tv6E-CK97eoaNvskApfKJETYKhLoq64gMvtqbBkGZNzeA3ZtcfenuhhYc9in9ILtv8v61cYyc6XN44obZmmMl\_hBylk53NNdwVEPujJDS0hLKwyN)

###

### **ISOLATION FOREST**

[**The best resource to explain isolation forest**](http://blog.easysol.net/using-isolation-forests-anamoly-detection/) **- the basic idea is that for an anomaly (in the example) only 4 partitions are needed, for a regular point in the middle of a distribution, you need many many more.**\


[**Isolation Forest**](http://scikit-learn.org/stable/auto\_examples/ensemble/plot\_isolation\_forest.html) **-Isolating observations:**

* **randomly selecting a feature**&#x20;
* **randomly selecting a split value between the maximum and minimum values of the selected feature.**

**Recursive partitioning can be represented by a tree structure, the number of splittings required to isolate a sample is equivalent to the path length from the root node to the terminating node.**\


**This path length, averaged over a forest of such random trees, is a measure of normality and our decision function.**

**Random partitioning produces noticeable shorter paths for anomalies.**\


**=> when a forest of random trees collectively produce shorter path lengths for particular samples, they are highly likely to be anomalies.**\


&#x20;[**the paper is pretty good too -**](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf)  **In the training stage, iTrees are constructed by recursively partitioning the given training set until instances are isolated or a specific tree height is reached of which results a partial model.**&#x20;

**Note that the tree height limit l is automatically set by the sub-sampling size ψ: l = ceiling(log2 ψ), which is approximately the average tree height \[7].**&#x20;

**The rationale of growing trees up to the average tree height is that we are only interested in data points that have shorter-than average path lengths, as those points are more likely to be anomalies**\
\


### **LOCAL OUTLIER FACTOR**

* [**LOF**](http://scikit-learn.org/stable/modules/outlier\_detection.html#local-outlier-factor) **computes a score (called local outlier factor) reflecting the degree of abnormality of the observations.**
* **It measures the local density deviation of a given data point with respect to its neighbors. The idea is to detect the samples that have a substantially lower density than their neighbors.**
* **In practice the local density is obtained from the k-nearest neighbors.**&#x20;
* **The LOF score of an observation is equal to the ratio of the average local density of his k-nearest neighbors, and its own local density:**&#x20;
  * **a normal instance is expected to have a local density similar to that of its neighbors,**&#x20;
  * **while abnormal data are expected to have much smaller local density.**

### **ELLIPTIC ENVELOPE**

1. **We  assume that the regular data come from a known distribution (e.g. data are Gaussian distributed).**&#x20;
2. **From this assumption, we generally try to define the “shape” of the data,**&#x20;
3. **And can define outlying observations as observations which stand far enough from the fit shape.**

### **ONE CLASS SVM**

1. [**A nice article about ocs, with github code, two methods are described.**](http://rvlasveld.github.io/blog/2013/07/12/introduction-to-one-class-support-vector-machines/)
2. [**Resources for ocsvm**](https://www.quora.com/What-is-a-good-resource-for-understanding-One-Class-SVM-for-distribution-esitmation)
3. **It looks like there are** [**two such methods**](http://rvlasveld.github.io/blog/2013/07/12/introduction-to-one-class-support-vector-machines/)**, - The 2nd one: The algorithm obtains a spherical boundary, in feature space, around the data. The volume of this hypersphere is minimized, to minimize the effect of incorporating outliers in the solution.**

**The resulting hypersphere is characterized by a center and a radius R>0 as distance from the center to (any support vector on) the boundary, of which the volume R2 will be minimized.**\
\


### **CLUSTERING METRICS**

**For community detection, text clusters, etc.**\


[**Google search for convenience**](https://www.google.com/search?biw=1600\&bih=912\&sxsrf=ALeKk00NbB52pfM6J1N42ieEddIOirBmcQ%3A1597514997743\&ei=9SQ4X-\_uLLLhkgWJsIbADw\&q=word+embedding+silhouette+score\&oq=word+embedding+silhouette+score\&gs\_lcp=CgZwc3ktYWIQAzoECAAQRzoECCMQJzoHCCMQsAIQJ1DVd1jqjQFgm5ABaARwAXgBgAGMAogBrQ2SAQUwLjkuMpgBAKABAaoBB2d3cy13aXrAAQE\&sclient=psy-ab\&ved=0ahUKEwivvdqP553rAhWysKQKHQmYAfg4ChDh1QMIDA\&uact=5)

**Silhouette:**

1. [**TFIDF, PCA, SILHOUETTE**](https://towardsdatascience.com/mmmm-foodporn-a-clustering-and-classification-study-using-natural-language-processing-e2eae8ddefe1) **for deciding how many clusters to use, the knee/elbow method.**
2. [**Embedding based silhouette community detection**](https://link.springer.com/article/10.1007/s10994-020-05882-8#Sec10)
3. [**A notebook**](https://rlbarter.github.io/superheat-examples/word2vec/)**, using the SuperHeat package, clustering w2v cosine similarity matrix, measuring using silhouette score.**&#x20;
4. [**Topic modelling clustering, cant access this document on github**](https://github.com/danielwilentz/Cuisine-Classifier/blob/master/topic\_modeling/clustering.ipynb)
