# Anomaly Detection

Anomaly detection asks whether a new point looks like the data you already have, and the first decision is whether you are doing novelty detection or outlier detection. The page defines that split, collects the general reading and the libraries, then walks the methods from elliptic envelopes and LOF through isolation forests and one-class SVM, and ends with clustering metrics such as silhouette for related checks.

The same notes are in [ANOMALY DETECTION TS](time-series-search.md#anomaly-detection-ts), [Drift](../ai-engineering/mlops/mlops-monitoring-and-alerts.md#drift), [Fraud Detection](../ai-product/fraud-detection.md), [HDBSCAN*](clustering-algorithms.md#hdbscan), and [PCA for log anomaly detection](templatization.md#pca-for-log-anomaly-detection).

The task is to decide “whether a new observation belongs to the same distribution as existing observations (it is an inlier), or should be considered as different (it is an outlier). Often, this ability is used to clean real data sets.

Two important distinctions must be made:

| novelty detection: | |
| --- | --- |
| | The training data is not polluted by outliers, and we are interested in detecting anomalies in new observations. |
| outlier detection: | |
| | The training data contains outliers, and we need to fit the central mode of the training data, ignoring the deviant observations |

With that split in mind, the general reading starts simple. Using IQR for AD and why IQR difference is 2.7 sigma is explained in [https://towardsdatascience.com/why-1-5-in-iqr-method-of-outlier-detection-5d07fdc82097](https://towardsdatascience.com/why-1-5-in-iqr-method-of-outlier-detection-5d07fdc82097). Medium — good: [https://towardsdatascience.com/anomaly-detection-for-dummies-15f148e559c1](https://towardsdatascience.com/anomaly-detection-for-dummies-15f148e559c1) is Julia Bohutska's "Anomaly Detection - How to Tell Good Performance from Bad". The [kdnuggets](https://www.kdnuggets.com/2017/04/datascience-introduction-anomaly-detection.html) piece is its Introduction to Anomaly Detection. For [Z-score and other moving averages.](https://turi.com/learn/userguide/anomaly_detection/moving_zscore.html), the link now lands on the apple/turicreate repository; Turi Create simplifies the development of custom machine learning models. A survey that sat here is kept at the end of the page.

From reading to code, [A great tutorial](https://www.analyticsvidhya.com/blog/2019/02/outlier-detection-python-pyod/) is an awesome tutorial on what an outlier is and how to use the PyOD library for outlier detection in Python. It is about AD using 20 algos in a [single python package](https://github.com/yzhao062/pyod); that repository now describes itself as a Python library for anomaly detection across tabular, time series, graph, text, image, and audio data, with 60+ detectors. For rare events in sequences, [Mastery on classifying rare events using lstm-autoencoder](https://machinelearningmastery.com/lstm-model-architecture-for-rare-event-time-series-forecasting/) is MachineLearningMastery's LSTM model architecture for rare event time series forecasting.

The same notes are in [AUTOENCODERS](../deep-learning/autoencoders.md#autoencoders) and [Timeseries](forecasting.md).

Before picking one method, it helps to see them side by side. A [comparison](http://scikit-learn.org/stable/modules/outlier_detection.html#outlier-detection) of One-class SVM versus Elliptic Envelope versus Isolation Forest versus LOF in sklearn is part of the novelty and outlier detection guide. (The examples below illustrate how the performance of the [covariance.EllipticEnvelope](http://scikit-learn.org/stable/modules/generated/sklearn.covariance.EllipticEnvelope.html#sklearn.covariance.EllipticEnvelope) degrades as the data is less and less unimodal. The [svm.OneClassSVM](http://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html#sklearn.svm.OneClassSVM) works better on data with multiple modes and [ensemble.IsolationForest](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html#sklearn.ensemble.IsolationForest) and [neighbors.LocalOutlierFactor](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html#sklearn.neighbors.LocalOutlierFactor) perform well in every cases.) Each of those four API pages links to the gallery examples that compare anomaly detection algorithms for outlier detection on toy datasets.

The applied examples come next. [Using Autoencoders](https://shiring.github.io/machine_learning/2017/05/01/fraud) — the information is there, but its all over the place. Twitter anomaly is another reference. Microsoft anomaly — a well documented black box, i cant find a description of the algorithm, just hints to what they sort of did. Its blog post, [up/down trend, dynamic range, tips and dips](https://blogs.technet.microsoft.com/machinelearning/2014/11/05/anomaly-detection-using-machine-learning-to-detect-abnormalities-in-time-series-data/), is "Anomaly Detection – Using Machine Learning to Detect Abnormalities in Time Series Data". The [Api here](https://docs.microsoft.com/en-us/azure/machine-learning/team-data-science-process/apps-anomaly-detection-api) link now lands on Microsoft's Cloud Adoption Framework page on the process to plan for AI adoption. STL and [LSTM for anomaly prediction](https://github.com/omri374/moda/blob/master/moda/example/lstm/LSTM_AD.ipynb) is the notebook in omri374/moda, a models and evaluation framework for trending topics detection.

Two more Medium posts round out the reading. Medium on AD is [https://towardsdatascience.com/machine-learning-for-anomaly-detection-and-condition-monitoring-d4614e7de770](https://towardsdatascience.com/machine-learning-for-anomaly-detection-and-condition-monitoring-d4614e7de770) on Towards Data Science, and Medium on AD using mahalanobis, AE and is [https://towardsdatascience.com/how-to-use-machine-learning-for-anomaly-detection-and-condition-monitoring-6742f82900d7](https://towardsdatascience.com/how-to-use-machine-learning-for-anomaly-detection-and-condition-monitoring-6742f82900d7).

### OUTLIER DETECTION

Once the problem is framed, the next question is which library to run it with. [Alibi Detect](https://github.com/SeldonIO/alibi-detect) is an open source Python library focused on outlier, adversarial and drift detection. The package aims to cover both online and offline detectors for tabular data, text, images and time series. The outlier detection methods should allow the user to identify global, contextual and collective outliers.

<figure><img src="../.gitbook/assets/gimg-c9ce572f20fe.png" alt=""><figcaption><p>Alibi Detect.</p><p>Credit: <a href="https://lh4.googleusercontent.com/QonFzFq66lICpFO_ZMwHOOVbf414oWxdIoV1CibK2OD5jlaRTgQGrs1cgitF2vv3HE0NitUn5XILiZRs3GRIGnDtBWbJEhcppaAhlxjThvS3_dBgyfkBoM1dKlFEgUk1Vy3yeVyc">copied from the original hosted image</a>.</p></figcaption></figure>

The PyOD package from the tutorial above has its own documentation, the [Pyod](https://pyod.readthedocs.io/en/latest/) pyod 3.6.6 documentation, and the three figures below are from PyOD.

<figure><img src="../.gitbook/assets/gimg-4462e0610ead.png" alt=""><figcaption><p>PyOD.</p><p>Credit: <a href="https://lh5.googleusercontent.com/ZKkwCMKak5EBt4hGR2NMnx_XLmc8UBkLb5-AlD83QnhpVddGHadQGajp0eutz-lo7WTK9cZdPwe6YWg4LeEgxbR5FtdxzAJ_KtE3JiXMnDfkzElJznOJQt_sqslltPkKPP3i-uv2">copied from the original hosted image</a>.</p></figcaption></figure>

<figure><img src="../.gitbook/assets/gimg-79ab319d3e40.png" alt=""><figcaption><p>PyOD.</p><p>Credit: <a href="https://lh3.googleusercontent.com/Shm9hSKFYXqN9ab4dYa92zlsTfBle5z_iTtLSobJPpjWyo53-vNtDI7DTL-h32mCX8lea-AxGXF9UxlY_9BhFn21UlduhYz74X8X92JxiMqSymRW4JgrFoaJMy6sizWbBEi7zM2N">copied from the original hosted image</a>.</p></figcaption></figure>

<figure><img src="../.gitbook/assets/gimg-0fdcf6a17ed3.png" alt=""><figcaption><p>PyOD.</p><p>Credit: <a href="https://lh6.googleusercontent.com/kmW2KZFP6OY0xth2NwTXwrMajzeXG6LY1PQpAkejy-hVmR32eauIwI2REmzahEBKRIAkooaDcwq4OXBs_I-nacg4ncZljKg9WTA2RDX3PJdM6oHUxC6O_fukyh6SEwnnvZQPsSvB">copied from the original hosted image</a>.</p></figcaption></figure>

From the same author, the (great) [Anomaly detection resources](https://github.com/yzhao062/anomaly-detection-resources) list collects anomaly detection related books, papers, videos, and toolboxes, last updated in late 2025 for LLM and VLM works. [Novelty and outlier detection inm sklearn](https://scikit-learn.org/stable/modules/outlier_detection.html) is the scikit-learn guide for applications that need to decide whether a new observation is an inlier or an outlier.

When one detector is not enough, [SUOD](https://github.com/yzhao062/suod) (Scalable Unsupervised Outlier Detection) is an acceleration framework for large-scale unsupervised outlier detector training and prediction. Notably, anomaly detection is often formulated as an unsupervised problem since the ground truth is expensive to acquire. To compensate for the unstable nature of unsupervised algorithms, practitioners often build a large number of models for further combination and analysis, e.g., taking the average or majority vote. However, this poses scalability challenges in high-dimensional, large datasets, especially for proximity-base models operating in Euclidean space.

<figure><img src="../.gitbook/assets/gimg-8bb2d61f1082.png" alt=""><figcaption><p>SUOD.</p><p>Credit: <a href="https://lh4.googleusercontent.com/lTrANgbDggSvC5zIKxuzzSKYYgMNJX7yN9Vni3FTWj7kKSpBuxhc2vvE2Oy_diF4uEalUovH3sVeIdmAfBtsTFKPL3vgzMfnX50_8yUVENyV1uMx6fRO4gKLjGAfhnZy38dAE6_y">copied from the original hosted image</a>.</p></figcaption></figure>

SUOD is therefore proposed to address the challenge at three complementary levels: random projection (data level), pseudo-supervised approximation (model level), and balanced parallel scheduling (system level). As mentioned, the key focus is to accelerate the training and prediction when a large number of anomaly detectors are presented, while preserving the prediction capacity. Since its inception in Jan 2019, SUOD has been successfully used in various academic researches and industry applications, include PyOD [[2]](https://github.com/yzhao062/suod#zhao2019pyod) and [IQVIA](https://www.iqvia.com/) medical claim analysis. It could be especially useful for outlier ensembles that rely on a large number of base estimators.

For time series metrics in production, [Skyline](https://github.com/earthgecko/skyline) is real-time anomaly detection for time series metrics: 11 years of production anomaly detection evolution, multi-algorithm ensembles, pattern recognition using semi-supervised and unsupervised learning, and correlation analysis, learning and getting better over time with no GPU required, so it can run on a VPS.

Scikit-lego outliers is the last library here; its page is kept at the end, and the figures below are from it.

<figure><img src="../.gitbook/assets/gimg-a2e83f028e2a.png" alt=""><figcaption><p>Scikit-lego outliers.</p><p>Credit: <a href="https://lh3.googleusercontent.com/unjrP1o3wqwUvv_J0WeX_9BZw8qrq9ToBVjSAHc1bWxOo3idh6CSLsVPTKSNovXve0-IOG5vaL5yqn4sg0a6OfvSM_X5t41wK-P_NFHjOzmmJyHKsv8I6se62OZtyildGKI5ZlrV">copied from the original hosted image</a>.</p></figcaption></figure>

<figure><img src="../.gitbook/assets/gimg-d3e77feb78ab.png" alt=""><figcaption><p>Scikit-lego outliers.</p><p>Credit: <a href="https://lh5.googleusercontent.com/bafZPqSAbvczD3CE2yIPsPlTaYZ5qSAMdz4l7WqeuhQK-XjONBQDP0-tTYXjFcnMPlvljiMr1_fvMlAFCLRtATsI3mcaXjxbcjcSD97OxVzVR41qecC1BZo9DKdYag7e97g2Jirk">copied from the original hosted image</a>.</p></figcaption></figure>

<figure><img src="../.gitbook/assets/gimg-c69974b4f0f1.png" alt=""><figcaption><p>Scikit-lego outliers.</p><p>Credit: <a href="https://lh5.googleusercontent.com/9bBkl9p2YSeKumH3C2nwIpGdQvBYqt63JHtQsfJfS2wJqRJBWcLyHpZ1yuFEHh4tFdcUAc9dm-ihYYIa_h9Doa_AZpv273V0T5kEpGRfigyNXtRmR2XQWYQAVc9VFaQ-r6LPuA1-">copied from the original hosted image</a>.</p></figcaption></figure>

<figure><img src="../.gitbook/assets/gimg-248cf19aa992.png" alt=""><figcaption><p>Scikit-lego outliers.</p><p>Credit: <a href="https://lh6.googleusercontent.com/FJ_1DRIuNjz3FY_9d1QGeFb4tv6E-CK97eoaNvskApfKJETYKhLoq64gMvtqbBkGZNzeA3ZtcfenuhhYc9in9ILtv8v61cYyc6XN44obZmmMl_hBylk53NNdwVEPujJDS0hLKwyN">copied from the original hosted image</a>.</p></figcaption></figure>

### ELLIPTIC ENVELOPE

The first method in the sklearn comparison is the one with the strongest assumption. We assume that the regular data come from a known distribution (e.g. data are Gaussian distributed). From this assumption, we generally try to define the “shape” of the data, and can define outlying observations as observations which stand far enough from the fit shape. That is also why it degrades as the data becomes less unimodal.

### LOCAL OUTLIER FACTOR

When no single distribution fits, LOF compares each point only to its neighbours. [LOF](http://scikit-learn.org/stable/modules/outlier_detection.html#local-outlier-factor) computes a score (called local outlier factor) reflecting the degree of abnormality of the observations. It measures the local density deviation of a given data point with respect to its neighbors. The idea is to detect the samples that have a substantially lower density than their neighbors. In practice the local density is obtained from the k-nearest neighbors.

The LOF score of an observation is equal to the ratio of the average local density of his k-nearest neighbors, and its own local density: a normal instance is expected to have a local density similar to that of its neighbors, while abnormal data are expected to have much smaller local density.

### ISOLATION FOREST

LOF measures density; an isolation forest measures how easy a point is to cut off. The best resource to explain isolation forest (kept at the end of the page) gives the basic idea: for an anomaly (in the example) only 4 partitions are needed, for a regular point in the middle of a distribution, you need many many more.

The scikit-learn [Isolation Forest](http://scikit-learn.org/stable/auto_examples/ensemble/plot_isolation_forest.html) example describes the model as an ensemble of "Isolation Trees" that "isolate" observations by recursive random partitioning. Isolating observations:

- randomly selecting a feature
- randomly selecting a split value between the maximum and minimum values of the selected feature.

Recursive partitioning can be represented by a tree structure, the number of splittings required to isolate a sample is equivalent to the path length from the root node to the terminating node. This path length, averaged over a forest of such random trees, is a measure of normality and our decision function. Random partitioning produces noticeable shorter paths for anomalies. So when a forest of random trees collectively produce shorter path lengths for particular samples, they are highly likely to be anomalies.

[the paper is pretty good too -](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf) In the training stage, iTrees are constructed by recursively partitioning the given training set until instances are isolated or a specific tree height is reached of which results a partial model.

Note that the tree height limit l is automatically set by the sub-sampling size ψ: l = ceiling(log2 ψ), which is approximately the average tree height [7].

The rationale of growing trees up to the average tree height is that we are only interested in data points that have shorter-than average path lengths, as those points are more likely to be anomalies

### ONE CLASS SVM

The last method draws a boundary around the normal data instead of scoring each point, which is why the sklearn comparison found it better on data with multiple modes. The same notes are in [SUPPORT VECTOR MACHINES (SVM)](linear-separator-algorithms.md#support-vector-machines-svm).

A nice article about ocs, with github code, two methods are described; it is kept at the end of the page. The [Resources for ocsvm](https://www.quora.com/What-is-a-good-resource-for-understanding-One-Class-SVM-for-distribution-esitmation) thread asks what a good resource is for understanding One Class SVM for distribution estimation. It looks like there are two such methods — The 2nd one: The algorithm obtains a spherical boundary, in feature space, around the data. The volume of this hypersphere is minimized, to minimize the effect of incorporating outliers in the solution.

The resulting hypersphere is characterized by a center and a radius R>0 as distance from the center to (any support vector on) the boundary, of which the volume R2 will be minimized.

### CLUSTERING METRICS

Anomalies and clusters are two sides of the same question, so the page ends with how to judge a clustering, for community detection, text clusters, etc. The same notes are in [Clustering Algorithms](clustering-algorithms.md).

For word embeddings in particular, the [Google search for convenience](https://www.google.com/search?biw=1600&bih=912&sxsrf=ALeKk00NbB52pfM6J1N42ieEddIOirBmcQ%3A1597514997743&ei=9SQ4X-_uLLLhkgWJsIbADw&q=word+embedding+silhouette+score&oq=word+embedding+silhouette+score&gs_lcp=CgZwc3ktYWIQAzoECAAQRzoECCMQJzoHCCMQsAIQJ1DVd1jqjQFgm5ABaARwAXgBgAGMAogBrQ2SAQUwLjkuMpgBAKABAaoBB2d3cy13aXrAAQE&sclient=psy-ab&ved=0ahUKEwivvdqP553rAhWysKQKHQmYAfg4ChDh1QMIDA&uact=5) is a saved Google Search for word embedding silhouette score.

**Silhouette:**

[TFIDF, PCA, SILHOUETTE](https://medium.com/data-science/mmmm-foodporn-a-clustering-and-classification-study-using-natural-language-processing-e2eae8ddefe1) is the study for deciding how many clusters to use, the knee/elbow method; the same post is also at [https://towardsdatascience.com/mmmm-foodporn-a-clustering-and-classification-study-using-natural-language-processing-e2eae8ddefe1](https://towardsdatascience.com/mmmm-foodporn-a-clustering-and-classification-study-using-natural-language-processing-e2eae8ddefe1), Dan Wilentz's "Mmmm Foodporn! A Clustering and Classification Study using Natural Language Processing", using data science to analyze cuisine popularity on Reddit. Its [Topic modelling clustering, cant access this document on github](https://github.com/danielwilentz/Cuisine-Classifier/blob/master/topic_modeling/clustering.ipynb) notebook uses unsupervised learning and NLP to cluster titles from posts on r/foodporn and classify new titles. [Embedding based silhouette community detection](https://link.springer.com/article/10.1007/s10994-020-05882-8#Sec10) applies silhouette to community detection. [A notebook](https://rlbarter.github.io/superheat-examples/word2vec/), using the SuperHeat package, clusters a w2v cosine similarity matrix, measuring using silhouette score.

## Deprecated links

{% hint style="warning" %}
These links and images no longer work. The original wording is kept here. A same-resource copy, when one was checked, is used above.
{% endhint %}

- A survey. This address no longer opens: https://d1wqtxts1xzle7.cloudfront.net/49916547/Mohiuddin_Survey_financial_2015.pdf?1477591055=&response-content-disposition=inline%3B+filename%3DA_survey_of_anomaly_detection_techniques.pdf&Expires=1594649751&Signature=U~N32meGWYyIIQz1zRYC4s2tCb7e5ut28GIBC3GSG4250UjhgTMQwEIB63zwPKtS5JyKew7RWVog8gytIhc3GSSfTwsRM7lqyghuDgbds-QMp3mNyVw2bYNztnoOWncHG8rhtkwUK1EbWcYeLKvqARnJoAS177C8r1GAhfKp14GgJzHpmnsoSkB6AowJ68nauf2VyA1b~w1m~UfSNoWtjbL59clAqHn7nfqw5PGBuLHSSSxCa5PX09mADy4VzuOySzYjIviRwOlgT1eQrART0KqozqVSiGKM3SeapuI3K5tSERVPPSTnpupp--WJyYCNzzvPrdjB121P2XU7fq73wQ__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA
- Pyod. This address no longer opens: https://pyod.readthedocs.io/en/latest/pyod.html
- Scikit-lego. This address no longer opens: https://scikit-lego.readthedocs.io/en/latest/outliers.html
- The best resource to explain isolation forest. This address no longer opens: http://blog.easysol.net/using-isolation-forests-anamoly-detection/
- A nice article about ocs, with github code, two methods are described. This address no longer opens: http://rvlasveld.github.io/blog/2013/07/12/introduction-to-one-class-support-vector-machines/
- two such methods. This address no longer opens: http://rvlasveld.github.io/blog/2013/07/12/introduction-to-one-class-support-vector-machines/
