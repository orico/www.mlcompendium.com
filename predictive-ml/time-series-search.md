### CLUSTERING TS

This subsection warns about rolling-window pitfalls and lists tslearn clustering.

The same notes are in [Clustering Algorithms](clustering-algorithms.md).

1. Clustering time series, subsequences with a rolling window, the pitfall.
2. [Clustering using tslearn](https://tslearn.readthedocs.io/en/stable/user_guide/clustering.html)
3. [Kmeans for variable length](https://medium.com/@iliazaitsev/how-to-classify-a-dataset-with-observations-of-various-length-96fab8e95baf), [notebook](https://github.com/devforfu/Blog/blob/master/trees/scikit_learn.py)

### ANOMALY DETECTION TS

This subsection surveys time-series anomaly detection methods and toolkits.

The same notes are in [Anomaly Detection](anomaly-detection.md).

1. [What is stationary (process](https://en.wikipedia.org/wiki/Stationary_process)), stationary time series analysis (shay palachi),
2. [mastery on arimas](https://machinelearningmastery.com/time-series-forecasting-methods-in-python-cheat-sheet/)
3. TS anomaly algos (stl, trees, arima)
4. [AD techniques](https://medium.com/dp6-us-blog/anomaly-detection-techniques-c3817e8e7b2f), part [2](https://medium.com/dp6-us-blog/anomaly-detection-techniques-part-ii-9a08b6562619), part [3](https://medium.com/dp6-us-blog/anomaly-detection-techniques-part-iii-d27e7b0d6c8a)
5. [Z-score, modified z-score and iqr an intro why z-score is not robust](http://colingorrie.github.io/outlier-detection.html)
6. [Adtk](https://adtk.readthedocs.io/en/stable/userguide.html) a sklearn-like toolkit with an amazing intro, various algorithms for non seasonal and seasonal, transformers, ensembles.
7. [Awesome TS anomaly detection](https://github.com/rob-med/awesome-TS-anomaly-detection) on github
8. [Transfer learning toolkit](https://github.com/FuzhenZhuang/Transfer-Learning-Toolkit), [paper and benchmarks](https://arxiv.org/pdf/1911.08967.pdf)
9. [Ransac is a good baseline](https://medium.com/@iamhatesz/random-sample-consensus-bd2bb7b1be75) - random sample consensus for outlier detection
   1. [Ransac](https://medium.com/@angel.manzur/got-outliers-ransac-them-f12b6b5f606e), [2](https://medium.com/@saurabh.dasgupta1/outlier-detection-using-the-ransac-algorithm-de52670adb4a), 3, 4, 5, 6
   2. You can feed ransac with tsfresh/tslearn features.
10. [Anomaly detection for time series](https://medium.com/@jetnew/anomaly-detection-of-time-series-data-e0cb6b382e33),
11. AD for TS, recommended by DTAIDistance, [anomatools](https://github.com/Vincent-Vercruyssen/anomatools)
12. STL:
    1. [AD where anomalies coincide with seasonal peaks!!](https://medium.com/@richa.mishr01/anomaly-detection-in-seasonal-time-series-where-anomalies-coincide-with-seasonal-peaks-9859a6a6b8ba)
    2. [AD challenges, stationary, seasonality, trend](https://cloudfabrix.com/blog/aiops/anomaly-detection-time-series-data/)
    3. [Rt anomaly detection for time series pinterest](https://medium.com/pinterest-engineering/building-a-real-time-anomaly-detection-system-for-time-series-at-pinterest-a833e6856ddd) using stl decomposition
    4. [AD](https://medium.com/wwblog/anomaly-detection-using-stl-76099c9fd5a7)
13. Sliding windows
    1. [Solving sliding window problems](https://medium.com/outco/how-to-solve-sliding-window-problems-28d67601a66)
    2. [Rolling window regression](https://medium.com/making-sense-of-data/time-series-next-value-prediction-using-regression-over-a-rolling-window-228f0acae363)
14. Forecasting using Arima 1, [2](http://alkaline-ml.com/pmdarima/)
15. Auto arima 1, [2](https://stackoverflow.com/questions/22770352/auto-arima-equivalent-for-python), [3](https://www.analyticsvidhya.com/blog/2018/08/auto-arima-time-series-modeling-python-r/)
16. [Twitters ESD test](https://medium.com/@elisha_12808/time-series-anomaly-detection-with-twitters-esd-test-50cce409ced1) for outliers, using z-score and t test
    1. Another esd test inside here
17. [Minimal sample size for seasonal forecasting](https://robjhyndman.com/papers/shortseasonal.pdf)
18. [Golden signals](https://www.usenix.org/conference/srecon19asia/presentation/chen-yu), [youtube](https://www.youtube.com/watch?v=3T9ZzQQiPSo)
19. [Graph-based Anomaly Detection and Description: A Survey](https://arxiv.org/pdf/1404.4679.pdf)
20. Time2vec, [paper](https://arxiv.org/pdf/1907.05321.pdf) (for deep learning, as a layer)

###

### Dynamic Time Warping (DTW)

This subsection covers DTW myths, code, and hierarchical clustering examples.

The same notes are in [Clustering Algorithms](clustering-algorithms.md) and [Distance](../data/feature-engineering.md#distance).

DTW, ie., how to compute a better distance for two time series.

1. The three myths of using DTW

Myth 1: The ability of DTW to handle sequences of different lengths is a great advantage, and therefore the simple lower bound that requires different-length sequences to be reinterpolated to equal length is of limited utility \[10]\[19]\[21]. In fact, as we will show, there is no evidence in the literature to suggest this, and extensive empirical evidence presented here suggests that comparing sequences of different lengths and reinterpolating them to equal length produce no statistically significant difference in accuracy or precision/recall.
Myth 2: Constraining the warping paths is a necessary evil that we inherited from the speech processing community to make DTW tractable, and that we should find ways to speed up DTW with no (or larger) constraints\[19]. In fact, the opposite is true. As we will show, the 10% constraint on warping inherited blindly from the speech processing community is actually too large for real world data mining.
Myth 3: There is a need (and room) for improvements in the speed of DTW for data mining applications. In fact, as we will show here, if we use a simple lower bounding technique, DTW is essentially O(n) for data mining applications. At least for CPU time, we are almost certainly at the asymptotic limit for speeding up DTW.

{% embed url="https://www.youtube.com/watch?v=_K1OsqCicBY" %}

2. [Python code](https://github.com/alexminnaar/time-series-classification-and-clustering) with a [good tutorial.](http://nbviewer.ipython.org/github/alexminnaar/time-series-classification-and-clustering/blob/master/Time%20Series%20Classification%20and%20Clustering.ipynb)
3. Another function for dtw distance in python
4. [Medium](https://medium.com/datadriveninvestor/dynamic-time-warping-dtw-d51d1a1e4afc), mentions prunedDTW, sparseDTW and fastDTW
5. [DTW in TSLEARN](https://tslearn.readthedocs.io/en/latest/user_guide/dtw.html#soft-dtw)
6. [DynamicTimeWarping](https://dynamictimewarping.github.io/py-api/html/api/dtw.dtw.html#dtw.dtw) git

<figure><img src="../.gitbook/assets/gimg-05a66ce85e85.png" alt=""><figcaption><p>Figure.</p><p>Credit: <a href="https://lh3.googleusercontent.com/aJYNIn2aCXwoZxIIcHbm-X03kwzJUqTBLQ96UPVUex_nRsO4eO1NuCWppkiMazcm5IQKUcnS9i2h2usU9GKLUAFIToRWxyx36W6SydTl4J1tVTd7vzLaywdvedmPSOQnmDj1sPZj">copied from the original hosted image</a>.</p></figcaption></figure>

1. (duplicate above in classification) [Stackexchange](https://stats.stackexchange.com/questions/131281/dynamic-time-warping-clustering/131284) - Yes, you can use DTW approach for classification and clustering of time series. I've compiled the following resources, which are focused on this very topic (I've recently answered a similar question, but not on this site, so I'm copying the contents here for everybody's convenience):

- UCR Time Series Classification/Clustering: main page, software page and corresponding paper
- Time Series Classification and Clustering with Python: a blog post
- Capital Bikeshare: Time Series Clustering: another blog post
- Time Series Classification and Clustering: [ipython notebook](http://nbviewer.ipython.org/github/alexminnaar/time-series-classification-and-clustering/blob/master/Time%20Series%20Classification%20and%20Clustering.ipynb)
- Dynamic Time Warping using rpy and Python: [another blog post](https://nipunbatra.wordpress.com/2013/06/09/dynamic-time-warping-using-rpy-and-python)
- Mining Time-series with Trillions of Points: Dynamic Time Warping at Scale: another blog post
- Time Series Analysis and Mining in R (to add R to the mix): yet another blog post
- And, finally, two tools implementing/supporting DTW, to top it off: R package and Python module

1. Time Series Hierarchical Clustering using Dynamic Time Warping in Python - notebook
2. K-Means with DTW, probably fixed length vectors, using tslearn
3. (nice) [With time series](https://medium.com/@shachiakyaagba_41915/dynamic-time-warping-with-time-series-1f5c05fb8950)

