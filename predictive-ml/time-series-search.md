# Time Series Search

Searching, grouping, and monitoring time series all depend on one question: how far apart are two series? The page starts with Dynamic Time Warping (DTW), the distance that answers it, along with its myths and code, then uses that distance for clustering series, and ends with anomaly detection on time series and the toolkits for it.

### Dynamic Time Warping (DTW)

Euclidean distance compares two series point by point and breaks as soon as one is shifted or stretched, so the first tool is a better distance. The same notes are in [Clustering Algorithms](clustering-algorithms.md) and [Distance](../data/feature-engineering.md#distance).

DTW, ie., how to compute a better distance for two time series. Before using it, it is worth knowing the three myths of using DTW, from the paper [http://alumni.cs.ucr.edu/~ratana/RatanamC.pdf](http://alumni.cs.ucr.edu/~ratana/RatanamC.pdf). The paper recalls that DTW has long been known in the speech recognition community, where it allows a non-linear mapping of one signal to another by minimizing the distance between the two, and that it was introduced into data mining for classification, clustering, and anomaly detection on time series. The three myths:

Myth 1: The ability of DTW to handle sequences of different lengths is a great advantage, and therefore the simple lower bound that requires different-length sequences to be reinterpolated to equal length is of limited utility \[10]\[19]\[21]. In fact, as we will show, there is no evidence in the literature to suggest this, and extensive empirical evidence presented here suggests that comparing sequences of different lengths and reinterpolating them to equal length produce no statistically significant difference in accuracy or precision/recall.

Myth 2: Constraining the warping paths is a necessary evil that we inherited from the speech processing community to make DTW tractable, and that we should find ways to speed up DTW with no (or larger) constraints\[19]. In fact, the opposite is true. As we will show, the 10% constraint on warping inherited blindly from the speech processing community is actually too large for real world data mining.

Myth 3: There is a need (and room) for improvements in the speed of DTW for data mining applications. In fact, as we will show here, if we use a simple lower bounding technique, DTW is essentially O(n) for data mining applications. At least for CPU time, we are almost certainly at the asymptotic limit for speeding up DTW.

The video below walks through DTW.

{% embed url="https://www.youtube.com/watch?v=_K1OsqCicBY" %}

To run it, the [Python code](https://github.com/alexminnaar/time-series-classification-and-clustering) repo holds time series classification and clustering code written in Python, with a [good tutorial.](http://nbviewer.ipython.org/github/alexminnaar/time-series-classification-and-clustering/blob/master/Time%20Series%20Classification%20and%20Clustering.ipynb) in the notebook viewer. For another function for dtw distance in python, Chathurangi Shyalika's [Medium](https://medium.com/datadriveninvestor/dynamic-time-warping-dtw-d51d1a1e4afc) post on Dynamic Time Warping (DTW) starts from what a time series is, a series of data points indexed in time order, and mentions prunedDTW sparseDTW and fastDTW. Library versions are [DTW in TSLEARN](https://tslearn.readthedocs.io/en/latest/user_guide/dtw.html#soft-dtw), the tslearn user guide chapter on Dynamic Time Warping that opens at soft-DTW, and [DynamicTimeWarping](https://dynamictimewarping.github.io/py-api/html/api/dtw.dtw.html#dtw.dtw), the dtw function in the dtw-python package documentation. The figure shows a warping alignment.

<figure><img src="../.gitbook/assets/gimg-05a66ce85e85.png" alt=""><figcaption><p>Figure.</p><p>Credit: <a href="https://lh3.googleusercontent.com/aJYNIn2aCXwoZxIIcHbm-X03kwzJUqTBLQ96UPVUex_nRsO4eO1NuCWppkiMazcm5IQKUcnS9i2h2usU9GKLUAFIToRWxyx36W6SydTl4J1tVTd7vzLaywdvedmPSOQnmDj1sPZj">copied from the original hosted image</a>.</p></figcaption></figure>

Once the distance exists, it can drive classification and clustering. The following answer is a (duplicate above in classification) of the forecasting page: [Stackexchange](https://stats.stackexchange.com/questions/131281/dynamic-time-warping-clustering/131284) - Yes, you can use DTW approach for classification and clustering of time series. I've compiled the following resources, which are focused on this very topic (I've recently answered a similar question, but not on this site, so I'm copying the contents here for everybody's convenience):

- UCR Time Series Classification/Clustering: main page, software page and corresponding paper
- Time Series Classification and Clustering with Python: a blog post
- Capital Bikeshare: Time Series Clustering: another blog post
- Jupyter Notebook Viewer. Time Series Classification and Clustering: [ipython notebook](http://nbviewer.ipython.org/github/alexminnaar/time-series-classification-and-clustering/blob/master/Time%20Series%20Classification%20and%20Clustering.ipynb)
- Dynamic Time Warping using rpy and Python:. [another blog post](https://nipunbatra.wordpress.com/2013/06/09/dynamic-time-warping-using-rpy-and-python)
- Mining Time-series with Trillions of Points: Dynamic Time Warping at Scale: another blog post
- Time Series Analysis and Mining in R (to add R to the mix): yet another blog post
- And, finally, two tools implementing/supporting DTW, to top it off: R package and Python module

Several of those answers have their own addresses. The main page is Welcome to the UCR Time Series Classification/Clustering Page, [http://www.cs.ucr.edu/~eamonn/time_series_data](http://www.cs.ucr.edu/~eamonn/time_series_data). The software page is the UCR Suite for Time Series Subsequence Search, [http://www.cs.ucr.edu/~eamonn/UCRsuite.html](http://www.cs.ucr.edu/~eamonn/UCRsuite.html). The corresponding paper is Searching and Mining Trillions of Time Series Subsequences under Dynamic Time Warping, [http://www.cs.ucr.edu/~eamonn/SIGKDD_trillion.pdf](http://www.cs.ucr.edu/~eamonn/SIGKDD_trillion.pdf). Time Series Classification and Clustering with Python –, a blog post, is at [http://alexminnaar.com/2014/04/16/Time-Series-Classification-and-Clustering-with-Python.html](http://alexminnaar.com/2014/04/16/Time-Series-Classification-and-Clustering-with-Python.html). Capital Bikeshare: Time Series Clustering, another blog post, shows that weekdays on the bike share network are very different from weekends: [http://ofdataandscience.blogspot.com/2013/03/capital-bikeshare-time-series-clustering.html](http://ofdataandscience.blogspot.com/2013/03/capital-bikeshare-time-series-clustering.html).

Two more clustering walkthroughs follow. Time Series Hierarchical Clustering using Dynamic Time Warping in Python - notebook, is kept at the end of the page along with its notebook. K-Means with DTW, probably fixed length vectors, using tslearn is at [https://towardsdatascience.com/how-to-apply-k-means-clustering-to-time-series-data-28d04a8f7da3](https://towardsdatascience.com/how-to-apply-k-means-clustering-to-time-series-data-28d04a8f7da3). (nice) [With time series](https://medium.com/@shachiakyaagba_41915/dynamic-time-warping-with-time-series-1f5c05fb8950) is Shachia Kyaagba's Dynamic Time Warping with Time Series, which starts from everyday series such as temperatures, commodity prices, and school grades.

### CLUSTERING TS

Those walkthroughs cluster whole series; clustering pieces of one series is where it goes wrong. The same notes are in [Clustering Algorithms](clustering-algorithms.md).

Clustering time series, subsequences with a rolling window, the pitfall., is the warning; that article is kept at the end of the page. For whole series, [Clustering using tslearn](https://tslearn.readthedocs.io/en/stable/user_guide/clustering.html) is the Time Series Clustering chapter of the tslearn documentation. When series differ in length, [Kmeans for variable length](https://medium.com/@iliazaitsev/how-to-classify-a-dataset-with-observations-of-various-length-96fab8e95baf) is How to Use K-means to Classify a Dataset with Non-Fixed Number of Features, built on the Wrist-Worn Accelerometer Dataset of 839 observations in 14 action classes such as Walking, Drinking, and Eating. Its code is the [notebook](https://github.com/devforfu/Blog/blob/master/trees/scikit_learn.py) in the i-zaitsev/Blog repo of projects supporting blog posts.

### ANOMALY DETECTION TS

With distances and clusters in hand, the last job is spotting the points or windows that do not fit. The same notes are in [Anomaly Detection](anomaly-detection.md).

Most methods assume stationarity first, and [What is stationary (process](https://en.wikipedia.org/wiki/Stationary_process) is the Wikipedia article on a stationary process. The forecasting baselines are in [mastery on arimas](https://machinelearningmastery.com/time-series-forecasting-methods-in-python-cheat-sheet/), MachineLearningMastery's 11 Classical Time Series Forecasting Methods in Python (Cheat Sheet). TS anomaly algos (stl, trees, arima) used to be linked here and is kept at the end.

A three-part series introduces the techniques. [AD techniques](https://medium.com/dp6-us-blog/anomaly-detection-techniques-c3817e8e7b2f) is the first part, part [2](https://medium.com/dp6-us-blog/anomaly-detection-techniques-part-ii-9a08b6562619) covers the types of anomalies found in digital marketing and an improvement on the earlier technique, the modified Z-score, and part [3](https://medium.com/dp6-us-blog/anomaly-detection-techniques-part-iii-d27e7b0d6c8a) asks how to apply those two techniques in a daily routine. [Z-score, modified z-score and iqr an intro why z-score is not robust](http://colingorrie.github.io/outlier-detection.html) is Colin Gorrie's Three ways to detect outliers, a collection of functions meant to make outlier detection in exploratory analysis accessible to non-specialists.

For a toolkit, [Adtk](https://adtk.readthedocs.io/en/stable/userguide.html) a sklearn-like toolkit with an amazing intro, various algorithms for non seasonal and seasonal, transformers, ensembles. [Awesome TS anomaly detection](https://github.com/rob-med/awesome-TS-anomaly-detection) is a list of tools & datasets for anomaly detection on time-series data. [Transfer learning toolkit](https://github.com/FuzhenZhuang/Transfer-Learning-Toolkit) is the Transfer Learning Toolkit for Primary Researchers, and its [paper and benchmarks](https://arxiv.org/pdf/1911.08967.pdf) is Transfer Learning Toolkit: Primers and Benchmarks.

A simpler approach fits a model while ignoring outliers. [Ransac is a good baseline](https://medium.com/@iamhatesz/random-sample-consensus-bd2bb7b1be75) random sample consensus for outlier detection; Tomasz Wrona's Random sample consensus motivates it with readings from a weather station that you want to fit a model to. [Ransac](https://medium.com/@angel.manzur/got-outliers-ransac-them-f12b6b5f606e) is another introduction, and [2](https://medium.com/@saurabh.dasgupta1/outlier-detection-using-the-ransac-algorithm-de52670adb4a) is Outlier detection using the RANSAC algorithm, which describes RANSAC as an iterative, non-deterministic algorithm for eliminating outliers that is commonly used in computer vision. Items 3 and 4 of that run are kept at the end. You can feed ransac with tsfresh/tslearn features.

[Anomaly detection for time series](https://medium.com/@jetnew/anomaly-detection-of-time-series-data-e0cb6b382e33) is Jet New's Anomaly Detection of Time Series Data, which opens with the Wikipedia definition: anomaly detection (also outlier detection) is the identification of rare items, events or observations which raise suspicions by differing significantly from the majority of the data. AD for TS, recommended by DTAIDistance, is [anomatools](https://github.com/Vincent-Vercruyssen/anomatools), a toolbox for anomaly detection.

STL decomposition is the next family. [AD where anomalies coincide with seasonal peaks!!](https://medium.com/@richa.mishr01/anomaly-detection-in-seasonal-time-series-where-anomalies-coincide-with-seasonal-peaks-9859a6a6b8ba) handles the hard case in seasonal time series. [AD challenges, stationary, seasonality, trend](https://cloudfabrix.com/blog/aiops/anomaly-detection-time-series-data/) is Ravi Teja Verma's Anomaly Detection and Typical Challenges with Time Series Data, on how to detect anomalies in time-series data with AIOPS. [Rt anomaly detection for time series pinterest](https://medium.com/pinterest-engineering/building-a-real-time-anomaly-detection-system-for-time-series-at-pinterest-a833e6856ddd) using stl decomposition is Pinterest Engineering's post by Kevin Chen and Brian Overstreet. A further STL post, AD, used to be linked here and is kept at the end.

Sliding windows are the other building block. [Solving sliding window problems](https://medium.com/outco/how-to-solve-sliding-window-problems-28d67601a66) is Sergey Piterman's How to Solve Sliding Window Problems, the interview-style problem family that is a subset of dynamic programming. [Rolling window regression](https://medium.com/making-sense-of-data/time-series-next-value-prediction-using-regression-over-a-rolling-window-228f0acae363) applies the window to next-value prediction.

ARIMA residuals are a common detector too. Forecasting using Arima 1 [2](http://alkaline-ml.com/pmdarima/) is pmdarima: ARIMA estimators for Python. Auto arima 1 [2](https://stackoverflow.com/questions/22770352/auto-arima-equivalent-for-python) is the Stack Overflow question auto.arima() equivalent for python, asked because statsmodels had no function for tuning the order (p,d,q) the way R's forecast::auto.arima() does. [3](https://www.analyticsvidhya.com/blog/2018/08/auto-arima-time-series-modeling-python-r/) is Aishwarya Singh's Build High Performance Time Series Models using Auto ARIMA in Python and R, a basic introduction to various time series forecasting methods and techniques.

For statistical tests, [Twitters ESD test](https://medium.com/@elisha_12808/time-series-anomaly-detection-with-twitters-esd-test-50cce409ced1) for outliers, using z-score and t test. Another esd test inside here, kept at the end of the page. Seasonal methods need enough history, so see [Minimal sample size for seasonal forecasting](https://robjhyndman.com/papers/shortseasonal.pdf).

In production, [Golden signals](https://www.usenix.org/conference/srecon19asia/presentation/chen-yu) is Yu Chen's Anomaly Detection on Golden Signals at USENIX, and the talk is on [youtube](https://www.youtube.com/watch?v=3T9ZzQQiPSo) as SREcon19 Asia/Pacific - Anomaly Detection on Golden Signals. Beyond single series, [Graph-based Anomaly Detection and Description: A Survey](https://arxiv.org/pdf/1404.4679.pdf) covers graphs. Finally, Time2vec [paper](https://arxiv.org/pdf/1907.05321.pdf), Time2Vec: Learning a Vector Representation of Time, is Time2vec for deep learning layer.

## Deprecated links

{% hint style="warning" %}
These links and images no longer work. The original wording is kept here. A same-resource copy, when one was checked, is used above.
{% endhint %}
- Clustering time series, subsequences with a rolling window, the pitfall.. This address no longer opens: https://towardsdatascience.com/dont-make-this-mistake-when-clustering-time-series-data-d9403f39bbb2
- TS anomaly algos (stl, trees, arima). This address no longer opens: https://blog.statsbot.co/time-series-anomaly-detection-algorithms-1cef5519aef2
- 3. This address no longer opens: https://towardsdatascience.com/detecting-the-fault-line-using-k-mean-clustering-and-ransac-9a74cb61bb96
- 4. This address no longer opens: http://www.cs.tau.ac.il/~turkel/imagepapers/RANSAC4Dummies.pdf
- here. This address no longer opens: https://towardsdatascience.com/anomaly-detection-def662294a4e
- Time Series Hierarchical Clustering using Dynamic Time Warping in Python. This address no longer opens: https://towardsdatascience.com/time-series-hierarchical-clustering-using-dynamic-time-warping-in-python-c8c9edf2fda5
- notebook. This address no longer opens: https://github.com/avchauzov/_articles/blob/master/1.1.trajectoriesClustering.ipynb
- AD. This address no longer opens: https://medium.com/wwblog/anomaly-detection-using-stl-76099c9fd5a7
