# Timeseries

This page collects time-series forecasting, decomposition, stationarity, DTW, clustering, and anomaly-detection resources.

The same notes are in [Anomaly Detection](anomaly-detection.md), [Time series entropy](../data/information-theory.md#time-series-entropy), and [Timeseries](../appendix/data-science-tools.md#timeseries).

1. [Random walk](https://machinelearningmastery.com/gentle-introduction-random-walk-times-series-forecasting-python/) - what is?

<figure><img src="../.gitbook/assets/gimg-71b2444e4eb4.png" alt=""><figcaption><p>Figure.</p><p>Credit: <a href="https://lh6.googleusercontent.com/EIjqJgNQyohogF9eaHNDSpOJHaXag5MgHLlShTtkSHRaEU0EitX_ZPMbVDE2cbHr02bzT46Io9sJH7EkeTTrW49KMbBbYe6Xh9yFp2Tq_0LA-CZdb7X0ZZNvMs0k4hj8epypkKft">copied from the original hosted image</a>.</p></figcaption></figure>

1. [Time series decomposition book](https://otexts.com/fpp2/forecasting-decomposition.html) - stl x11 seats
2. [Mastery on ts decomposition](https://machinelearningmastery.com/decompose-time-series-data-trend-seasonality/)

### TOOLS

This section lists Python time-series libraries and feature-extraction tools.

1. SKtime - is a sk-based api, medium, integrates algos from tsfresh and tslearn
2. (really good) A LightGBM Autoregressor — Using Sktime, explains about the basics in time series prediction, splitting, next step, delayed step, multi step, deseason.
3. [SKtime-DL - using keras and DL](https://github.com/sktime/sktime-dl)
4. [TSFresh](http://tsfresh.readthedocs.io) - extracts 1200 features, filters them using FDR for time series classification etc
5. [TSlearn](http://tslearn.readthedocs.io) - DTW, shapes, shapelets (keras layer), time series kmeans/clustering/svm/svr/KNN/bary centers/PAA/SAX

<figure><img src="../.gitbook/assets/gimg-0700069a56bf.png" alt=""><figcaption><p>Figure.</p><p>Credit: <a href="https://lh5.googleusercontent.com/q4duc9XMnsYnOvbMeBcWLWf6T1uyPMrhBoPZEVVL16hS2UJJTalHA3MUE12kMo308fF1nO-qCGxeDefjvoLz106E7ZjkUTiFriggG98iX6H9vlaROGNnOdNpjEy6zZViK4Tl43mn">copied from the original hosted image</a>.</p></figcaption></figure>

6. [DTAIDistance](https://dtaidistance.readthedocs.io/en/latest/index.html) - Library for time series distances (e.g. Dynamic Time Warping) used in the [DTAI Research Group](https://dtai.cs.kuleuven.be/). The library offers a pure Python implementation and a faster implementation in C. The C implementation has only Cython as a dependency. It is compatible with Numpy and Pandas and implemented to avoid unnecessary data copy operations
   [dtaidistance.clustering.hierarchical](https://dtaidistance.readthedocs.io/en/latest/modules/clustering/hierarchical.html)
7. [Darts](https://unit8co.github.io/darts/) is a Python library for user-friendly forecasting and anomaly detection on time series. [Forecasting models](https://unit8co.github.io/darts/#forecasting-models) & [Examples](https://unit8co.github.io/darts/#example-usage)

[Ddtaidistance.clustering.kmeans](https://dtaidistance.readthedocs.io/en/latest/modules/clustering/kmeans.html)

[Dtaidistance.clustering.medoids](https://dtaidistance.readthedocs.io/en/latest/modules/clustering/medoids.html)

- Identify anomalies, outliers or abnormal behaviour (see for example the [anomatools package](https://github.com/Vincent-Vercruyssen/anomatools)).

<figure><img src="../.gitbook/assets/gimg-bf36052b3a66.png" alt=""><figcaption><p>Figure.</p><p>Credit: <a href="https://lh3.googleusercontent.com/7nxg_PC85TDLAnkrIt2lNm3VhLRcFKwlGlEZOd4Ua7UnPFGctGheUcyzzIwVW39N8cAW7fF8cwvMJUySX6K4rkQNz1C5kGRL5P4LIPB0lNUl9gIietACvRxm4nokLL1Chr57024F">copied from the original hosted image</a>.</p></figcaption></figure>

1. Semi supervised with DTAIDistance - Active semi-supervised clustering

The recommended method for perform active semi-supervised clustering using DTAIDistance is to use the COBRAS for time series clustering: [https://github.com/ML-KULeuven/cobras](https://github.com/ML-KULeuven/cobras). COBRAS is a library for semi-supervised time series clustering using pairwise constraints, which natively supports both dtaidistance.dtw and kshape.

1. [Affine warp](https://github.com/ahwillia/affinewarp), a neural net with time warping - as part of the following manuscript, which focuses on analysis of large-scale neural recordings (though this code can be also be applied to many other data types)
2. [Neural warp](https://github.com/josifgrabocka/neuralwarp) - [NeuralWarp](https://arxiv.org/pdf/1812.08306.pdf): Time-Series Similarity with Warping Networks

[A great introduction into time series](https://medium.com/making-sense-of-data/time-series-next-value-prediction-using-regression-over-a-rolling-window-228f0acae363) - “The approach is to come up with a list of features that captures the temporal aspects so that the auto correlation information is not lost.” basically tells us to take sequence features and create (auto)-correlated new variables using a time window, i.e., “Time series forecasts as regression that factor in autocorrelation as well.”. we can transform raw features into other type of features that explain the relationship in time between features. we measure success using loss functions, MAE RMSE MAPE RMSEP AC-ERROR-RATE

[Interesting idea](http://blog.kaggle.com/2016/02/03/rossmann-store-sales-winners-interview-2nd-place-nima-shahbazi/) on how to define ‘time series’ dummy variables that utilize beginning\end of certain holiday events, including important information on what NOT to filter even if it seems insignificant, such as zero sales that may indicate some relationship to many sales the following day.

<figure><img src="../.gitbook/assets/gimg-fef67e8b36ca.png" alt=""><figcaption><p>Figure.</p><p>Credit: <a href="https://lh6.googleusercontent.com/hUyX6IBOvCb8hrjHVG8edxDWmnHLwe6J2hf-_cGXhpSuhWAGPg7ahENwlXftItTY6kn1rw4GZxeGBwqJRa51XAQxTu4zZD_p_S93yCZvaXlU6QJJPV8jJHdq8HVVX88sOE95QBo_">copied from the original hosted image</a>.</p></figcaption></figure>

[Time series patterns: ](https://www.otexts.org/fpp/2/1)

- A trend (a,b,c) exists when there is a long-term increase or decrease in the data.
- A seasonal (a - big waves) pattern occurs when a time series is affected by seasonal factors such as the time of the year or the day of the week. The monthly sales induced by the change in cost at the end of the calendar year.
- A cycle (a) occurs when the data exhibit rises and falls that are not of a fixed period - sometimes years.

[Some statistical measures](https://www.otexts.org/fpp/2/2) (mean, median, percentiles, iqr, std dev, bivariate statistics - correlation between variables)

Bivariate Formula: this correlation measures the extent of a linear relationship between two variables. high number = high correlation between two variable. The value of r always lies between -1 and 1 with negative values indicating a negative relationship and positive values indicating a positive relationship. Negative = decreasing, positive = increasing.<figure><img src="../.gitbook/assets/gimg-49e1a9c9f846.png" alt=""><figcaption><p>Figure.</p><p>Credit: <a href="https://lh4.googleusercontent.com/POsJ_tINRsrxwTHCkx56YjkN9irF-Z3atalMhZobbKPqz2zVmNEUnmtXMwimpCAMcNSutHGtk8Nn5bJTLgURfqNxmmg6BzpE2ruf5hbDxBSwB_IIafOvoFbQQARwZDGvohhEIgvY">copied from the original hosted image</a>.</p></figcaption></figure>

But correlation can LIE, the following has 0.8 correlation for all of the graphs:

<figure><img src="../.gitbook/assets/gimg-6ef2dbb63a24.png" alt=""><figcaption><p>Figure.</p><p>Credit: <a href="https://lh6.googleusercontent.com/fsfNgKsimmzrD6j2OduCclfo9Facq9w6caU5fXNaGq3dBG6TdY1cDVOBDJ5eNGT7Sjwgi1PCrQtteuFnps1tWqKiTFXpUhUgrFVG_GT--CHZhyF2kOGpuYp9pRJjZmE8KTRFfPsA">copied from the original hosted image</a>.</p></figcaption></figure>

Autocorrelation measures the linear relationship between lagged values of a time series.

L8 is correlated, and has a high measure of 0.83

- White-noise has autocorrelation of 0.<figure><img src="../.gitbook/assets/gimg-0ff80aebaef9.png" alt=""><figcaption><p>Figure.</p><p>Credit: <a href="https://lh6.googleusercontent.com/G2xzLwQkZaWNkLQseUHw2A1PHdq5zx0en1EZRhIKfK8m4QdxFvZ0k5wDNZDj3xMDV8IygVeQRBAeRHEtrVCULdlr9HKRuP3cjNHFwT996Ul07FXP-e8SlDFCOSQiXPdWo01ldecp">copied from the original hosted image</a>.</p></figcaption></figure>

### [Forecasting methods](https://www.otexts.org/fpp/2/3)

This subsection lists simple benchmark forecasts from the forecasting textbook.

- Average: Forecasts of all future values are equal to the mean of the historical data.
- Naive: Forecasts are simply set to be the value of the last observation.
- Seasonal Naive: forecast to be equal to the last observed value from the same season of the year
- Drift: A variation on the naïve method is to allow the forecasts to increase or decrease over time, the drift is set to be the average change seen in the historical data.

### [Data Transformations](https://www.otexts.org/fpp/2/4)

This subsection links log, Box-Cox, and calendar adjustments for series.

- Log
- Box cox
- Back transform
- Calendrical adjustments
- Inflation adjustment

Transforming time series data to tabular (in order to use tabular based approach)

### SPLITTING TIME SERIES DATA

This subsection links time-series cross-validation with gaps.

1. SK-lego With a gap - now with even timeseries split by group

### [Evaluate forecast accuracy](https://www.otexts.org/fpp/2/5)

This subsection covers forecast accuracy metrics and dummy-variable pitfalls.

<figure><img src="../.gitbook/assets/gimg-4520aee72acc.png" alt=""><figcaption><p>Figure.</p><p>Credit: <a href="https://lh6.googleusercontent.com/-t5303-rJtTF8gUP5GRrHwx9gVJTaM5zObpxRFO5iD1jgkSC-qxX1Q8-7fPnP1cb9Vo3reKMtL5f_d41XvX0xjHxTlCtHOJ7i99aaHj7YLSa_vu4E5nCg1IQCWi5YyZvQt-O1TJ3">copied from the original hosted image</a>.</p></figcaption></figure>

- Dummy variables: sunday, monday, tues,wed,thurs, friday. NO SATURDAY!
- notice that only six dummy variables are needed to code seven categories. That is because the seventh category (in this case Sunday) is specified when the dummy variables are all set to zero. Many beginners will try to add a seventh dummy variable for the seventh category. This is known as the "dummy variable trap" because it will cause the regression to fail.
- Outliers: If there is an outlier in the data, rather than omit it, you can use a dummy variable to remove its effect. In this case, the dummy variable takes value one for that observation and zero everywhere else.
- Public holidays: For daily data, the effect of public holidays can be accounted for by including a dummy variable predictor taking value one on public holidays and zero elsewhere.
- Easter: is different from most holidays because it is not held on the same date each year and the effect can last for several days. In this case, a dummy variable can be used with value one where any part of the holiday falls in the particular time period and zero otherwise.
- Trading days: The number of trading days in a month can vary considerably and can have a substantial effect on sales data. To allow for this, the number of trading days in each month can be included as a predictor. An alternative that allows for the effects of different days of the week has the following predictors. # Mondays in month;# Tuesdays in month;# Sundays in month.
- Advertising: $advertising for previous month;$advertising for two months previously

### [Rolling window analysis](https://link.springer.com/chapter/10.1007%2F978-0-387-32348-0_9)

This subsection quotes rolling-window parameter stability checks.

“compute parameter estimates over a rolling window of a fixed size through the sample. If the parameters are truly constant over the entire sample, then the estimates over the rolling windows should not be too different. If the parameters change at some point during the sample, then the rolling estimates should capture this instability”

### [Moving average window](https://www.otexts.org/fpp/6/2)

This subsection describes moving-average trend–cycle estimation.

estimate the trend cycle

- 3-5-7-9? If its too large its going to flatten the curve, too low its going to be similar to the actual curve.
- two tier moving average, first 4 then 2 on the resulted moving average.

[Visual example](https://www.youtube.com/watch?v=_YXoRTQQI3U) of ARIMA algorithm - captures the time series trend or forecast.

### Decomposition

This subsection links seasonal basis construction and decomposition plots.

1. Creating curves to explain a complex seasonal fit.
2. <figure><img src="../.gitbook/assets/gimg-5f3d766110b9.png" alt=""><figcaption><p>Figure.</p><p>Credit: <a href="https://lh6.googleusercontent.com/dnGmy5HVE3eGuaObKDMwyoabjNYBFQX_qXgoppg2hCIRHPttAYPVXCDl5qEIVmoMQk-74JGr_ol58rv-ScpTEC7bQgn8nEI2cjFj0a74qZLS47sNQQXEeHzLb0XGylhsa-uilNgs">copied from the original hosted image</a>.</p></figcaption></figure>
3. <figure><img src="../.gitbook/assets/gimg-3c78cf25bcb4.png" alt=""><figcaption><p>Figure.</p><p>Credit: <a href="https://lh6.googleusercontent.com/8iRS-_lbtCP1bVVwihykUC_Kw3LlzoyrAWika8cvNQ2UTw3bUZXqbmz7vb4N5GYgl9ne2QnDlJKU1pb9OwtFRqiAVB-XYmV7dchKF0kib0Mqe2AAHmMeCEyJxjzjPXcUuoK6T3pI">copied from the original hosted image</a>.</p></figcaption></figure>

### Weighted “window”

This subsection links scikit-lego decayed estimation for weighted windows.

1, scikit-lego with a decay estimator

<figure><img src="../.gitbook/assets/gimg-c7749fe7f90f.png" alt=""><figcaption><p>Figure.</p><p>Credit: <a href="https://lh5.googleusercontent.com/LKgjfaw-oOaNI7with2FnVnDSoR2LXpzN2boi3cM29HsfzELrpk6FTId0nZj1JnnTI79SBnQgLlM10Awyz7eFC8jEEaUMtax-BowEK3QFKyJEv3P-LnCrrNP9CdSxii1u2d_GEl9">copied from the original hosted image</a>.</p></figcaption></figure>

### [Time Series Components](http://machinelearningmastery.com/time-series-forecasting/)

This subsection lists level, trend, seasonality, and noise in series.

1. Level. The baseline value for the series if it were a straight line.
2. Trend. The optional and often linear increasing or decreasing behavior of the series over time.
3. Seasonality. The optional repeating patterns or cycles of behavior over time.
4. Noise. The optional variability in the observations that cannot be explained by the model.

All time series have a level, most have noise, and the trend and seasonality are optional.

One step forecast using a window of “1” and a typical sample “time, measure1, measure2”:

- linear/nonlinear classifiers: predict a single output value - using the t-1 previous line, i.e., “measure1 t, measure 2 t, measure 1 t+1, measure 2 t+1 (as the class)”
- Neural networks: predict multiple output values, i.e., “measure1 t, measure 2 t, measure 1 t+1(class1), measure 2 t+1(class2)”

One-Step Forecast: This is where the next time step (t+1) is predicted.

Multi-Step Forecast: This is where two or more future time steps are to be predicted.

Multi-step forecast using a window of “1” and a typical sample “time, measure1”, i.e., using the current value input we label it as the two future input labels:

- “measure1 t, measure1 t+1(class), measure1 t+2(class1)”

This article explains about ML Methods for Sequential Supervised Learning - Six methods that have been applied to solve sequential supervised learning problems:

The same notes are in [CONDITIONAL RANDOM FIELDS (CRF)](probabilistic-models.md#conditional-random-fields-crf) and [MARKOV MODELS / HIDDEN MARKOV MODEL](probabilistic-models.md#markov-models--hidden-markov-model).

1. sliding-window methods - converts a sequential supervised problem into a classical supervised problem
2. recurrent sliding windows
3. hidden Markov models
4. maximum entropy Markov models
5. input-output Markov models
6. conditional random fields
7. graph transformer networks

##

### STATIONARY TIME SERIES

This subsection defines stationarity and differencing to remove trend and seasonality.

[What is?](https://machinelearningmastery.com/time-series-data-stationary-python/) A time series without a trend or seasonality, in other words non-stationary has a trend or seasonality

There are ways to [remove the trend and seasonality](https://machinelearningmastery.com/difference-time-series-dataset-python/), i.e., take the difference between time points.

1. T+1 - T
2. Bigger lag to support seasonal changes
3. pandas.diff()
4. Plot a histogram, plot a log(X) as well.
5. Test for the unit root null hypothesis - i.e., use the Augmented dickey fuller test to determine if two samples originate in a stationary or a non-stationary (seasonal/trend) time series

Shay on stationary time series, AR, ARMA

(amazing) [STL](https://otexts.com/fpp2/stl.html) and more.

### SHORT TIME SERIES

This subsection lists short-series forecasting advice and pmdarima.

1. [Short time series](https://robjhyndman.com/hyndsight/short-time-series/)
2. PDarima - Pmdarima‘s auto_arima function is extremely useful when building an ARIMA model as it helps us identify the most optimal p,d,q parameters and return a fitted ARIMA model.
3. [Min sample size for short seasonal time series](https://robjhyndman.com/papers/shortseasonal.pdf)
4. [More mastery on short time series.](https://machinelearningmastery.com/time-series-forecasting-methods-in-python-cheat-sheet/)
   1. Autoregression (AR)
   2. Moving Average (MA)
   3. Autoregressive Moving Average (ARMA)
   4. Autoregressive Integrated Moving Average (ARIMA)
   5. Seasonal Autoregressive Integrated Moving-Average (SARIMA)
   6. Seasonal Autoregressive Integrated Moving-Average with Exogenous Regressors (SARIMAX)
   7. Vector Autoregression (VAR)
   8. Vector Autoregression Moving-Average (VARMA)
   9. Vector Autoregression Moving-Average with Exogenous Regressors (VARMAX)
   10. Simple Exponential Smoothing (SES)
   11. Holt Winter’s Exponential Smoothing (HWES)

Predicting actual Values of time series using observations

1. [Using kalman filters](https://www.youtube.com/watch?v=CaCcOwJPytQ) - explains the concept etc, 1 out of 55 videos.

### [Kalman filters in matlab](https://www.youtube.com/watch?v=4OerJmPpkRg)

This subsection links a MATLAB-oriented Kalman filter video.

### [LTSM for time series](http://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/)

This subsection describes LSTM gates and sunspot prediction walkthroughs.

The same notes are in [Deep Neural Time Series](../deep-learning/deep-neural-time-series.md) and [LSTM](../deep-learning/recurrent-nets.md#lstm).

There are three types of gates within a unit:

- Forget Gate: conditionally decides what information to throw away from the block.
- Input Gate: conditionally decides which values from the input to update the memory state.
- Output Gate: conditionally decides what to output based on input and the memory of the block.

Using lstm to predict sun spots, has some autocorrelation usage

- [Part 1](https://www.business-science.io/timeseries-analysis/2018/04/18/keras-lstm-sunspots-time-series-prediction.html)
- [Part 2](https://www.business-science.io/timeseries-analysis/2018/07/01/keras-lstm-sunspots-part2.html)

### CLASSIFICATION

This subsection compiles DTW-based time-series classification resources.

1. [Stackexchange](https://stats.stackexchange.com/questions/131281/dynamic-time-warping-clustering/131284) - Yes, you can use DTW approach for classification and clustering of time series. I've compiled the following resources, which are focused on this very topic (I've recently answered a similar question, but not on this site, so I'm copying the contents here for everybody's convenience):

- UCR Time Series Classification/Clustering: main page, software page and corresponding paper
- Time Series Classification and Clustering with Python: a blog post
- Capital Bikeshare: Time Series Clustering: another blog post
- Time Series Classification and Clustering: [ipython notebook](http://nbviewer.ipython.org/github/alexminnaar/time-series-classification-and-clustering/blob/master/Time%20Series%20Classification%20and%20Clustering.ipynb)
- Dynamic Time Warping using rpy and Python: [another blog post](https://nipunbatra.wordpress.com/2013/06/09/dynamic-time-warping-using-rpy-and-python)
- Mining Time-series with Trillions of Points: Dynamic Time Warping at Scale: another blog post
- Time Series Analysis and Mining in R (to add R to the mix): yet another blog post
- And, finally, two tools implementing/supporting DTW, to top it off: R package and Python module

