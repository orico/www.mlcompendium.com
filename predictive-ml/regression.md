# Regression

When the target is a continuous number rather than a class, the questions become which regressor to fit and how to judge its error. The page first walks regression libraries and algorithms, from sk-lego and lightning through CART, SVR, neural regression, and logistic regression, then kernel regression, and ends with the error metrics used to judge them, R2, RMSE, and related measures.

The same notes are in [Feature Types](../data/feature-types.md), [Normalization & Scaling](../data/normalization-and-scaling.md), [NYC TAXI](../ai-product/examples.md), [REGRESSION ALGORITHMS](#regression-algorithms), and [SUPPORT VECTOR REGRESSION (SVR)](linear-separator-algorithms.md#support-vector-regression-svr).

### REGRESSION ALGORITHMS

The algorithms come first, starting with tools that bend a linear regressor to fit data that is not linear. The same notes are in [Regression](regression.md) and [SUPPORT VECTOR REGRESSION (SVR)](linear-separator-algorithms.md#support-vector-regression-svr).

Sk-lego is used to fit with intervals a linear regressor on top of non linear data (its interval encoder page is kept at the end of this page). The figure shows that fit, copied from the original hosted image.

<figure><img src="../.gitbook/assets/gimg-cda337febb72.png" alt=""><figcaption><p>Figure.</p><p>Credit: <a href="https://lh6.googleusercontent.com/7yCwBKFpFonYWiaBrAy1AeM10-3YMc_HJayDR9-whuLp3K5TRxoIVeyP8EJqqQeO0MImgFpQFGuLa3mVo0tr-390ns4dErivP7jDNsE7NaJXo5k2l6Od4aJpKLrzpM1lZ73USG_Y">copied from the original hosted image</a>.</p></figcaption></figure>

The same library can also force the fit to be monotonic, Sk-lego monotonic:

<figure><img src="../.gitbook/assets/gimg-7e832f5c64c1.png" alt=""><figcaption><p>Figure.</p><p>Credit: <a href="https://lh3.googleusercontent.com/P1FIn55eoT2vzJ86cyyFMLklCph_Sk0KsFJiMgH4VMYstg9iED7hOP8fR8lVt9u5e0nVXsc8wTvb5iX3BgePkGY7p6BkHkDsyVywRZHWKNOpMJGSiJFFBGzkB3j76MHypzlwxE4g">copied from the original hosted image</a>.</p></figcaption></figure>

When the data is large, [Lightning](https://github.com/scikit-learn-contrib/lightning) - lightning is a library for large-scale linear classification, regression and ranking in Python.

<figure><img src="../.gitbook/assets/gimg-8b37986fda51.png" alt=""><figcaption><p>Figure.</p><p>Credit: <a href="https://lh6.googleusercontent.com/IP4Qg9ynzzWdjcFVqiy9TJfOzX7l8_9t8upL8ORVj4zHie6p1GKnuOoWBvth6yXCBQjmGi6W8wXVNPfBQkNwJqdo29TB6y3YTe23PsMOwgES9uF6U_8iGaYu8jHvmG2zvjriT3QV">copied from the original hosted image</a>.</p></figcaption></figure>

Linear regression itself is marked TBC. Beyond it, most classifiers have a regression twin. CART - [classification and regression tree](http://www.simafore.com/blog/bid/62482/2-main-differences-between-classification-and-regression-trees), basically the diff between classification and regression trees - instead of IG we use sum squared error.

The same notes are in [CART TREES](decision-trees.md#cart-trees).

SVR is regression based svm, with kernel only. [NNR](https://deeplearning4j.org/linear-regression)- regression based NN, one output node, points at the Eclipse Deeplearning4j project.

Logistic regression is the odd one out, a regressor used to classify. [LOGREG](http://www.statisticssolutions.com/what-is-logistic-regression/) - Logistic regression - is used as a classification algo to describe data and to explain the relationship between one dependent binary variable and one or more nominal, ordinal, interval or ratio-level independent variables. Output is BINARY. I.e., If the likelihood of killing the bug is > 0.5 it is assumed dead, if it is < 0.5 it is assumed alive. It assumes a binary outcome, assumes no outliers, and assumes no intercorrelations among predictors (inputs?).

Regression Measurements:

Once a regressor is fitted, the first number people check is R^2, and it can mislead. The Minitab blog post on five reasons why your R-squared can be too high lists [several reasons it can be too high.](http://blog.minitab.com/blog/adventures-in-statistics-2/five-reasons-why-your-r-squared-can-be-too-high) The author's short version: too many variables, overfitting, and in time series, seasonality trends can cause this. The other common pair is [RMSE vs MAE](https://medium.com/human-in-a-machine-world/mae-and-rmse-which-metric-is-better-e60ac3bde13d), a post on which of those two most common accuracy metrics for continuous variables is better, noting that publications now mostly use RMSE or some version of R-squared.

#### KERNEL REGRESSION

All of the above fit one global function; kernel regression instead predicts each point from its neighbours. [Gaussian Kernel Regression](http://mccormickml.com/2014/02/26/kernel-regression/) does–it takes a weighted average of the surrounding points.

The key parameter is the variance, sigma^2. Informally, this parameter will control the smoothness of your approximated function. Smaller values of sigma will cause the function to overfit the data points, while larger values will cause it to underfit. There is a proposed method to find sigma in the post! Gaussian Kernel Regression is equivalent to creating an RBF Network with the following properties: - described in the post, and drawn in the figure below, copied from the original hosted image.

<figure><img src="../.gitbook/assets/gimg-1489877d6d16.png" alt=""><figcaption><p>Gaussian kernel regression.</p><p>Credit: <a href="https://lh4.googleusercontent.com/V9zIvIq9putPPvzrwOOSayDsZllNCgwMhMvYNBu2rSYGSLFI9LfIxzjMWy2Z0wSw4T1CwOqQBd5qX45pgAq4lpfUbMR0CiGmu5rec38RTusLA1Fg5XaqqPZ3D4zvIQoR2Kb5w8fb">copied from the original hosted image</a>.</p></figcaption></figure>

## Metrics

Whatever regressor is chosen, it is judged by an error metric, and the first one is the coefficient of determination, [R2](https://en.wikipedia.org/wiki/Coefficient_of_determination). The Medium reading on the rest is a short series: [1](https://medium.com/data-science/regression-an-explanation-of-regression-metrics-and-what-can-go-wrong-a39a9793d914) is Divyanshu Mishra's explanation of regression metrics and what can go wrong, [2](https://medium.com/@george.drakos62/how-to-select-the-right-evaluation-metric-for-machine-learning-models-part-1-regrression-metrics-3606e25beae0) and [3](https://medium.com/@george.drakos62/how-to-select-the-right-evaluation-metric-for-machine-learning-models-part-2-regression-metrics-d4a1a9ba3d74) are parts 1 and 2 of how to select the right evaluation metric for regression, and [4](https://medium.com/usf-msds/choosing-the-right-metric-for-machine-learning-models-part-1-a99d7d7414e4) is Alvira Swalin's part 1 of choosing the right metric for evaluating machine learning models.

## Deprecated links

{% hint style="warning" %}
These links and images no longer work. The original wording is kept here. A same-resource copy, when one was checked, is used above.
{% endhint %}

- Medium 1. This address no longer opens: https://towardsdatascience.com/regression-an-explanation-of-regression-metrics-and-what-can-go-wrong-a39a9793d914
3. Tutorial. This address no longer opens: https://www.dataquest.io/blog/understanding-regression-error-metrics/
- Sk-lego. This address no longer opens: https://scikit-lego.readthedocs.io/en/latest/preprocessing.html#Interval-Encoders
