# Fraud Detection

Fraud is rare and costly, so a model has to find a few bad events in a sea of normal ones. The page starts with a handbook and the objectives of fraud detection, moves to papers and methods such as autoencoders and graphs, then streaming tools, and ends with the data sets and how useful they look.

The same notes are in [Anomaly Detection](../predictive-ml/anomaly-detection.md).

The best entry point is a full book. [Machine Learning for Credit Card Fraud Detection](https://fraud-detection-handbook.github.io/fraud-detection-handbook/Foreword.html) is the Reproducible Machine Learning for Credit Card Fraud detection practical handbook, starting from its foreword, and its code — Practical Handbook [Git](https://github.com/Fraud-Detection-Handbook/fraud-detection-handbook) is the matching repository. The figure below is from the handbook.

<figure><img src="../.gitbook/assets/image (25).png" alt=""><figcaption><p>Machine Learning for Credit Card Fraud Detection — Practical Handbook.</p></figcaption></figure>

Before any model, the goal has to be stated. [Fraud detection objectives.](https://nethone.com/post/beginners-guide-to-machine-learning) is from Nethone, a vendor whose AI-powered user and risk analysis engine is built to detect fraud; the figure below shows those objectives.

<figure><img src="../.gitbook/assets/image (28).png" alt=""><figcaption><p>Fraud detection objectives.</p></figcaption></figure>

With the objective clear, the literature is the next stop. [Awesome fraud papers](https://github.com/benedekrozemberczki/awesome-fraud-detection-papers) is a curated list of data mining papers about fraud detection. Because fraud looks like an anomaly, one standard method is reconstruction error: [Credit card fraud using an autoencoder in Keras](https://github.com/curiousily/Credit-Card-Fraud-Detection-using-Autoencoders-in-Keras/blob/master/fraud_detection.ipynb) is a notebook and pre-trained model that builds a deep autoencoder in Keras for anomaly detection in credit card transactions data.

The same notes are in [AUTOENCODERS](../deep-learning/autoencoders.md#autoencoders).

Fraud also hides in relationships between accounts, which is where graphs come in. [Graph fraud papers](https://github.com/safe-graph/graph-fraud-detection-papers) is a curated list of Graph/Transformer-based fraud, anomaly, and outlier detection papers and resources.

The same notes are in [Graph Theory](../predictive-ml/graph-theory.md).

A method still has to run on live transactions. [Fraud using Flink](https://github.com/afedulov/fraud-detection-demo) is the repository for the Advanced Flink Application Patterns series, and its [docs](https://flink.apache.org/2020/01/15/advanced-flink-application-patterns-vol.1-case-study-of-a-fraud-detection-system/) are the first case study, a fraud detection system that teaches three streaming patterns: dynamic updates of application logic, dynamic data partitioning controlled at runtime, and low latency alerting based on custom windowing logic. For classic models side by side, [Credit card fraud on Kaggle](https://github.com/georgymh/ml-fraud-detection) does credit card fraud detection through logistic regression, k-means, and deep learning, and [Deep graph for fraud](https://github.com/safe-graph/DGFraud) is a deep graph-based toolbox for fraud detection. Close to the same money problem, [Transforming Financial Forecasting with Data Science and Machine Learning at Uber](https://www.uber.com/en-IL/blog/transforming-financial-forecasting-machine-learning/) is Uber's account of financial forecasting.

## Data sets

After the methods, the question is which data can actually test them, and the public sets are weaker than they look.

Credit card fraud detection: everyone hit 99%+, seems too easy.

The chargeback set is the opposite problem: questionable fraud data set, it has no usable features and as a time series it doesn't look too informative. [https://www.kaggle.com/dmirandaalves/predict-chargeback-frauds-payment](https://www.kaggle.com/dmirandaalves/predict-chargeback-frauds-payment)

## Deprecated links

{% hint style="warning" %}
These links and images no longer work. The original wording is kept here. A same-resource copy, when one was checked, is used above.
{% endhint %}

- Fraud detection on money pools, using social network & pool size, future optimizing using f-beta. This address no longer opens: https://towardsdatascience.com/frauddetection-f801b781410b
