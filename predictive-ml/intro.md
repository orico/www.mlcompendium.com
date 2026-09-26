# Predictive ML

Each classical learner is one inductive bias, and each bias uses the one before it.
After this chapter the reader can pick a linear model, a tree, a cluster, a forecast, or a graph method and say what it assumes.
The chapter moves from supervised learners with a label, to unsupervised methods without one, to signals that unfold in time, and ends with graphs.

## Supervised

The supervised part starts before there is a label at all. [Data Mining](data-mining.md) is about association rules, Apriori, FP-Growth, and how to read support, confidence, and lift when the goal is frequent itemsets rather than a prediction. The first real boundary between classes comes next: [Linear Separator Algorithms](linear-separator-algorithms.md) covers support vector machines and related linear separators, with kernels, SMO, libsvm vs liblinear, and overfitting advice. The opposite bias, trusting whatever is close, is [Nearest Neighbors](nearest-neighbors.md), the scalable nearest-neighbour libraries and notes on choosing k.

Once a model can fit, it can also fit noise. [Regularization](regularization.md) is the penalty added to a loss so that such a model is discouraged, with norms, L1/L2 sparsity intuition, and priors. [Probabilistic Models](probabilistic-models.md) replaces geometry with probability: naive Bayes, Bayesian networks, Markov models, HMMs, IOHMM, and CRF. When the target is continuous rather than a class, [Regression](regression.md) covers the algorithms, kernel regression, and the error metrics used to judge them.

Trees give a different bias again. [Decision Trees](decision-trees.md) covers Hellinger splits, CART, KD-trees, random forests, and extremely randomized trees, and [Ensembles](ensembles.md) then combines many models through voting, bagging, boosting, and stacking, up to gradient boosting, XGBoost, and CatBoost. The last two pages evolve the learner itself: [Genetic Algorithms & Genetic Programming](genetic-algorithms-and-genetic-programming.md) contrasts evolving parameters with evolving programs, and [Learning Classifier Systems](learning-classifier-systems.md) defines LCS and XCS, where the learner is a rule population rather than a single model.

## Unsupervised

Without a label, the question becomes what structure is already in the data. [Clustering Algorithms](clustering-algorithms.md) groups unlabeled points, from k-means and GMM through density-based and constrained clustering. [Dimensionality Reduction Methods](dimensionality-reduction-methods.md) projects high-dimensional data with PCA, SVD, kernel PCA, LDA, ICA, LSA, and manifold methods such as t-SNE. [Anomaly Detection](anomaly-detection.md) separates novelty from outlier detection and runs from elliptic envelopes and LOF through isolation forests and one-class SVM.

Logs are a common unlabeled source of their own. [Process Mining](process-mining.md) recovers the real process from event logs when the documented one is incomplete or ignored, and [Log Parsing / Templatization](templatization.md) turns raw log lines into templates and uses them for log-based anomaly detection.

## Signals

Some data has an order in time, and that order is the signal. [Forecasting](forecasting.md) covers time-series components, stationarity, decomposition, forecasting methods, and accuracy. [Time Series Search](time-series-search.md) compares series with DTW, then clusters them and detects anomalies in them. [Fourier Transform](fourier-transform.md) gives the frequency-domain view of a signal, plus wavelets, before modeling, and [Digital Signal Processing (DSP)](digital-signal-processing-dsp.md) is the toolbox shelf of SciPy, Librosa, and beat-detection helpers.

Audio is the signal case with its own path. [Audio Basics](audio-basics.md) is Ketan Doshi's series from sound and spectrograms to speech recognition, [Audio Terminology](audio-terminology.md) defines source separation and sound event detection so the task names match the methods, [Audio Feature Engineering](audio-feature-engineering.md) covers Mel spectrograms, MFCC, and related features, and [Audio Algorithms](audio-algorithms.md) collects the concrete methods and packages for those tasks.

## Graphs

The last bias is that the relations between points matter more than the points. [Graph Theory](graph-theory.md) covers centrality, community detection, courses, and tools, and [Social Network Analysis](social-network-analysis.md) is its applied social-graph companion, with definitions, people, and traits that spread in networks.
