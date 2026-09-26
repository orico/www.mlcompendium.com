# Benchmarking

A score means little until it is compared with something: another dataset, another algorithm, another machine. The page walks those comparisons in order, from datasets and classifiers through NumPy's BLAS, hardware, cloud providers, and deep-learning platforms, then NLP and multi-task learning leaderboards, and ends with notes on scaling networks.

## Algorithms

The comparisons start with what is being run, the data and the algorithm, before the machine it runs on.

### Datasets

The first benchmark is of the data itself. EFF FF Benchmarks in AI used to be the pointer here; that address no longer opens and is kept at the end of the page. On the library side, [scikit bench](https://github.com/IntelPython/scikit-learn_bench) is described in its own words: "scikit-learn_bench benchmarks various implementations of machine learning algorithms across data analytics frameworks. It currently support the scikit-learn, DAAL4PY, cuML, and XGBoost frameworks for commonly used machine learning algorithms."

### Algorithms (classifiers)

With the implementations benchmarked, the next question is which classifier to pick. [Comparing](https://martin-thoma.com/comparing-classifiers/) is the side-by-side of accuracy, speed, memory and 2D visualization of classifiers, and every model in it links to its scikit-learn page, each with its own gallery of examples:

[SVM,](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) [k-nearest neighbors,](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) [Random Forest,](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) [AdaBoost Classifier,](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) [Gradient Boosting,](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html) [Naive, Bayes,](http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html) [LDA,](http://scikit-learn.org/0.16/modules/generated/sklearn.lda.LDA.html) [QDA,](http://scikit-learn.org/0.16/modules/generated/sklearn.qda.QDA.html) [RBMs,](http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.BernoulliRBM.html) [Logistic Regression,](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) [RBM](http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.BernoulliRBM.html) + Logistic Regression Classifier

The SVC, KNeighborsClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, GaussianNB, and LogisticRegression pages all list the classifier comparison among their examples, LDA and QDA point at the scikit-learn 0.16.1 documentation, and BernoulliRBM shows Restricted Boltzmann Machine features for digit classification. The same comparison holds inside one model family: [LSTM vs cuDNN LSTM](https://chainer.org/general/2017/03/15/Performance-of-LSTM-Using-CuDNN-v5.html) shows that batch size of power 2 matters, and the latter is faster.

### Numpy Blas

Under every classifier sits linear algebra, so the library NumPy links against changes the timing. [How do i know which version of blas is installed](https://stackoverflow.com/questions/37184618/find-out-if-which-blas-library-is-used-by-numpy) is the question of how to be sure NumPy uses a BLAS library at all when it was installed from a package manager rather than compiled by hand. [Benchmark OpenBLAS, Intel MKL vs ATLAS](https://github.com/tmolteno/necpp/issues/18) is a shared benchmark of LAPACK / BLAS libraries from someone replacing CPU and math library for a huge simulation model, who concluded that Intel MKL is the best. The figure is that benchmark.

<figure><img src="../.gitbook/assets/gimg-eaf5612396a1.png" alt=""><figcaption><p>OpenBLAS, Intel MKL vs ATLAS benchmark.</p>
<p>Credit: <a href="https://lh5.googleusercontent.com/podTyc9Z0eDjObB4aW6-2AVWxhlG3pE8M3ccWBUj3oIGDgB6uWmXlt96aiuVAm9vvw33iShedQ1Gn_w6J3qhRGKThnZH-Puy5ZfoYmHL3GFTMxxUh_EIXOCtOTqjQHdqrjCZzh3N">copied from the original hosted image</a>.</p>
</figcaption></figure>

Another comparison of NumPy BLAS builds is in the next figure; its source post no longer opens and is kept at the end of the page. Both figures are copied from the original hosted images.

<figure><img src="../.gitbook/assets/gimg-02e0dbef7ebf.png" alt=""><figcaption><p>Another NumPy BLAS comparison.</p>
<p>Credit: <a href="https://lh5.googleusercontent.com/6tufYNKWkxO5azzf07erA8QIeXhDuWpz8VRaWVw1x16rHahEbj5PRyZ4e6Dr_65ccBGDxj18EKXljVgl1DiO4SAqw_pZqGDlzTs5zsjInsRut8ebtQFgDXkoDnpskD9JbYApijwK">copied from the original hosted image</a>.</p>
</figcaption></figure>

### Hardware

Past the libraries, the machine decides the rest, and the first choice is GPU versus CPU. [Nvidia](https://www.phoronix.com/scan.php?page=article&item=nvidia-rtx2080ti-tensorflow&num=1) compares the 1070 vs 1080 vs 2080. [Cpu vs GPU benchmarking for CNN/Test/LTSM/BDLTSM](http://minimaxir.com/2017/07/cpu-or-gpu/) benchmarks TensorFlow on cloud CPUs and finds them cheaper for deep learning than cloud GPUs, because of the massive cost differential afforded by preemptible instances. [Nvidia GPUs](https://www.pugetsystems.com/labs/hpc/TitanXp-vs-GTX1080Ti-for-Machine-Learning-937/) is TitanXp vs GTX1080Ti for Machine Learning: the GTX1080Ti proved as good as the Titan X Pascal at a much lower price, and the new Titan Xp is tested to see how much faster it is. As of March/17, [Nvidia GPUs for desktop](https://medium.com/@timcamber/deep-learning-pc-build-5cffa71ad97) compares cards in terms of price and cuda units, and the bottom line is 1060-1080. [Another bench up to 2013](http://timdettmers.com/2017/04/09/which-gpu-for-deep-learning/) is regarding many GPUS vs CPUs in terms of BW.

### Cloud providers

Renting the hardware brings its own comparison, the hardware benchmarks across cloud providers from the gensim series. [Part 1](https://rare-technologies.com/machine-learning-hardware-benchmarks/) and [part 2 y gensim](https://rare-technologies.com/machine-learning-benchmarks-hardware-providers-gpu-part-2/) were that series; both addresses now open PII Tools, a product for analyzing personal data and sensitive information at scale, not the benchmarks.

### Platforms

On a given machine, the framework is the last variable. Cntk vs tensorflow was one comparison; its address no longer opens and is kept at the end of the page. [CNTK, TEnsor, torch, etc on cpu and gpu](https://arxiv.org/pdf/1608.07249.pdf) is the paper Benchmarking State-of-the-Art Deep Learning Software Tools.

### NLP

Leaderboards move the comparison from speed to capability, and language has its own. [XTREME: A Massively Multilingual Multi-task Benchmark for Evaluating Cross-lingual Generalization](https://github.com/google-research/xtreme/blob/master/README.md) is a benchmark for the cross-lingual generalization ability of pre-trained multilingual models that covers 40 typologically diverse languages and includes nine tasks.

#### Multi-Task Learning

A multi-task benchmark raises the question of how one model should balance the tasks, through loss weighting and shared representations. [Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics](https://arxiv.org/abs/1705.07115) (Yarin Gal), with an unofficial implementation on [GitHub](https://github.com/ranandalon/mtl), states the problem: "In this paper we make the observation that the performance of such systems is strongly dependent on the relative weighting between each task’s loss. Tuning these weights by hand is a difficult and expensive process, making multi-task learning prohibitive in practice. We propose a principled approach to multi-task deep learning which weighs multiple loss functions by considering the homoscedastic uncertainty of each task. " [Ruder on Multi Task Learning](https://ruder.io/multi-task/) gives the reason to share at all: "By sharing representations between related tasks, we can enable our model to generalize better on our original task. This approach is called Multi-Task Learning (MTL) and will be the topic of this blog post."

### GLUE

For English understanding, the standard leaderboard is GLUE and its successor. The same notes are in [BERT](../language-ai/pretrained-language-models.md#bert).

[Glue / super glue](https://gluebenchmark.com/leaderboard/) is the leaderboard of the General Language Understanding Evaluation (GLUE) benchmark, a collection of resources for training, evaluating, and analyzing natural language understanding systems.

### State of the art in AI

Beyond single leaderboards, state of the art can be tracked across the field, in terms of [domain X datasets](https://www.stateoftheart.ai/).

### Scaling networks and predicting performance of NN

The last comparison is forward-looking: predicting train time and accuracy before scaling networks across GPUs. [A great overview of NN types](https://www.youtube.com/watch?v=lgK0BlXdOCw&feature=youtu.be) is the video for it, but the idea behind the video is to create a system that can predict train time and possibly accuracy when scaling networks using multiple GPUs, there is also a nice slide about general hardware recommendations. The figure below is that slide, copied from the original hosted image.

<figure><img src="../.gitbook/assets/gimg-5fcba0363084.png" alt=""><figcaption><p>Hardware recommendations when scaling networks across GPUs.</p>
<p>Credit: <a href="https://lh4.googleusercontent.com/mmxNCa6J3W7s3h1LUkxzEBzKxvSOlCFTzEYgaE1zcOFJV59SCQ4j5jKWMvP9JZGmaGE29VJiALogJlgK8x_V_nUo2fvBPRaXA41K1t9w39WDLM_aKVHh-yithcHZE-A0x9zSvBAy">copied from the original hosted image</a>.</p>
</figcaption></figure>

## Deprecated links

{% hint style="warning" %}
These links and images no longer work. The original wording is kept here. A same-resource copy, when one was checked, is used above.
{% endhint %}

- Another comparison (NumPy BLAS). This address no longer opens: http://markus-beuckelmann.de/blog/boosting-numpy-blas.html
- EFF FF Benchmarks in AI. This address no longer opens: https://www.eff.org/ai/metrics
- Cntk vs tensorflow. This address no longer opens: http://minimaxir.com/2017/06/keras-cntk/
