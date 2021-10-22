# Ensembles

**Numpy Blas:**

1. [**How do i know which version of blas is installed**](https://stackoverflow.com/questions/37184618/find-out-if-which-blas-library-is-used-by-numpy)
2. [**Benchmark OpenBLAS, Intel MKL vs ATLAS**](https://github.com/tmolteno/necpp/issues/18)** **

![](https://lh5.googleusercontent.com/podTyc9Z0eDjObB4aW6-2AVWxhlG3pE8M3ccWBUj3oIGDgB6uWmXlt96aiuVAm9vvw33iShedQ1Gn_w6J3qhRGKThnZH-Puy5ZfoYmHL3GFTMxxUh_EIXOCtOTqjQHdqrjCZzh3N)

1. [**Another comparison**](http://markus-beuckelmann.de/blog/boosting-numpy-blas.html)
2. ![](https://lh5.googleusercontent.com/6tufYNKWkxO5azzf07erA8QIeXhDuWpz8VRaWVw1x16rHahEbj5PRyZ4e6Dr\_65ccBGDxj18EKXljVgl1DiO4SAqw_pZqGDlzTs5zsjInsRut8ebtQFgDXkoDnpskD9JbYApijwK)

**GLUE:**

1. [**Glue / super glue **](https://gluebenchmark.com/leaderboard/?fbclid=IwAR17Xo2pgpDVE_ZuwITDSi07FLM6S2f1VTXiLywwr2NnUGqS8AdndZLQpXI)

**State of the art in AI:**

1. **In terms of **[**domain X datasets**](https://www.stateoftheart.ai)

**Cloud providers:**

* [**Part 1**](https://rare-technologies.com/machine-learning-hardware-benchmarks/)**, **[**part 2 y gensim**](https://rare-technologies.com/machine-learning-benchmarks-hardware-providers-gpu-part-2/)
*

**Datasets: **

* [**EFF FF Benchmarks in AI**](https://www.eff.org/ai/metrics)

**Hardware:**

* [**Nvidia**](https://www.phoronix.com/scan.php?page=article\&item=nvidia-rtx2080ti-tensorflow\&num=1)** 1070 vs 1080 vs 2080**
* [**Cpu vs GPU benchmarking for CNN\Test\LTSM\BDLTSM**](http://minimaxir.com/2017/07/cpu-or-gpu/)** - google and amazon vs gpu**
* [**Nvidia GPUs**](https://www.pugetsystems.com/labs/hpc/TitanXp-vs-GTX1080Ti-for-Machine-Learning-937/)** - titax Xp\1080TI\1070 on googlenet**
* **March\17 - **[**Nvidia GPUs for desktop**](https://medium.com/@timcamber/deep-learning-pc-build-5cffa71ad97)**, in terms of price and cuda units, the bottom line is 1060-1080. **
* [**Another bench up to 2013**](http://timdettmers.com/2017/04/09/which-gpu-for-deep-learning/)** - regarding many GPUS vs CPUs in terms of BW**

**Platforms**

* [**Cntk vs tensorflow**](http://minimaxir.com/2017/06/keras-cntk/)
* [**CNTK, TEnsor, torch, etc on cpu and gpu**](https://arxiv.org/pdf/1608.07249.pdf)** **

**Algorithms:**

* [**Comparing**](https://martin-thoma.com/comparing-classifiers/)** accuracy, speed, memory and 2D visualization of classifiers:**

[**SVM,**](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)** **[**k-nearest neighbors,**](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)** **[**Random Forest,**](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)** **[**AdaBoost Classifier,**](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)** **[**Gradient Boosting,**](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)** **[**Naive, Bayes,**](http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html)** **[**LDA,**](http://scikit-learn.org/0.16/modules/generated/sklearn.lda.LDA.html)** **[**QDA,**](http://scikit-learn.org/0.16/modules/generated/sklearn.qda.QDA.html)** **[**RBMs,**](http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.BernoulliRBM.html)** **[**Logistic Regression,**](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)** **[**RBM**](http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.BernoulliRBM.html)** + Logistic Regression Classifier**

* [**LSTM vs cuDNN LS1TM**](https://chainer.org/general/2017/03/15/Performance-of-LSTM-Using-CuDNN-v5.html)** - batch size of power 2 matters, the latter is faster.**

**Scaling networks and predicting performance of NN:**

* [**A great overview of NN type**](https://www.youtube.com/watch?v=lgK0BlXdOCw\&feature=youtu.be)**s, but the idea behind the video is to create a system that can predict train time and possibly accuracy when scaling networks using multiple GPUs, there is also a nice slide about general hardware recommendations.**

![](https://lh4.googleusercontent.com/mmxNCa6J3W7s3h1LUkxzEBzKxvSOlCFTzEYgaE1zcOFJV59SCQ4j5jKWMvP9JZGmaGE29VJiALogJlgK8x_V_nUo2fvBPRaXA41K1t9w39WDLM_aKVHh-yithcHZE-A0x9zSvBAy)

**NLP**

* [**XTREME: A Massively Multilingual Multi-task Benchmark for Evaluating Cross-lingual Generalization**](https://github.com/google-research/xtreme/blob/master/README.md)

#### Multi-Task Learning

1. [Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics](https://arxiv.org/abs/1705.07115) (Yarin Gal) [GitHub](https://github.com/ranandalon/mtl) - "In this paper we make the observation that the performance of such systems is strongly dependent on the relative weighting between each task’s loss. Tuning these weights by hand is a difficult and expensive process, making multi-task learning prohibitive in practice. We propose a principled approach to multi-task deep learning which weighs multiple loss functions by considering the homoscedastic uncertainty of each task. "
2. [Ruder on Multi Task Learning](https://ruder.io/multi-task/) - "By sharing representations between related tasks, we can enable our model to generalize better on our original task. This approach is called Multi-Task Learning (MTL) and will be the topic of this blog post."
