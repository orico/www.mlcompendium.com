# Types Of Machine Learning

![Image via Abdul Rahid, via Dan Shewan, wrong credit? let me know](https://lh3.googleusercontent.com/AZPfRGUTS-0AN0SRSjjBfc3tFlIpYSGsOyjaX00K9\_QIOWVU\_GLlgxwNWZCB4bWXVo1Wb52-D6KCRD8uEYuxcbaqJJ9CCEPa-gy\_\_DbCMJ4esb2A9hRLJuapX\_tKGJZi8rRlrDzE)

[A wonderful introduction into machine learning, and how to choose the right algorithm or family of algorithms for the task at hand.](https://blogs.sas.com/content/subconsciousmusings/2017/04/12/machine-learning-algorithm-use/)

### VARIOUS MODEL FAMILIES

[Stanford cs221](https://stanford.edu/\~shervine/teaching/cs-221/) - reflex, variable, state, logic

### WEAKLY SUPERVISED

1. [Text classification with extremely small datasets](https://towardsdatascience.com/text-classification-with-extremely-small-datasets-333d322caee2), relies heavily on feature engineering methods such as number of hashtags, number of punctuations and other insights that are really good for this type of text.
2. A great [review paper](https://pdfs.semanticscholar.org/3adc/fd254b271bcc2fb7e2a62d750db17e6c2c08.pdf) for weakly supervision, discusses:
   1. Incomplete supervision
   2. Inaccurate
   3. Inexact
   4. Active learning
3. [Stanford on](https://dawn.cs.stanford.edu/2017/07/16/weak-supervision/) weakly
4. [Stanford ai on snorkel](http://ai.stanford.edu/blog/weak-supervision/)
5. [Intro to Snorkel](https://medium.com/@towardsai/data-centric-ai-with-snorkel-ai-the-enterprise-ai-platform-a8ed0803c24c)
6. [Hazy research on weak and snorkel](https://hazyresearch.github.io/snorkel/blog/ws\_blog\_post.html)
7. [Out of distribution generalization using test-time training](https://arxiv.org/abs/1909.13231) - "Test-time training turns a single unlabeled test instance into a self-supervised learning problem, on which we update the model parameters before making a prediction on this instance. "
8. [Learning Deep Networks from Noisy Labels with Dropout Regularization](https://arxiv.org/pdf/1705.03419.pdf) - "Large datasets often have unreliable labels—such as those obtained from Amazon’s Mechanical Turk or social media platforms—and classifiers trained on mislabeled datasets often exhibit poor performance. We present a simple, effective technique for accounting for label noise when training deep neural networks. We augment a standard deep network with a softmax layer that models the label noise statistics. Then, we train the deep network and noise model jointly via end-to-end stochastic gradient descent on the (perhaps mislabeled) dataset. The augmented model is overdetermined, so in order to encourage the learning of a non-trivial noise model, we apply dropout regularization to the weights of the noise model during training. Numerical experiments on noisy versions of the CIFAR-10 and MNIST datasets show that the proposed dropout technique outperforms state-of-the-art methods."
9. [Distill to label weakly supervised instance labeling using knowledge distillation](https://arxiv.org/pdf/1907.12926.pdf) - “Weakly supervised instance labeling using only image-level labels, in lieu of expensive fine-grained pixel annotations, is crucial in several applications including medical image analysis. In contrast to conventional instance segmentation scenarios in computer vision, the problems that we consider are characterized by a small number of training images and non-local patterns that lead to the diagnosis. In this paper, we explore the use of multiple instance learning (MIL) to design an instance label generator under this weakly supervised setting. Motivated by the observation that an MIL model can handle bags of varying sizes, we propose to repurpose an MIL model originally trained for bag-level classification to produce reliable predictions for single instances, i.e., bags of size 1. To this end, we introduce a novel regularization strategy based on virtual adversarial training for improving MIL training, and subsequently develop a knowledge distillation technique for repurposing the trained MIL model. Using empirical studies on colon cancer and breast cancer detection from histopathological images, we show that the proposed approach produces high-quality instance-level prediction and significantly outperforms state-of-the MIL methods.”
10. [Yet another article summarising FAIR](https://neurohive.io/en/state-of-the-art/semi-weakly-supervised-learning-increasing-classification-accuracy-with-billion-scale-unlabeled-images/)

### SEMI SUPERVISED

1. [Paper review](https://pdfs.semanticscholar.org/3adc/fd254b271bcc2fb7e2a62d750db17e6c2c08.pdf)
2. [Ruder an overview of proxy labeled for  semi supervised (AMAZING)](https://ruder.io/semi-supervised/)
3. Self training
   1. [Self training and tri training](https://github.com/zidik/Self-labeled-techniques-for-semi-supervised-learning)
   2. [Confidence regularized self training](https://github.com/yzou2/CRST)
   3. [Domain adaptation for semantic segmentation using class balanced self-training](https://github.com/yzou2/CBST)
   4. [Self labeled techniques for semi supervised learning](https://github.com/zidik/Self-labeled-techniques-for-semi-supervised-learning)
4. Tri training
   1. [Trinet for semi supervised Deep learning](https://www.ijcai.org/Proceedings/2018/0278.pdf)
   2. [Tri training exploiting unlabeled data using 3 classes](https://www.researchgate.net/publication/3297469\_Tri-training\_Exploiting\_unlabeled\_data\_using\_three\_classifiers), [paper](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.487.2431\&rep=rep1\&type=pdf)
   3. [Improving tri training with unlabeled data](https://link.springer.com/chapter/10.1007/978-3-642-25349-2\_19)
   4. [Tri training using NN ensemble](https://link.springer.com/chapter/10.1007/978-3-642-31919-8\_6)
   5. [Asymmetric try training for unsupervised domain adaptation](https://github.com/corenel/pytorch-atda), [another implementation](https://github.com/vtddggg/ATDA), [another](https://github.com/ksaito-ut/atda), [paper](https://arxiv.org/abs/1702.08400)
   6. [Tri training git](https://github.com/LiangjunFeng/Tri-training)
5. [Fast ai forums](https://forums.fast.ai/t/semi-supervised-learning-ssl-uda-mixmatch-s4l/56826)
6. [UDA GIT](https://github.com/google-research/uda), [paper](https://arxiv.org/abs/1904.12848), [medium\*](https://medium.com/syncedreview/google-brain-cmu-advance-unsupervised-data-augmentation-for-ssl-c0a6157505ce), medium 2 ([has data augmentation articles)](https://medium.com/towards-artificial-intelligence/unsupervised-data-augmentation-6760456db143)
7. [s4l](https://arxiv.org/abs/1905.03670)
8. [Google’s UDM and MixMatch dissected](https://mlexplained.com/2019/06/02/papers-dissected-mixmatch-a-holistic-approach-to-semi-supervised-learning-and-unsupervised-data-augmentation-explained/)- For text classification, the authors used a combination of back translation and a new method called TF-IDF based word replacing.

Back translation consists of translating a sentence into some other intermediate language (e.g. French) and then translating it back to the original language (English in this case). The authors trained an English-to-French and French-to-English system on the WMT 14 corpus.

TF-IDF word replacement replaces words in a sentence at random based on the TF-IDF scores of each word (words with a lower TF-IDF have a higher probability of being replaced).

1. [MixMatch](https://arxiv.org/abs/1905.02249), [medium](https://towardsdatascience.com/a-fastai-pytorch-implementation-of-mixmatch-314bb30d0f99), [2](https://medium.com/@sanjeev.vadiraj/eureka-mixmatch-a-holistic-approach-to-semi-supervised-learning-125b14e82d2f), [3](https://medium.com/@sshleifer/mixmatch-paper-summary-1995f3d11cf), [4](https://medium.com/@literallywords/tl-dr-papers-mixmatch-9dc4cd217121), that works by guessing low-entropy labels for data-augmented unlabeled examples and mixing labeled and unlabeled data using MixUp. We show that MixMatch obtains state-of-the-art results by a large margin across many datasets and labeled data amounts
2. ReMixMatch - [paper](https://arxiv.org/pdf/1911.09785.pdf) is really good. “We improve the recently-proposed “MixMatch” semi-supervised learning algorithm by introducing two new techniques: distribution alignment and augmentation anchoring”
3.  [FixMatch](https://amitness.com/2020/03/fixmatch-semi-supervised/) - FixMatch is a recent semi-supervised approach by Sohn et al. from Google Brain that improved the state of the art in semi-supervised learning(SSL). It is a simpler combination of previous methods such as UDA and ReMixMatch.\
    ![](https://lh6.googleusercontent.com/9gNryK4qk-1VHSlpbSFThr0rTnKe6EDiwSDxqDaW4EEx-rIm9LGqs5uGFYHfMsQtJWd9Ls\_NAnap\_wHHAe\_qOBGcZgMJ7ruGkuxv2nIY8AP1mq82PgDxtgmsVO59G\_rDOnoNvUDk)

    _Image via_ [Amit Chaudhary](https://amitness.com) _wrong credit?_ [_let me know_](mailto:ori@oricohen.com)
4. [Curriculum Labeling: Self-paced Pseudo-Labeling for Semi-Supervised Learning](https://arxiv.org/pdf/2001.06001.pdf)
5. [FAIR](https://ai.facebook.com/blog/billion-scale-semi-supervised-learning/) [2](https://ai.facebook.com/blog/mapping-the-world-to-help-aid-workers-with-weakly-semi-supervised-learning/) original, [Summarization of FAIR’s student teacher weak/ semi supervision](https://analyticsindiamag.com/how-to-do-machine-learning-when-data-is-unlabelled/)
6. [Leveraging Just a Few Keywords for Fine-Grained Aspect Detection Through Weakly Supervised Co-Training](https://www.aclweb.org/anthology/D19-1468.pdf)
7. [Fidelity-Weighted](https://openreview.net/forum?id=B1X0mzZCW) Learning - “fidelity-weighted learning” (FWL), a semi-supervised student- teacher approach for training deep neural networks using weakly-labeled data. FWL modulates the parameter updates to a student network (trained on the task we care about) on a per-sample basis according to the posterior confidence of its label-quality estimated by a teacher (who has access to the high-quality labels). Both student and teacher are learned from the data."
8. [Unproven student teacher git](https://github.com/EricHe98/Teacher-Student-Training)
9. [A really nice student teacher git with examples](https://github.com/yuanli2333/Teacher-free-Knowledge-Distillation).

![Image by yuanli2333. wrong credit? let me know](https://lh6.googleusercontent.com/tlo5HqMjycySNl9Pbmr-uW-azozTC5cc7if-7r6-0LCeRJO2snTm-hsEf7mUpr1hp6wSnIVy6GnqFG6pEbxTPgu9fjjHP6gtn1dKQCwEI-x12UxYzWBWfidqMwVxZetA10VznMhs)

10\. [Teacher student for tri training for unlabeled data exploitation](https://arxiv.org/abs/1909.11233)

![Image by the late Dr. Hui Li, @ SAS. wrong credit? let me know](https://lh6.googleusercontent.com/J648WfIzGrbgjfSCK4S4lkCFbPWrSq6vwN1KERJ-yk5E21Jl3ZIeX7V98LS6rNIuY1Yc631oKIX-8H-dUyoqBHSoQEerZG\_KnKpwKWbhk5IHK3G0nTpCZ4ddGYGP-beBydYVOkKx)

### REGRESSION

Metrics:&#x20;

1. [R2](https://en.wikipedia.org/wiki/Coefficient\_of\_determination)
2. Medium [1](https://towardsdatascience.com/regression-an-explanation-of-regression-metrics-and-what-can-go-wrong-a39a9793d914), [2](https://medium.com/@george.drakos62/how-to-select-the-right-evaluation-metric-for-machine-learning-models-part-1-regrression-metrics-3606e25beae0), [3](https://medium.com/@george.drakos62/how-to-select-the-right-evaluation-metric-for-machine-learning-models-part-2-regression-metrics-d4a1a9ba3d74), [4](https://medium.com/usf-msds/choosing-the-right-metric-for-machine-learning-models-part-1-a99d7d7414e4),
3. [Tutorial](https://www.dataquest.io/blog/understanding-regression-error-metrics/)

### ACTIVE LEARNING

1. If you need to start somewhere start [here](https://www.datacamp.com/community/tutorials/active-learning) - types of AL, the methodology, examples, sample selection functions.
2. A thorough [review paper](http://burrsettles.com/pub/settles.activelearning.pdf) about AL
3. [The book on AL](http://burrsettles.com/pub/settles.activelearning.pdf)
4. [Choose your model first, then do AL, from lighttag](https://www.lighttag.io/blog/active-learning-optimization-is-not-imporvement/)
   1. The alternative is Query by committee - Importantly, the active learning method we presented above is the most naive form of what is called "uncertainty sampling" where we chose to sample based on how uncertain our model was. An alternative approach, called Query by Committee, maintains a collection of models (the committee) and selecting the most "controversial" data point to label next, that is one where the models disagreed on. Using such a committee may allow us to overcome the restricted hypothesis a single model can express, though at the onset of a task we still have no way of knowing what hypothesis we should be using.
   2. [Paper](https://arxiv.org/pdf/1807.04801.pdf): warning against transferring actively sampled datasets to other models
5. [How to increase accuracy with AL ](http://www.ijcte.org/papers/910-AC0013.pdf)
6. [AL with model selection](http://www.alnurali.com/papers/paper\_aaai\_2014.pdf) - paper
7. Using weak and strong oracle in AL, [paper](http://publications.lib.chalmers.se/records/fulltext/248447/248447.pdf).
8. [The pitfalls of AL](http://www.kdd.org/exploration\_files/v12-02-9-UR-Attenberg.pdf) - how to choose (cost-effectively) the active learning technique when one starts without the labeled data needed for methods like cross-validation; 2. how to choose (cost-effectively) the base learning technique when one starts without the labeled data needed for methods like cross-validation, given that we know that learning curves cross, and given possible interactions between active learning technique and base learner; 3. how to deal with highly skewed class distributions, where active learning strategies find few (or no) instances of rare classes; 4. how to deal with concepts including very small subconcepts (“disjuncts”)—which are hard enough to find with random sampling (because of their rarity), but active learning strategies can actually avoid finding them if they are misclassified strongly to begin with; 5. how best to address the cold-start problem, and especially 6. whether and what alternatives exist for using human resources to improve learning, that may be more cost efficient than using humans simply for labeling selected cases, such as guided learning \[3], active dual supervision \[2], guided feature labeling \[1], etc.
9. [Confidence based stopping criteria paper](http://www.cs.cmu.edu/\~./hovy/papers/10ACMjournal-activelearning-stopping.pdf)
10. A great [tutorial ](http://hunch.net/\~active\_learning/active\_learning\_icml09.pdf)
11. [An ok video](https://www.youtube.com/watch?v=Et7h1A1j4ns\&feature=youtu.be)
12. [Active learning framework in python](https://github.com/bwallace/curious\_snake)
13. [Active Learning Using Pre-clustering](https://www.researchgate.net/profile/Arnold\_Smeulders/publication/221345455\_Active\_learning\_using\_pre-clustering/links/54c3cc440cf2911c7a4cc74a/Active-learning-using-pre-clustering.pdf)
14. [A literature survey of active machine learning in the context of natural language processing](http://eprints.sics.se/3600/)
15. [Mnist competition (unpublished) using AL](http://dag.cvc.uab.es/mnist/statistics/)
16. [Practical Online Active Learning for Classification](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.87.5536\&rep=rep1\&type=pdf)
17. [Video 2](https://www.youtube.com/watch?v=8Jwp4\_WbRio\&index=7\&list=PLegWUnz91Wfsn6skGOofRoeFoOyfdqSyN)
18. [Active learning in R - code](https://github.com/gsimchoni/ActiveLearningExercise)
19. [Deep bayesian active learning with image data](https://arxiv.org/pdf/1703.02910.pdf)
20. [Medium on AL](https://news.voyage.auto/active-learning-and-why-not-all-data-is-created-equal-8a43a758c6f9)\*\*\*
21. [Integrating Human-in-the-Loop (HITL) in machine learning is a necessity, not a choice. Here’s why?](https://medium.com/@supriya2211/integrating-human-in-the-loop-hitl-in-machine-learning-application-is-a-necessity-not-a-choice-f25e131ca84e) By Supriya Ghosh

![Basic Framework for HITL Supriya Ghosh wrong credit? let me know](<.gitbook/assets/image (12) (1) (1) (1).png>)

#### Human In The loop ML book by [Robert munro](https://www.manning.com/books/human-in-the-loop-machine-learning#ref)

1. [GIT](https://github.com/rmunro/pytorch\_active\_learning)
2. [Active transfer learning](https://medium.com/pytorch/active-transfer-learning-with-pytorch-71ed889f08c1)
3. [Uncertainty sampling](https://towardsdatascience.com/uncertainty-sampling-cheatsheet-ec57bc067c0b)&#x20;
   1. Least Confidence: difference between the most confident prediction and 100% confidence
   2. Margin of Confidence: difference between the top two most confident predictions
   3. Ratio of Confidence: ratio between the top two most confident predictions
   4. Entropy: difference between all predictions, as defined by information theory

![by Robert (Munro) Monarch](https://lh3.googleusercontent.com/GK8uZ-WZg-0QFkXuxjR9iUM9tAhKJUeW-LApwTbknab37JXvvMQlQc-bvK2GpF5HGqoFCabSGzwWoSIzL6TdHg9\_WclZhopIbn6s4JO3eG6-\_yX8Q1S8C9tU90gvDGL\_kSPNFU1J)

[Diversity sampling](https://towardsdatascience.com/https-towardsdatascience-com-diversity-sampling-cheatsheet-32619693c304) -  you want to make sure that it covers as diverse a set of data and real-world demographics as possible.

1. Model-based Outliers: sampling for low activation in your logits and hidden layers to find items that are confusing to your model because of lack of information
2. Cluster-based Sampling: using Unsupervised Machine Learning to sample data from all the meaningful trends in your data’s feature-space
3. Representative Sampling: sampling items that are the most representative of the target domain for your model, relative to your current training data
4. Real-world diversity: using sampling strategies that increase fairness when trying to support real-world diversity

![by Robert (Munro) Monarch](https://lh6.googleusercontent.com/fsXyZEAvwEbhm7sGt7EcfxDz85zTKEwz4VvRdxzpXSaB2t\_5jZ3g3mjdClqUcORG8PgmtUNFAKF8nrIRYGCfl5bNVxjvYt9bn0NxmsM2U7J4NtebGxXKQSaXaZubAKx9s4v29-FP)

[Combine uncertainty sampling and diversity sampling](https://towardsdatascience.com/advanced-active-learning-cheatsheet-d6710cba7667)

1. Least Confidence Sampling with Clustering-based Sampling: sample items that are confusing to your model and then cluster those items to ensure a diverse sample (see diagram below).
2. Uncertainty Sampling with Model-based Outliers: sample items that are confusing to your model and within those find items with low activation in the model.
3. Uncertainty Sampling with Model-based Outliers and Clustering: combine methods 1 and 2.
4. Representative Cluster-based Sampling: cluster your data to capture multinodal distributions and sample items that are most like your target domain (see diagram below).
5. Sampling from the Highest Entropy Cluster: cluster your unlabeled data and find the cluster with the highest average confusion for your model.
6. Uncertainty Sampling and Representative Sampling: sample items that are both confusing to your current model and the most like your target domain.
7. Model-based Outliers and Representative Sampling: sample items that have low activation in your model but are relatively common in your target domain.
8. Clustering with itself for hierarchical clusters: recursively cluster to maximize the diversity.
9. Sampling from the Highest Entropy Cluster with Margin of Confidence Sampling: find the cluster with the most confusion and then sample for the maximum pairwise label confusion within that cluster.
10. Combining Ensemble Methods and Dropouts with individual strategies: aggregate results that come from multiple models or multiple predictions from one model via Monte-Carlo Dropouts aka Bayesian Deep Learning.

![by Robert (Munro) Monarch](https://lh5.googleusercontent.com/Ln4CzdRRCmVVrSNMhC5Ku6P5rhOFtaPcPduUFStCemdeZiASbU4G\_bf98-VRPEIfwW6zXdxjXG9ujkez3iqHUPgGEk3o0naDD5yx65ET\_YlssSv0Vfzp9MGthh9WWQpnKuqGmhCX)

Active transfer learning.

![by Robert (Munro) Monarch](https://lh4.googleusercontent.com/v\_wNRSX8ql9QU9ibjNkxGN9Z6KtgAxZ1jZk\_wZo62Hcyt-p4XAh5ErtRdkU7pG9J8kVZ22PuxMhTrWrsJ7uehnMIGZlwR13kukFc7i63YmzBAC3Ow7NTnAjnG2rPsTkbKkcLbCq9)

Machine in the loop

1. [Similar to AL, just a machine / model / algo adds suggestions. This is obviously a tradeoff of bias and clean dataset](https://www.lighttag.io/blog/when-to-use-machine-in-the-loop/)

### ONLINE LEARNING

1. If you want to start with OL - [start here](https://dziganto.github.io/data%20science/online%20learning/python/scikit-learn/An-Introduction-To-Online-Machine-Learning/) & [here](https://www.analyticsvidhya.com/blog/2015/01/introduction-online-machine-learning-simplified-2/)
2. Shay Shalev - [A thesis about online learning](http://ttic.uchicago.edu/\~shai/papers/ShalevThesis07.pdf)&#x20;
3. [Some answers about what is OL,](https://www.quora.com/What-is-the-best-way-to-learn-online-machine-learning) the first one actually talks about S.Shalev’s [other paper.](http://www.cs.huji.ac.il/\~shais/papers/OLsurvey.pdf)
4. Online learning - Andrew Ng - [coursera](https://www.coursera.org/learn/machine-learning/lecture/ABO2q/online-learning)
5. [Chip Huyen on online prediction & learning](https://huyenchip.com/2020/12/27/real-time-machine-learning.html)

### ONLINE DEEP LEARNING (ODL)

1. [Hedge back propagation (HDP), Autonomous DL, Qactor](https://towardsdatascience.com/online-deep-learning-odl-and-hedge-back-propagation-277f338a14b2) - online AL for noisy labeled stream data.

### N-SHOT LEARNING

1. [Zero shot, one shot, few shot](https://blog.floydhub.com/n-shot-learning/) (siamese is one shot)

### ZERO SHOT LEARNING

[Instead of using class labels](https://www.youtube.com/watch?v=jBnCcr-3bXc), we use some kind of vector representation for the classes, taken from a co-occurrence-after-svd or word2vec. - quite clever. This enables us to figure out if a new unseen class is near one of the known supervised classes. KNN can be used or some other distance-based classifier. Can we use word2vec for similarity measurements of new classes?\
![](https://lh3.googleusercontent.com/Rim9\_QVRRSj7eJTYeCcs1FfXzf-k7Qp2Wdmgcd1H-N\_ZZ6-krl1O3pH8GLkZMAVk2eQ5Ye\_Os2nUMqqsKzq92iP2rtlt1lix\_KnsMQsSrpMDPYcqI02TU0RrcZZMBmqfiLQj7xeN)

Image by [Dr. Timothy Hospedales, Yandex](https://www.youtube.com/watch?v=jBnCcr-3bXc)

for classification, we can use nearest neighbour or manifold-based labeling propagation.\
![](https://lh4.googleusercontent.com/nwZTsm4rfemR9-hNsyVpn1sFc4jJ9b2RAf\_gZKds51ki81crI9\_C6L5xI5M1F7OMK6a2Et7vS4JKWwtFMODKj\_RfQ6jTmCtrSPfQb4jMoZrZ5ZEoIm4uxublmBTgkJLkSvsMqYYF)

Image by [Dr. Timothy Hospedales, Yandex](https://www.youtube.com/watch?v=jBnCcr-3bXc)

Multiple category vectors? Multilabel zero-shot also in the video

#### GPT3 is ZERO, ONE, FEW

1. [Prompt Engineering Tips & Tricks](https://blog.andrewcantino.com/blog/2021/04/21/prompt-engineering-tips-and-tricks/)
2. [Open GPT3 prompt engineering](https://medium.com/swlh/openai-gpt-3-and-prompt-engineering-dcdc2c5fcd29)
