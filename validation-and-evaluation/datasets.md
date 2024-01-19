# Datasets

### Structured / Unstructured data

1. [Unstructured ](https://www.webopedia.com/TERM/U/unstructured\_data.html)
2. [Structured](https://www.webopedia.com/TERM/S/structured\_data.html)

### BIAS / VARIANCE

1. [Various Bias types](https://queue.acm.org/detail.cfm?id=3466134) by queue.acm

![](<../.gitbook/assets/image (22).png>)

1. [Overfitting your test set, a statistican view point, a great article](https://lukeoakdenrayner.wordpress.com/2019/09/19/ai-competitions-dont-produce-useful-models/?fbclid=IwAR1WM5U7imq-2LFPifyCoTPp-MFwPoGROMLr2TZWAp41qgVeLdT-\_2bkLyk\&blogsub=confirming#subscribe-blog), bottom line use bonferroni correction.
2.  Understanding what is the next stage in DL (& ML) algorithm development: basic approach - [Andrew NG](https://www.youtube.com/watch?v=F1ka6a13S9I) on youtube

    Terms: training, validation, test.

    Split: training & validation 70%, test 30%

    Procedure: cross fold training and validation, or further split 70% to training and validation.



    BIAS - Situation 1 - doing much worse than human:&#x20;

    Human expert: 1% error

    Training set error: 5% error (test on train)

    Validation set error: 6% error (test on validation or CFV)

    Conclusion: there is a BIAS between human expert and training set

    Solution: 1. Train deeper or bigger\larger networks, 2. train longer, 3. May needs more data to get to the human expert level, Or 4. New model architecture.\


    VARIANCE - Situation 2 - validation set not close to training set error:

    Human expert: 1% error

    Training set error: 2% error

    Validation set error: 6% error

    Conclusion: there is a VARIANCE problem, i.e. OVERFITTING, between training and validation.

    Solution: 1. Early stopping, 2. Regularization or 3. get more data, or 4. New model architecture.

    Situation 3 - both:Human expert: 1% error

    Training set error: 5% error

    Validation set error: 10% error

    Conclusion: both problems occur, i.e., BIAS as and VARIANCE.

    Solution:  do it al

* Underfitting = Get more data&#x20;
* Overfitting = Early stop, regularization, reason: models detail & noise.
* Happens more in non parametric (and non linear) algorithms such as decision trees.
* Bottom line, bigger model or more data will solve most issues.
* In practice advice with [regularized linear regression.](http://www.holehouse.org/mlclass/10\_Advice\_for\_applying\_machine\_learning.html)\


![](https://lh4.googleusercontent.com/Zg\_aGmWE7DxzEUboiliygq923F9Dj6kwmXuCZ2-D4uti4R5HApLcTC-TDaHyb4BLvqRZns6dgTgxABzOObqPvtHIl9Enm5wGCtkC27gNRsnCjzhDxZwaHdwJUTRGu-MpSGvyl72q)

![](https://lh5.googleusercontent.com/0T7HwSvfvzgWXZTPeKGHmqQK0LhY1B7gJMMxXjAA4UEFlL1H9\_7pngyLM8LXnqdvMglswd\_UDH2GjymXZs-Lt3ET5ETZSNc3PsGXH5wbccfr61fUiUlRWN1ya6sI-9hHqn1Rg0PP)

![](https://lh6.googleusercontent.com/A7XbPpsAfZ59Mehdl96Vm\_GfICYZQvl9dNZD-WWuxbvPvbkBJ6DB6KFWFoMtu2nMow9V7yDwpItj4PVi2m8pLYoOkzbCOKscftUvVP-2N49kTWxRedfO7IIQnA-IHIdWoN89Ad-D)

![](https://lh4.googleusercontent.com/CUYtuclj3O7kKkb8M103Tx96LdES40KCqdXB5-t7tByYj3m-rgEdBLtWdtgdggj8-i-qOTh-GdZA\_zJoP-R69sXg2VwelR3glO1zqrvhAt9uvYD5zH\_DfqxU4m5wMcmLhL2EgKtQ)

IMPORTANT! For Test Train efficiency when the data is from different distributions:

E.g: TRAIN: 50K hours of voice chatter as the train set for a DLN, TEST: 10H for specific voice-based problem, i.e, taxi chatter.

Best practice: better to divide the validation & test from the same distribution, i.e. the 10H set.

Reason: improving scores on validation which is from a diff distribution will not be the same quality as improving scores on a validation set originated from the actual distribution of the problem’s data, i.e., 10H.

NOTE: Unlike the usual supervised learning, where all the data is from the same distribution, where we split the training to train and validation (cfv).\


[Situation 4](https://youtu.be/F1ka6a13S9I?t=47m26s): However, when there are 2 distributions it’s possible to extend the division of the training set to validation\_training and training, and the test to validation and test.

Split:  Train, Valid\_Train = 48K\2K & Valid, Test, 5K & 5K.

![](https://lh6.googleusercontent.com/Fllv8NnciZ-EQsdO2zvfLdLt90e3t1BIrXWR5NvAap64k0JdChd7j3ABT6RoE83d0BM5EFgTwW9asrN99yDW58hAPoaOLG8eI43rlO\_tK68e-SkHej65LEV0xCfFT5aUI78g4oIQ)

So situation 1 stays the same,&#x20;

Situation 2 is Valid\_Train error (train\_dev)

Situation 3 is Valid\_Test error - need more data, data synthesis - tweak test to be similar to train data, new architecture as a solution

Situation 4 is now Test set error - get more data\
\


### SPARSE DATASETS

[Sparse matrices](https://machinelearningmastery.com/sparse-matrices-for-machine-learning/) in ML - one hot/tfidf, dictionary/list of lists/ coordinate list.

###

### TRAINING METHODOLOGIES

1. Train test split
2. Cross validation
3. Transfer learning - using a pre existing classifier similar to your domain, usually trained on millions of samples. fine-tuned on new data in order to create a new classifier that utilizes that information in the new domain. Examples such as w2v or classic resnet fine-tuning.
4. Bootstrapping training- using a similar dataset, such as yelp, with 5 stars to create a pos/neg sentiment classifier based on 1 star and 5 stars. Finally using that to label or sample select from an unlabelled dataset, in order to create a new classifier or just to sample for annotation etc.
5. [Student-teacher paradigm](https://developers.facebook.com/videos/2019/from-visual-recognition-to-reasoning/) (facebook), using a big labelled dataset to train a teacher classifier, predicting on unlabelled data, choosing the best classified examples based on probability, using those to train a new student model, finally fine-tune on the labeled dataset to create a more robust model, which is expected to know the unlabelled dataset and the labelled dataset with higher accuracy. With respect to the fully supervised teacher model / baseline.

![](https://lh6.googleusercontent.com/U7Zn0WtBMVLvvN4rinTJhzRU4P8zMJB\_1SNiGPQzboJfltWzdTUmcoDcc\_0lx94qlfHW4QU11wftCujikfvR3StMxOPCE3FTWPhwPqsfCrYj29NIVt8jb1PlU3hv7hq2Y1DscOWH)

1. Yoav’s method for transfer learning for languages - train a classifier on labelled data from english and spanish, fine tune using left out spanish data, stop before overfitting. This can be generalized to other domains.

#### TRANSFER LEARNING

1. [In deep learning](https://towardsdatascience.com/a-comprehensive-hands-on-guide-to-transfer-learning-with-real-world-applications-in-deep-learning-212bf3b2f27a)
2. ![](https://lh3.googleusercontent.com/xUFaHrHjaypItfpjfzNEZ\_Zv2BZJWieQuoBGLXfEnqNJr1PjQXt6D-TJpgaSfhU-BmoMiNqVfQFXMwBFIuvnxRYM6yZS2fxLfd9RoYRto8Bm5oeQZekUqQzO1HZP203PRu3wQT07)

###

### TRAIN / TEST / CROSS VALIDATION

[Scikit-lego on group-based splitting and transformation](https://scikit-lego.readthedocs.io/en/latest/meta.html#Grouped-Prediction)

[Images from here](https://www.kdnuggets.com/2017/08/dataiku-predictive-model-holdout-cross-validation.html).

![](https://lh3.googleusercontent.com/v\_T8IXtpI7PhjIrwPjLqsh0rEGm-ejpzFK1FlDRByqkpm1sWHxtKCMkspBW9omVpJo-EhuURiipbqEFM\_yVZIviCp7XtI8RMLPd347ccOkmjOADJjPuSUl8sd-2eQmpK1SoJgg\_R)

![](https://lh4.googleusercontent.com/5MFk9a4mEfSMCu4za3oxTshh4TD5X4cAvyqXuIYqJhiV7UwG4sybQWKXk-PfWpZ15lZtzEurEFH7r-LoF-kvZMqzreRCsZUf9VLoujj8sCf-4EsnIQgkuEjnhNGiYYO7AQ12mf0C)

[Train Test methodology](http://machinelearningmastery.com/how-to-choose-the-right-test-options-when-evaluating-machine-learning-algorithms/) -&#x20;

“[The training](https://stats.stackexchange.com/questions/19048/what-is-the-difference-between-test-set-and-validation-set) set is used to fit the models; the validation set is used to estimate prediction error for model selection; the test set is used for assessment of the generalization error of the final chosen model. Ideally, the test set should be kept in a “vault,” and be brought out only at the end of the data analysis”\


* Random Split tests 66\33 - problem: variance each time we rerun.
* Multiple times random split tests - problem: samples may not be included in train\test or selected multiple times.
* Cross validation - pretty good, diff random seed results in diff mean accuracy, variance due to randomness
* Multiple cross validation - accounts for the randomness of the CV
* Statistical significance ( t-test)  on multi CV - are two samples drawn from the same population? (no difference). If “yes”, not significant, even if the mean and std deviations differ.

Finally, When in doubt, use k-fold cross validation (k=10) and use multiple runs of k-fold cross validation with statistical significance tests.\


[Out of fold](https://machinelearningmastery.com/out-of-fold-predictions-in-machine-learning/) - leave unseen data, do cross fold on that. Good for ensembles.\


### VARIOUS DATASETS

1. [26 of them](https://www.analyticsvidhya.com/blog/2018/05/24-ultimate-data-science-projects-to-boost-your-knowledge-and-skills/?utm\_source=facebook.com\&utm\_medium=social)
2. [24](https://lionbridge.ai/datasets/25-best-parallel-text-datasets-for-machine-translation-training/)
3. [Eu-](https://datarepository.wolframcloud.com/resources/Europarl-English-Spanish-Machine-Translation-Dataset-V7)es, [2](https://data.europa.eu/euodp/en/data/dataset/elrc\_339)
4. 50K -  [ModelDepot](https://modeldepot.io/) alone has over 50,000 freely accessible pre-trained models with search functionality to
5.

### IMBALANCED DATASETS

1. ([the BEST resource and a great api for python)](http://contrib.scikit-learn.org/imbalanced-learn/stable/over\_sampling.html) with visual samples - it actually works well on clustering.
2. [Mastery on](https://machinelearningmastery.com/cost-sensitive-learning-for-imbalanced-classification/?fbclid=IwAR0\_DeIydTAAkutypcMBfrnC4QyuyqVxDu\_uej5t48AvQKShcRUqfMm8Rqo) cost sensitive sampling
3. [Smote for imbalance](https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/?fbclid=IwAR3W59c54ohoaIHnHLQFCcZZanFXI4QzIzuWiUtaUC851JFkwlevCAgvpbM)

[Systematic Investigation of imbalance effects in CNN’s](https://arxiv.org/abs/1710.05381), with several observations. This is crucial when training networks, because in real life you don’t always get a balanced DS.

They recommend the following:&#x20;

1. (i) the effect of class imbalance on classification performance is detrimental;
2. (ii) the method of addressing class imbalance that emerged as dominant in almost all analyzed scenarios was oversampling;&#x20;
3. (iii) oversampling should be applied to the level that totally eliminates the imbalance, whereas undersampling can perform better when the imbalance is only removed to some extent;&#x20;
4. (iv) as opposed to some classical machine learning models, oversampling does not necessarily cause overfitting of CNNs;&#x20;
5. (v) thresholding should be applied to compensate for prior class probabilities when overall number of properly classified cases is of interest.

General Rules:&#x20;

1. Many samples - undersampling
2. Few  samples  - over sampling
3. Consider random and non-random schemes
4. Different sample rations, instead of 1:1 (proof? papers?)

Balancing data sets ([wiki](https://en.wikipedia.org/wiki/Oversampling\_and\_undersampling\_in\_data\_analysis), [scikit learn](https://github.com/scikit-learn-contrib/imbalanced-learn) & [examples in SKLEARN](http://contrib.scikit-learn.org/imbalanced-learn/auto\_examples/index.html)):

1. Oversampling the minority class
   1. (Random) duplication of samples
   2. SMOTE [(in weka + needs to be installed](http://www.jair.org/media/953/live-953-2037-jair.pdf) & [paper)](http://www.jair.org/media/953/live-953-2037-jair.pdf) - find k nearest neighbours,&#x20;

New\_Sample = (random num in \[0,1] ) \* vec(ki,current\_sample)&#x20;

* (in weka) The nearestNeighbors parameter says how many nearest neighbor instances (surrounding the currently considered instance) are used to build an in between synthetic instance. The default value is 5. Thus the attributes of 5 nearest neighbors of a real existing instance are used to compute a new synthetic one.
* (in weka) The percentage parameter says how many synthetic instances are created based on the number of the class with less instances (by default - you can also use the majority class by setting the -Coption). The default value is 100. This means if you have 25 instances in your minority class, again 25 instances are created synthetically from these (using their nearest neighbours' values). With 200% 50 synthetic instances are created and so on.

1. ADASYN - shifts the classification boundary to the minority class, synthetic data generated for majority class.
2. Undersampling the majority class
   1. Remove samples
   2. Cluster centroids - replaces a cluster of samples (k-means) with a centroid.
   3. Tomek links - cleans overlapping samples between classes in the majority class.
   4. Penalizing the majority class during training
3. Combined over and under (hybrid) - i.e., SMOTE and tomek/ENN
4. Ensemble sampling&#x20;
   1. EasyEnsemble
   2. BalanceCascade
5. Dont balance, try algorithms that perform well with unbalanced DS
   1. Decision trees - C4.5\5\CART\Random Forest
   2. SVM
6. Penalize Models -&#x20;
   1. added costs for misclassification on the minority class during training such as penalized-SVM
   2. a [CostSensitiveClassifier](http://weka.sourceforge.net/doc.dev/weka/classifiers/meta/CostSensitiveClassifier.html) meta classifier in Weka that wraps classifiers and applies a custom penalty matrix for miss classification.
   3. complex

##

### SAMPLE SELECTION

1. [How to choose your sample size from a population based on confidence interval](https://www.checkmarket.com/blog/how-to-estimate-your-population-and-survey-sample-size/)

![](https://lh3.googleusercontent.com/gzSA5OXGcheJTZbY8Vj10NOBmumc9-v87G0G1sKF8cRP8rQegw5vE\_hvadFSZLNwY9p6ZQ7bgL61RIcSwv-gBUUycp\_0dx6yCpDgr3G2JAKVt4-Bq9Hpqri65B0Jr57MDqUekf-d)

1. [Data advice, should we get more data? How much](https://machinelearningmastery.com/much-training-data-required-machine-learning/)

[Gibbs sampling](https://wiseodd.github.io/techblog/2015/10/09/gibbs-sampling/): - Gibbs Sampling is a MCMC method to draw samples from a potentially really really complicated, high dimensional distribution, where analytically, it’s hard to draw samples from it. The usual suspect would be those nasty integrals when computing the normalizing constant of the distribution, especially in Bayesian inference. Now Gibbs Sampler can draw samples from any distribution, provided you can provide all of the conditional distributions of the joint distribution analytically.

###

### LEARNING CURVES

1. [Git examples](https://gist.github.com/orico/260097cb1a2926c6b6ca6f71c37c135b)
2. [Sklearn examples](https://stats.stackexchange.com/questions/283738/sklearn-learning-curve-example)
3. [Understanding bias variance via learning curves](http://digitheadslabnotebook.blogspot.com/2011/12/practical-advice-for-applying-machine.html)
4. [Unread - learning curve sampling applied to  model based clustering](http://www.jmlr.org/papers/volume2/meek02a/meek02a.pdf) - seemed like active learning, i.e., sample using EM/cluster to achieve nearly as accurate on all data
5. Predicting sample size required for training
6. [Advice on many things, including learning curves](https://blog.acolyer.org/2018/03/28/deep-learning-scaling-is-predictable-empirically/amp/?fbclid=IwAR0V1X1vuCZYmeku12YHJI7wwK7RCKEyE2Q7aRDDT58hjRPzAOrHfvo98WY)

This is a really wonderful study with far-reaching implications that could even impact company strategies in some cases. It starts with a simple question: “how can we improve the state of the art in deep learning?” We have three main lines of attack:

1. We can search for improved model architectures.
2. We can scale computation
3. We can create larger training data sets.

### DISTILLING DATA

1. [Medium on](https://towardsdatascience.com/data-maps-datasets-can-be-distilled-too-1991c3c260d6)  this [Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics](https://arxiv.org/abs/2009.10795). What I found interesting about this paper is that it challenges the common approach of “the more the merrier” when it comes to training data, and shifts the focus from the quantity of the data to the quality of the data.

### DATASET SELECTION

1. [Medium](https://medium.com/@amielmeiseles/how-to-choose-the-best-source-model-for-transfer-learning-41d5c91c1338)
2. ![](https://lh3.googleusercontent.com/J9qBdrVcRj5iz0X7-8XjFV4zqNNQpT\_MNOCt2Xb1wh34kX8ui82KagDKV88iyUb4BG9Tkos8CfMTjfd25xT1D4DY9869qmaQX\_fWVg6KG4qaMCMDCUfPMVQiPaRACAlQ8r40Kesh)
