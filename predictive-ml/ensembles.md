# Ensembles

One model has one set of mistakes; several models combined can cancel each other's. This page answers how to combine them, starting with an overview of voting, bagging, boosting, and stacking, then the WEKA view of each, then bagging and boosting in detail, and finally gradient boosting, XGBoost, and CatBoost.

The same notes are in [Active Learning](../problem-framing/active-learning.md), [EXTRA TREES](decision-trees.md#extra-trees), [Interview questions](../ai-product/data-science-management.md#interview-questions), and [RANDOM FOREST](decision-trees.md#random-forest).

The overview comes first. Toptal's Ensemble Methods: The Kaggle Machine Learning Champion is the (good) [review on voting, bagging, boosting stacking, cascading methodologies](https://www.toptal.com/machine-learning/ensemble-methods-kaggle-machine-learn), built on the point that ensembles often win Kaggle contests. The simplest ensemble is a vote, and sentdex's Combining Algos with a Vote, part 16 of Natural Language Processing With Python and NLTK, shows [How to combine several sklearn algorithms into a voting ensemble](https://www.youtube.com/watch?v=vlTQLb_a564&list=PLQVvvaa0QuDf2JswnfiGkliBInZnIC4HL&index=16). One step up from voting is stacking, where a model learns how to combine the others; Sebastian Raschka's mlxtend StackingClassifier is the [Stacking api, MLXTEND](http://rasbt.github.io/mlxtend/user_guide/classifier/StackingClassifier/).

Machine learning Mastery has a run of posts on the same ideas, each with its own outline. Stacking Ensemble for Deep Learning Neural Networks in Python is [stacking neural nets — really good](https://machinelearningmastery.com/stacking-ensemble-for-deep-learning-neural-networks/):

 1. Stacked Generalization Ensemble
 2. Multi-Class Classification Problem
 3. Multilayer Perceptron Model
 4. Train and Save Sub-Models
 5. Separate Stacking Model
 6. Integrated Stacking Model

The ways to merge outputs are in [How to Combine Predictions for Ensemble Learning](https://machinelearningmastery.com/combine-predictions-for-ensemble-learning/):

 1. Plurality Voting.
 2. Majority Voting.
 3. Unanimous Voting.
 4. Weighted Voting.

The family of stacking methods is in [Essence of Stacking Ensembles for Machine Learning](https://machinelearningmastery.com/essence-of-stacking-ensembles-for-machine-learning/):

 1. Voting Ensembles
 2. Weighted Average
 3. Blending Ensemble
 4. Super Learner Ensemble

Instead of one fixed combination, the choice of models can depend on the input. [Dynamic Ensemble Selection (DES) for Classification in Python](https://machinelearningmastery.com/dynamic-ensemble-selection-in-python/) — Dynamic Ensemble Selection algorithms operate much like DCS algorithms, except predictions are made using votes from multiple classifier models instead of a single best model. In effect, each region of the input feature space is owned by a subset of models that perform best in that region. The post's outline:

 1. k-Nearest Neighbor Oracle (KNORA) With Scikit-Learn
 1. KNORA-Eliminate (KNORA-E)
 2. KNORA-Union (KNORA-U)
 2. Hyperparameter Tuning for KNORA
 1. Explore k in k-Nearest Neighbor
 2. Explore Algorithms for Classifier Pool

A gating model that routes inputs to experts is the same idea made trainable, in [A Gentle Introduction to Mixture of Experts Ensembles](https://machinelearningmastery.com/mixture-of-experts/):

 1. Mixture of Experts
 1. Subtasks
 2. Expert Models
 3. Gating Model
 4. Pooling Method
 2. Relationship With Other Techniques
 1. Mixture of Experts and Decision Trees
 2. Mixture of Experts and Stacking

The last post in the run explains why weak models are worth combining at all. [Strong Learners vs. Weak Learners in Ensemble Learning](https://machinelearningmastery.com/strong-learners-vs-weak-learners-for-ensemble-learning/) — Weak learners are models that perform slightly better than random guessing. Strong learners are models that have arbitrarily good accuracy.

Weak and strong learners are tools from computational learning theory and provide the basis for the development of the boosting class of ensemble methods.

Most practical ensembles are built from trees. Himanshi Singh's tree-based algorithms tutorial covers decision trees, random forest, ensemble methods, and their implementation in R & python: [Vidhya on trees, bagging boosting, gbm, xgb](https://www.analyticsvidhya.com/blog/2016/04/complete-tutorial-tree-based-modeling-scratch-in-python/#three). For speed, [Parallel grad boost treest](http://zhanpengfang.github.io/418home.html) is the Parallel Gradient Boosting Decision Trees project. Aishwarya Singh's guide to ensemble learning techniques, from simple methods to bagging and boosting and the key algorithms, is [A comprehensive guide to ensembles read!](https://www.analyticsvidhya.com/blog/2018/06/comprehensive-guide-for-ensemble-models/), with this outline:

 1. Basic Ensemble Techniques
 2. 2.1 Max Voting
 3. 2.2 Averaging
 4. 2.3 Weighted Average
 5. Advanced Ensemble Techniques
 6. 3.1 Stacking
 7. 3.2 Blending
 8. 3.3 Bagging
 9. 3.4 Boosting
 10. Algorithms based on Bagging and Boosting
 11. 4.1 Bagging meta-estimator
 12. 4.2 Random Forest
 13. 4.3 AdaBoost
 14. 4.4 GBM
 15. 4.5 XGB
 16. 4.6 Light GBM
 17. 4.7 CatBoost

For competitions, the [Kaggler guide to stacking](http://blog.kaggle.com/2016/12/27/a-kagglers-guide-to-model-stacking-in-practice/) is stacking in practice, and [Blending vs stacking](https://www.quora.com/What-are-examples-of-blending-and-stacking-in-Machine-Learning) separates the two terms. A Kaggle ensemble guide used to sit here too; its address is kept at the end of the page.

### [Ensembles in WEKA](http://machinelearningmastery.com/use-ensemble-machine-learning-algorithms-weka/)

With the landscape laid out, the WEKA post by Jason Brownlee puts the five main ensembles side by side, since WEKA makes so many of them available. Bagging is random sample selection and multi classifier training. Random forest is random feature selection for each tree, multi tree training. Boosting means creating stumps, each new stump tries to fix the previous error, at last combining results using new data, each model is assigned a skill weight and accounted for in the end. Voting is a majority vote, any set of algorithms within weka, results combined via mean or some other way. Stacking is the same as voting but combining predictions using a meta model is used.

### BAGGING - bootstrap aggregating

Bagging is the first of those to unpack, because it is the simplest. [Bagging](https://www.youtube.com/watch?v=2Mg8QD0F1dQ&list=PLAwxTw4SYaPnIRwl6rad_mYwEk4Gmj7Mx&index=192) — best example so far, create m bags, put n’\<n samples (60% of n) in each bag — with replacement which means that the same sample can be selected twice or more, query from the test (x) each of the m models, calculate mean, this is the classification. The figure below draws those bags, copied from the original hosted image.

<figure><img src="../.gitbook/assets/gimg-f2649232c408.png" alt=""><figcaption><p>Bagging.</p><p>Credit: <a href="https://lh5.googleusercontent.com/U0_wGc2DQhx1TYC_ntWSyW9J0XtJJwP4bZ8ONOLgbqb4LM0K7c6-As1HX9wT0LGRON6sOvl3l-WeEOOmuTCupNN3q8Q_kQU8Y1nhhBi6-Of2bcajJfVjhqRRcY-qudAm_u3jXOuF">copied from the original hosted image</a>.</p></figcaption></figure>

Overfitting — not an issue with bagging, as the mean of the models actually averages or smoothes the “curves”. Even if all of them are overfitted. The next figure shows that smoothing.

<figure><img src="../.gitbook/assets/gimg-2eb3a4109a08.png" alt=""><figcaption><p>Bagging and overfitting.</p><p>Credit: <a href="https://lh4.googleusercontent.com/KOj9utriFKEjOxhw8hFE2iX8gq5ljjBHruuhH1Q-deWVPYrEA2RHWaAhKfs-Q1XivON_F7KA3vXL4Mo-GqI4OZTgi0WhC9iNdo4IoOSxQ8gUyoa_F56TOFiXf-hgMsdIFGWLoq6k">copied from the original hosted image</a>.</p></figcaption></figure>

A random forest is bagging over trees, and in xgboost it is one round of many parallel trees:

#Random Forest™ - 1000 trees

bst <- xgboost(data = train$data, label = train$label, max\_depth = 4, num\_parallel\_tree = 1000, subsample = 0.5, colsample\_bytree =0.5, nrounds = 1, objective = "binary:logistic")

### BOOSTING

Bagging trains its models independently; boosting trains them in sequence, each one focused on what the previous ones got wrong. Mastery on using [all the boosting algorithms](https://machinelearningmastery.com/gradient-boosting-with-scikit-learn-xgboost-lightgbm-and-catboost/): Gradient Boosting with Scikit-Learn, XGBoost, LightGBM, and CatBoost

Adaboost: similar to bagging, create a system that chooses from samples that were modelled poorly before.

1. create bag\_1 with n’ features \<n with replacement, create the model\_1, test on ALL train.
2. Create bag\_2 with n’ features with replacement, but add a bias for selecting from the samples that were wrongly classified by the model\_1. Create a model\_2. Average results from model\_1 and model\_2. I.e., who was classified correctly or not.
3. Create bag\_3 with n’ features with replacement, but add a bias for selecting from the samples that were wrongly classified by the model\_1+2. Create a model\_3. Average results from model\_1, 2 & 3 I.e., who was classified correctly or not. Iterate onward.
4. Create bag\_m with n’ features with replacement, but add a bias for selecting from the samples that were wrongly classified by the previous steps.

The figure below shows those reweighted rounds, copied from the original hosted image.

<figure><img src="../.gitbook/assets/gimg-4ad806b940e4.png" alt=""><figcaption><p>Boosting.</p><p>Credit: <a href="https://lh5.googleusercontent.com/iwKa08rChrddn1TM9GoSwmc3gGfxhUbOnPpwHoBS8YHEwUPUOkHifHAO88DR2uiDgRg1VL-dgmnQ2NWFFPJ4CTWvoYdFtBCW-feiBX8SdZ1waY0VkGYclr_m48OzHazmHWrNV3G-">copied from the original hosted image</a>.</p></figcaption></figure>

In xgboost, boosting is the same call with sequential rounds instead of parallel trees:

#Boosting - 3 rounds

bst <- xgboost(data = train$data, label = train$label, max\_depth = 4, nrounds = 3, objective = "binary:logistic")

The two settings side by side:

RF1000: - max\_depth = 4, num\_parallel\_tree = 1000, subsample = 0.5, colsample\_bytree =0.5, nrounds = 1, nthread = 2

XG: nrounds = 10, max\_depth = 4, eta = 0.5, nthread = 2

### Gradient Boosting Classifier

Gradient boosting generalizes that reweighting into fitting the gradient of a loss, and the practical question is how scikit-learn's GBC compares with XGB. [Loss functions and GBC vs XGB](https://stats.stackexchange.com/questions/202858/loss-function-approximation-with-taylor-expansion) is the question on how XGBoost approximates its loss function with a Taylor expansion, one of the key steps for fast calculation. [Why is XGB faster than SK GBC](https://datascience.stackexchange.com/questions/10943/why-is-xgboost-so-much-faster-than-sklearn-gradientboostingclassifier) starts from XGBClassifier handling 500 trees in 43 seconds while GradientBoostingClassifier handles only 10 trees in about a minute. A good XGB vs GBC tutorial used to be linked here and is kept at the end of the page. [XGB vs GBC](https://stats.stackexchange.com/questions/282459/xgboost-vs-python-sklearn-gradient-boosted-trees) asks whether XGBoost works the same way as sklearn's gradient boosted trees, but faster, or whether there are fundamental differences.

### XGBOOST

That comparison makes XGBoost the next stop. What is XGBOOST? — XGBoost is an optimized distributed gradient boosting system designed to be highly efficient, flexible and portable [#2nd link](http://dmlc.cs.washington.edu/xgboost.html). The first link on that question no longer opens and is kept at the end of the page.

Because boosting keeps fitting the errors, it is fair to ask whether it overfits: [Does it cause overfitting?](https://stats.stackexchange.com/questions/20714/does-ensembling-boosting-cause-overfitting) is a question about accuracy changing when boosting is turned on with 400 ensembles. The [Authors Youtube lecture.](https://www.youtube.com/watch?v=Vly8xGnNiWs) is XGBoost A Scalable Tree Boosting System, June 02, 2016, from Real Data Science USA (formerly DataScience.LA). The code is on [GIT here](https://github.com/dmlc/xgboost): dmlc/xgboost, a scalable, portable and distributed gradient boosting (GBDT, GBRT or GBM) library for Python, R, Java, Scala, C and more. A How to use XGB tutorial on medium (comparison to GBC) is kept at the end of the page.

For hands-on use, the [How to code tutorial](https://www.youtube.com/watch?v=87xRqEAx6CY), short and makes sense, with info about the parameters: threads, rounds, tree height, loss function, error, and cross fold. The [Beautiful Video Class about XGBOOST](https://www.youtube.com/playlist?list=PLZnYQQzkMilqTC12LmnN4WpQexB9raKQG) — mostly practical in jupyter but with some insight about the theory. [Machine learning mastery](http://machinelearningmastery.com/gentle-introduction-xgboost-applied-machine-learning/) has A Gentle Introduction to XGBoost for Applied Machine Learning.

XGBoost can also run from WEKA. WekaMOOC's Advanced Data Mining with Weka lesson on setting up R covers [R Installation in Weka](https://www.youtube.com/watch?v=EGwHXC3baWU&list=PLm4W7_iX_v4Msh-7lDOpSFWHRYU_6H5Kx&index=15), then XGBOOST in weka through R

Parameters for weka mlr class.xgboost. The forum thread on them is kept at the end of the page; the reference is the R package manual for xgboost, Extreme Gradient Boosting, the R interface to Chen & Guestrin's implementation:

- [https://cran.r-project.org/web/packages/xgboost/xgboost.pdf](https://cran.r-project.org/web/packages/xgboost/xgboost.pdf)
- Here is an example configuration for multi-class classification:
- weka.classifiers.mlr.MLRClassifier -learner “nrounds = 10, max\_depth = 2, eta = 0.5, nthread = 2”
- classif.xgboost -params "nrounds = 1000, max\_depth = 4, eta = 0.05, nthread = 5, objective = \\"multi:softprob\\"

Copy: nrounds = 10, max\_depth = 2, eta = 0.5, nthread = 2

Special case of random forest using XGBOOST: the vignette for it is kept at the end of the page, and the RF1000 setting above is the same idea.

## CatBoost

XGBoost is not the last word in gradient boosting. CatBoost is based on gradient boosting, a technique developed by Yandex that outperforms many existing boosting algorithms like XGBoost and Light GBM; that is the (great) [what is so special?](https://hanishrohit.medium.com/whats-so-special-about-catboost-335d64d754ae) post. [the fastest algo](https://medium.com/almabetter/catboost-the-fastest-algorithm-c21d44f8b990) is a deep dive into what it actually is, and [a new game in ML](https://affine.medium.com/catboost-a-new-game-of-machine-learning-72a7dcea0ac4) places it among gradient boosted decision trees and random forest, one of the best ML models for tabular heterogeneous datasets. A fourth post, use it here is why, is kept at the end of the page.

## Deprecated links

{% hint style="warning" %}
These links and images no longer work. The original wording is kept here. A same-resource copy, when one was checked, is used above.
{% endhint %}

- Kaggle ensemble guide. This address no longer opens: https://mlwave.com/kaggle-ensembling-guide/
- What is XGBOOST?. This address no longer opens: http://homes.cs.washington.edu/~tqchen/2016/03/10/story-and-lessons-behind-the-evolution-of-xgboost.html
- How to use XGB tutorial on medium (comparison to GBC). This address no longer opens: https://towardsdatascience.com/boosting-algorithm-xgboost-4d9ec0207d
- Parameters for weka mlr class.xgboost. This address no longer opens: http://weka.8497.n7.nabble.com/XGBoost-in-Weka-through-R-or-Python-td40282.html
- Special case of random forest using XGBOOST. This address no longer opens: https://github.com/dmlc/xgboost/blob/master/R-package/vignettes/discoverYourData.Rmd#special-note-what-about-random-forests
- Good XGB vs GBC tutorial. This address no longer opens: https://towardsdatascience.com/boosting-algorithm-xgboost-4d9ec0207d
- use it here is why. This address no longer opens: https://towardsdatascience.com/you-should-use-catboost-heres-why-72f124dcdad7
