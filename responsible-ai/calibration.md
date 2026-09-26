# Calibration

This page explains why predicted probabilities need calibration and how to adjust them in classical models and neural nets.

The same notes are in [Evaluation Metrics](../evals/evaluation-metrics.md).

[Why do we need to calibrate models, or in other words, dont trust predict_proba to give you probabilities](https://medium.com/data-science/pythons-predict-proba-doesn-t-actually-predict-probabilities-and-how-to-fix-it-f582c21d63fc)

## Classic Model Calibration

This section covers isotonic and Platt scaling, sklearn tools, and when softmax scores mislead.

1. **How do we do isotonic and sigmoid calibration - read** this**, then** [this](http://fastml.com/classifier-calibration-with-platts-scaling-and-isotonic-regression/)**,** [how to use in sklearn](https://stats.stackexchange.com/questions/263393/scikit-correct-way-to-calibrate-classifiers-with-calibratedclassifiercv)
2. How to speed up isotonic regression for sklearn
3. TODO: how to calibrate a DNN (except sklearn wrapper for keras)
4. **Allows us to use the probability as confidence. I.e, Well calibrated classifiers are probabilistic classifiers for which the output of the predict_proba method can be directly interpreted as a confidence level**
5. **(good)** [Probability Calibration Essentials (with code)](https://medium.com/analytics-vidhya/probability-calibration-essentials-with-code-6c446db74265)
6. **The** [Brier score](https://en.wikipedia.org/wiki/Brier_score) **is a** [proper score function](https://en.wikipedia.org/wiki/Scoring_rule#ProperScoringRules) **that measures the accuracy of probabilistic predictions.**
7. [Sk learn](http://scikit-learn.org/stable/auto_examples/calibration/plot_calibration_curve.html#sphx-glr-auto-examples-calibration-plot-calibration-curve-py) **example**
8. [**‘calibrated classifier cv in sklearn**](http://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html#sklearn.calibration.CalibratedClassifierCV) **- The method to use for calibration. Can be ‘sigmoid’ which corresponds to Platt’s method or ‘isotonic’ which is a non-parametric approach. It is not advised to use isotonic calibration with too few calibration samples (<<1000) since it tends to overfit. Use sigmoids (Platt’s calibration) in this case.**

   **However, not all classifiers provide well-calibrated probabilities, some being over-confident while others being under-confident. Thus, a separate calibration of predicted probabilities is often desirable as a postprocessing. This example illustrates two different methods for this calibration and evaluates the quality of the returned probabilities using Brier’s score**
9. **Example** [1](http://scikit-learn.org/stable/auto_examples/calibration/plot_calibration.html#sphx-glr-auto-examples-calibration-plot-calibration-py) **- binary class below,** [2](http://scikit-learn.org/stable/auto_examples/calibration/plot_calibration_multiclass.html#sphx-glr-auto-examples-calibration-plot-calibration-multiclass-py) **- 3 class moving prob vectors to a well defined location,** [3](http://scikit-learn.org/stable/auto_examples/calibration/plot_compare_calibration.html#sphx-glr-auto-examples-calibration-plot-compare-calibration-py) **- comparison of non calibrated models, only logreg is calibrated naturally**

   <figure><img src="../.gitbook/assets/gimg-eab6cc5ec48e.png" alt=""><figcaption><p>Sklearn calibration examples.</p><p>Credit: <a href="https://lh4.googleusercontent.com/pgzEadilkxa1ihkvs-8aw5wBnxfAaBBfLsutGQ38mAWcANEKQEOowO_6A5O6tbaj7DgeRt1vDBk74IYCFBqQX61lTo5YHhFE5NXJu7J5XYYsRzhjLIyoeaPz59WlF4NDDjUNgzsp">copied from the original hosted image</a>.</p></figcaption></figure>
10. [Mastery on why we need calibration](https://machinelearningmastery.com/calibrated-classification-model-in-scikit-learn/)
11. [Why softmax is not good as an uncertainty measure for DNN](https://stats.stackexchange.com/questions/309642/why-is-softmax-output-not-a-good-uncertainty-measure-for-deep-learning-models)
12. [If a model doesn't have probabilities use the decision function](http://scikit-learn.org/stable/auto_examples/calibration/plot_calibration_curve.html#sphx-glr-auto-examples-calibration-plot-calibration-curve-py)

**y_pred = clf.predict(X_test)**

       **if hasattr(clf, "predict_proba"):**

           **prob_pos = clf.predict_proba(X_test)[:, 1]**

       **else:  # use decision function**

           **prob_pos = clf.decision_function(X_test)**

           **prob_pos = \\**

               **(prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())**

## Neural Net Calibration

This section links to neural-network calibration papers, blog posts, and temperature scaling.

1. [Paper: Calibration of modern NN](https://arxiv.org/pdf/1706.04599.pdf)
2. Calibration post

### Temperature

This subsection collects softmax temperature scaling explanations and a BNN dropout trick.

The same notes are in [ACTIVATION FUNCTIONS](../deep-learning/deep-neural-nets.md#activation-functions) and [Softmax](../data/information-theory.md#softmax).

1. (great) [Softmax temperature](https://medium.com/mlearning-ai/softmax-temperature-5492e4007f71) by Harshit
2. [Interactive demo](https://lukesalamone.github.io/posts/what-is-temperature/)
3. [lower level explanation](http://www.kasimte.com/2020/02/14/how-does-temperature-affect-softmax-in-machine-learning.html) by kasim
4. [short explanation](https://medium.com/@majid.ghafouri/why-should-we-use-temperature-in-softmax-3709f4e0161) by Majid
5. [Change temperature in Keras](https://stackoverflow.com/questions/37246030/how-to-change-the-temperature-of-a-softmax-output-in-keras)
6. Calibration can also come in a different flavor, you want to make your algorithm certain, one trick is to use dropout layers when inferring/predicting/classifying, do it 100 times and average the results in some capacity , [see this chapter on BNN](https://docs.google.com/document/d/1dXELAcJn9KCPSRMDvZoumUyHx8K8Yn7wfFxesSpbNCM/edit#heading=h.slqfz2k65bd2)

[How Can We Know When Language Models Know? This paper is about calibration.](http://phontron.com/paper/jiang20lmcalibration.pdf)

“Recent works have shown that language models (LM) capture different types of knowledge regarding facts or common sense. However, because no model is perfect, they still fail to provide appropriate answers in many cases. In this paper, we ask the question “how can we know when language models know, with confidence, the answer to a particular query?” We examine this question from the point of view of calibration, the property of a probabilistic model’s predicted probabilities actually being well correlated with the probability of correctness. We first examine a state-ofthe-art generative QA model, T5, and examine whether its probabilities are well calibrated, finding the answer is a relatively emphatic no. We then examine methods to calibrate such models to make their confidence scores correlate better with the likelihood of correctness through fine-tuning, post-hoc probability modification, or adjustment of the predicted outputs or inputs. Experiments on a diverse range of datasets demonstrate the effectiveness of our methods. We also perform analysis to study the strengths and limitations of these methods, shedding light on further improvements that may be made in methods for calibrating LMs.”

## Deprecated links

{% hint style="warning" %}
These links and images no longer work. The original wording is kept here. A same-resource copy, when one was checked, is used above.
{% endhint %}

- Why do we need to calibrate models, or in other words, dont trust predict_proba to give you probabilities. This address no longer opens: https://towardsdatascience.com/pythons-predict-proba-doesn-t-actually-predict-probabilities-and-how-to-fix-it-f582c21d63fc
- How to speed up isotonic regression for sklearn. This address no longer opens: http://tullo.ch/articles/speeding-up-isotonic-regression/
- **How do we do isotonic and sigmoid calibration - read** this. This address no longer opens: http://tullo.ch/articles/speeding-up-isotonic-regression/
- Calibration post. This address no longer opens: http://geoffpleiss.com/nn_calibration
- this. This address no longer opens: https://web.archive.org/web/2020/http://tullo.ch/articles/speeding-up-isotonic-regression/
- Calibration post. This address no longer opens: https://web.archive.org/web/2020/http://geoffpleiss.com/nn_calibration
