# Information Theory

A model that ranks classes or splits trees needs a measure of surprise and a measure of how wrong a predicted distribution is. The page starts with entropy and information gain, moves through the cross-entropy and divergence family and softmax, and then turns to tools, tutorials, time-series entropy, and complement objective training.
The same notes are in [Collocation](../language-ai/foundation-nlp.md#collocation) and [MUTUAL INFORMATION COEFFICIENT](dependence-and-selection.md#mutual-information-coefficient).

## Entropy / Information Gain

Surprise comes first, because every later measure on the page is built from it.

The same notes are in [Active Learning](../problem-framing/active-learning.md).

Shannon entropy in python is, basically, entropy(value counts): count how often each value appears, turn the counts into probabilities, and sum p log p. MachineLearningMastery's "A Gentle Introduction to Information Entropy" is the [Mastery on plogp entropy function](https://machinelearningmastery.com/what-is-information-entropy/) walkthrough, and the [Entropy functions](https://gist.github.com/jaradc/eeddf20932c0347928d0da5a09298147) gist shows four different ways to calculate entropy in Python.

## Cross entropy, relative ent, KL-D, JS-D, soft max

Entropy measures one distribution; training needs a measure between two, the true one and the predicted one.

The same notes are in [Comparing distributions (distance methods)](distribution.md#comparing-distributions-distance-methods), [LOSS](../deep-learning/deep-neural-nets.md#loss), and [LOSS IN KERAS](../deep-learning/deep-neural-frameworks.md#loss-in-keras).

Kullback–Leibler divergence is a very useful way to measure the difference between two probability distributions, and Count Bayesie's "Kullback-Leibler Divergence Explained" is [A really good explanation on all of them](https://www.countbayesie.com/blog/2017/5/9/kullback-leibler-divergence-explained). From the loss side, Raúl Gómez Bruballa's post on categorical and binary cross-entropy, softmax loss, logistic loss, focal loss, and all those confusing names is [Another good one on all of them](https://gombru.github.io/2018/05/23/cross_entropy_loss/). MachineLearningMastery's "A Gentle Introduction to Cross-Entropy for Machine Learning" is the [mastery on a gentle intro to CE](https://machinelearningmastery.com/cross-entropy-for-machine-learning/), and the [Mastery on entropy](https://machinelearningmastery.com/divergence-between-probability-distributions/) follow-up covers divergence between distributions: kullback leibler divergence (asymmetry), jensen-shannon divergence (symmetry) (has code).

The same family is tied together in [Entropy, mutual information and KL Divergence by AurelienGeron](https://www.techleer.com/articles/496-a-short-introduction-to-entropy-cross-entropy-and-kl-divergence-aurelien-geron/). Gensim also has notes on divergence metrics such as KL jaccard etc, pros and cons, lda is a mess on small data. For computing it by hand, the Stack Exchange question on calculating KL divergence in Python, where every pair of lists returned the same value, is the [Advise on KLD](https://datascience.stackexchange.com/questions/9262/calculating-kl-divergence-in-python)ivergence thread. Neural machine translation using pytorch and CE is where the same loss shows up in a full model.

## Softmax

Cross-entropy needs a predicted distribution to compare against, and softmax is what turns a network's scores into one.

The same notes are in [ACTIVATION FUNCTIONS](../deep-learning/deep-neural-nets.md#activation-functions) and [Temperature](../responsible-ai/calibration.md#temperature).

[Understanding softmax](https://medium.com/data-science-bootcamp/understand-the-softmax-function-in-minutes-f3a59641e86d) is Uniqtech Learning's short explanation of the activation function that turns logits into probabilities that sum to one. LJ V. MIRANDA's notebook, [Softmax and negative likelihood (NLL)](https://ljvmiranda921.github.io/notebook/2017/08/13/softmax-and-the-negative-log-likelihood/), explains the softmax function, its relationship with the negative log-likelihood, and its derivative during backpropagation. The naming question is settled in [Softmax vs cross entropy](https://www.quora.com/Is-the-softmax-loss-the-same-as-the-cross-entropy-loss) - Softmax loss and cross-entropy loss terms are used interchangeably in industry. Technically, there is no term as such Softmax loss. people use the term "softmax loss" when referring to "cross-entropy loss". The softmax classifier is a linear classifier that uses the cross-entropy loss function. In other words, the gradient of the above function tells a softmax classifier how exactly to update its weights using some optimization like [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent).

The softmax() part simply normalises your network predictions so that they can be interpreted as probabilities. Once your network is predicting a probability distribution over labels for each input, the log loss is equivalent to the cross entropy between the true label distribution and the network predictions. As the name suggests, softmax function is a “soft” version of max function. Instead of selecting one maximum value, it breaks the whole (1) with maximal element getting the largest portion of the distribution, but other smaller elements getting some of it as well.

This property of softmax function that it outputs a probability distribution makes it suitable for probabilistic interpretation in classification tasks.

Cross entropy indicates the distance between what the model believes the output distribution should be, and what the original distribution is. Cross entropy measure is a widely used alternative of squared error. It is used when node activations can be understood as representing the probability that each hypothesis might be true, i.e. when the output is a probability distribution. Thus it is used as a loss function in neural networks which have softmax activations in the output layer.

## Tools

Once the measures are clear, computing them should not mean writing them from scratch. [EntroPy](https://raphaelvallat.com/entropy/build/html/index.html) is the entropy 0.1.3 documentation, and [AntroPy](https://raphaelvallat.com/antropy/) has its own 0.2.2 documentation; the AntroPy [[Git](https://github.com/raphaelvallat/antropy) repo describes it as entropy and complexity of (EEG) time-series in Python. [PyInform](https://github.com/ELIFE-ASU/PyInform) [[Docs](https://elife-asu.github.io/PyInform/index.html)]- PyInform is a python library of information-theoretic measures for time series data. PyInform is backed by the [Inform](https://github.com/elife-asu/inform) C library, a cross platform library for information analysis of dynamical systems. [PyEntropy](https://github.com/nikdon/pyEntropy) is one more package, entropy for Python.

## Tutorials

The libraries compute the numbers; the tutorials show why a decision tree wants them.

The same notes are in [CART TREES](../predictive-ml/decision-trees.md#cart-trees).

The scikit-learn decision tree learning tutorial on entropy, Gini, and information gain is the [Great tutorial on all of these topics](https://www.bogotobogo.com/python/scikit-learn/scikt_machine_learning_Decision_Tree_Learning_Informatioin_Gain_IG_Impurity_Entropy_Gini_Classification_Error.php).

[Entropy](https://www.techleer.com/articles/496-a-short-introduction-to-entropy-cross-entropy-and-kl-divergence-aurelien-geron/) - lack of order or lack of predictability ([excellent slide lecture by Aurelien Geron](https://www.youtube.com/watch?time_continue=3&v=ErfnhcEV1O8)). The figures below come from that short introduction to entropy, cross-entropy, and KL-divergence.

<figure><img src="../.gitbook/assets/gimg-102a11303beb.png" alt=""><figcaption><p>Entropy, cross-entropy, and KL divergence (Aurelien Geron).</p><p>Credit: <a href="https://lh6.googleusercontent.com/_MSZGPguSXitn80COZLJ3rOIScBmTXNR6LIOLt3UiyfwNYeTQHUOAVzK1bpaSeoHRPImGnJiHFqsS8Tl3ETkGs32KNgDWwVpJ3nTfxJ7gfzambo0AwY8VBvAKwDKK-7GWoOLdONT">copied from the original hosted image</a>.</p></figcaption></figure>

<figure><img src="../.gitbook/assets/gimg-785fe96bfd79.png" alt=""><figcaption><p>Entropy, cross-entropy, and KL divergence continued.</p><p>Credit: <a href="https://lh3.googleusercontent.com/2c0wvDS4SFXYjHPKiPCtwyW488sV1aMN8MGdUavZ64n1bVlxvJPPqG5oaodPIRgHk-sMNO46s57c8yoqtiMu_kLG6LPbe4SrK--bt9ro6Kc7WQpiaMukMV04wsOFXfa6wliDhff8">copied from the original hosted image</a>.</p></figcaption></figure>


Cross entropy will be equal to entropy if the probability distributions of p (true) and q(predicted) are the same. However, if cross entropy is bigger (known as relative_entropy or kullback leibler divergence)

<figure><img src="../.gitbook/assets/gimg-f763f8ab0174.png" alt=""><figcaption><p>Relative entropy / Kullback–Leibler divergence when p and q differ.</p><p>Credit: <a href="https://lh5.googleusercontent.com/JwW1SuPBqCiI0G-NG2V24DysK-j_ND-xSXHVimiNfq4cCzrTR47qcyHJLcngywO6_tVLd9wLVAHucSMBbm3Cluxkybv1Jj6icXyEvt4o3tmfnx2jZe1H9Z7Hvp-4Mqfr0ifvQAtK">copied from the original hosted image</a>.</p></figcaption></figure>

<figure><img src="../.gitbook/assets/gimg-2a9b4a8685c1.png" alt=""><figcaption><p>Cross-entropy versus entropy illustration.</p><p>Credit: <a href="https://lh4.googleusercontent.com/OGcrihHtrOv1-dODvqwJjsOXbP9fB_t8EIYmj11l8qJL61_I2gg1h9wW0kiEiRDaDoBT6QXxqk5oZncfXK5_un44bYXWa9iTjjsuw8R2t5l5YyrNnQ6fADE1txRRRKvOc7n8KtOQ">copied from the original hosted image</a>.</p></figcaption></figure>


In this example we want the cross entropy loss to be zero, i.e., when we have a one hot vector and a predicted vector which are identical, i.e., 100% in the same class for predicted and true, we get 0. In all other cases we get some number that gets larger if the predicted class probability is lower than zero as seen here:

<figure><img src="../.gitbook/assets/gimg-7f0ca579b802.png" alt=""><figcaption><p>Cross-entropy loss with one-hot true and predicted class probabilities.</p><p>Credit: <a href="https://lh4.googleusercontent.com/BJTEdxhb4RSPIib7CEIm0-ti8vcZtbEL0metallPrMltfR4WC2ADmx6oUaPp67akBGXiyF-7mHL_tQRSucIsVLy-8LXCmEwz5euV4c0lqJhqzgg6XR09Zpv9PBJ7wT4QAmMMrBcd">copied from the original hosted image</a>.</p></figcaption></figure>


Formula for 2 classes:

<figure><img src="../.gitbook/assets/gimg-2aef43320fd6.png" alt=""><figcaption><p>Cross-entropy formula for 2 classes.</p><p>Credit: <a href="https://lh4.googleusercontent.com/8OIzaeni1DtdFjaoyA3K0hAM_cnkgLiwiDFI3FC1iUNIx6sQfq0yum1TR4dV93282q-lBUgf6jWVfHWovjtlvQ9CjKFa2vRN_xyZGuUnasnuniv2FNx6uDmJwpaEAjs-BGOjYO8b">copied from the original hosted image</a>.</p></figcaption></figure>


NOTE: Entropy can be generalized as a formula for N > 2 classes:

<figure><img src="../.gitbook/assets/gimg-f9ecbbfbc2b3.png" alt=""><figcaption><p>Entropy generalized as a formula for N > 2 classes.</p><p>Credit: <a href="https://lh6.googleusercontent.com/N-CK4gLV67dfxLjDbty1SnsWsNlBm2GLM2TXL8HXef2EzsFZxvY4urwUnFiSE2A4SSBRQrFKuluQzb7cm0mTKUIUuwxbqj1NbC-4igh3pGIMrBjSFN7lppKJAktDvLNNJGflwo_A">copied from the original hosted image</a>.</p></figcaption></figure>


The same loss logic now carries over to trees. The notes below follow an awesome pdf tutorial whose address no longer opens and is kept at the end of the page. (We want to grow a simple tree): a good attribute prefers attributes that split the data so that each successor node is as pure as possible, i.e., the distribution of examples in each node is so that it mostly contains examples of a single class. In other words: We want a measure that prefers attributes that have a high degree of „order“. Maximum order: All examples are of the same class. Minimum order: All classes are equally likely → Entropy is a measure for (un-)orderedness. Another interpretation: Entropy is the amount of information that is contained, and all examples of the same class → no information.

<figure><img src="../.gitbook/assets/gimg-c3fc62bc4be7.png" alt=""><figcaption><p>Entropy as unorderedness in the class distribution of S.</p><p>Credit: <a href="https://lh3.googleusercontent.com/s4tfIeHpR4H9GimwTPjFVoV0nCKwEUQYRFpz93x-d5jZCxDFIub8jiK7PFbkSNU1X__OXHK7XLSH_BO0xUQIjS6HEnHfUEiuY0KWJpb1ZX0NowqyKG4A2guA3wN_b52UKeVluv9f">copied from the original hosted image</a>.</p></figcaption></figure>


Entropy is the amount of unorderedness in the class distribution of S. In the IMAGE above it has its maximal value when the equal class distribution holds, and its minimal value when only one class is in S.

That is the entropy of one node; a split has one node per category. So basically if we have the outlook attribute and it has 3 categories, we calculate the entropy for E(feature=category) for all 3.

<figure><img src="../.gitbook/assets/gimg-fbedcd82edef.png" alt=""><figcaption><p>Entropy for each category of an attribute such as Outlook.</p><p>Credit: <a href="https://lh5.googleusercontent.com/aTcovXALgA4bT15GabT1Z3ce7GpKoMkAUVAly_v7Jn2EgcKmSr2eq18ANSU1TxHJt2-_Lfk-fSoiF9DimirF57D0-bNQrAtfBp3hT3205e-C4XQEn87w2lu8m8LZl3f7RYlCtnIn">copied from the original hosted image</a>.</p></figcaption></figure>


INFORMATION: The I(S,A) formula below.

What we actually want is the average entropy of the entire split, that corresponds to an entire attribute, i.e., OUTLOOK (sunny & overcast & rainy)

<figure><img src="../.gitbook/assets/gimg-15b45e8fd2fd.png" alt=""><figcaption><p>The I(S,A) average entropy of an entire attribute split.</p><p>Credit: <a href="https://lh6.googleusercontent.com/DikgymC_A5YqhfvObk9JcAMdHrnVIhNksx20IMI7yMZKxI-vLQeU2lAQOxY8tu78cEq_DgpkeMW63UBaL-2fkjpF-J5HHSo5BirtthZou8KUFqHwF6vHFOj7426FMgcRjZk_-Ran">copied from the original hosted image</a>.</p></figcaption></figure>


Information Gain: is actually what we gain by subtracting information from the entropy.

In other words we find the attributes that maximizes that difference, in other other words, the attribute that reduces the unorderness / lack of order / lack of predictability.

The BIGGER GAIN is selected.

<figure><img src="../.gitbook/assets/gimg-9eeba04d914d.png" alt=""><figcaption><p>Information Gain as entropy minus information.</p><p>Credit: <a href="https://lh6.googleusercontent.com/7Jalf8E7EozkxR_lUjJ9RFlpoh8BcOy0Vojxjjxa8Us5pOOF6uRpXK6_ddm2PkG5azDfDcDfgZLDrpaFNUye343EJ8xpro8AS9uoPxK6hGyHsCIkEzwAnEe74xtRzZUz9ph9v_Mz">copied from the original hosted image</a>.</p></figcaption></figure>


There are some properties to Entropy that influence INFO GAIN (?):

<figure><img src="../.gitbook/assets/gimg-16f412550095.png" alt=""><figcaption><p>Properties of Entropy that influence Information Gain.</p><p>Credit: <a href="https://lh4.googleusercontent.com/36h-4HJT2n9WqgzSVKAqlDF55qzxHEGhUJCMR80bXjQ-pfShcmxDZhegYKVugG-uQwmIal_jUWyhU0GWdqtfNIg9su1pY0HIXCt517e8-HpJRllCoInM_TeI3cctpNUKxI6yY455">copied from the original hosted image</a>.</p></figcaption></figure>


There are some disadvantages with INFO GAIN, done use it when an attribute has many number values, such as “day” (date wise) 05/07, 06/07, 07/07..31/07 etc.

Information gain is biased towards choosing attributes with a large number of values, and that bias causes overfitting and fragmentation.

<figure><img src="../.gitbook/assets/gimg-b7008514d45b.png" alt=""><figcaption><p>Information Gain bias toward attributes with many values.</p><p>Credit: <a href="https://lh5.googleusercontent.com/vGjXAG-G2hmkJkt4xhcxycm5BG6LM-sRPOWnXOrXuCFpSGOQSBcL2mZUoVRhsqRTrr83wXKRDp5rF2hqYn1DGnJdIGvWezoSxy9zOmy2e5Yqc_OIJ6sXXA1YAbZksmY4-f0JWaDp">copied from the original hosted image</a>.</p></figcaption></figure>


We measure Intrinsic information of an attribute, i.e., Attributes with higher intrinsic information are less useful.

We define Gain Ratio as info-gain with less bias toward multi value attributes, ie., “days”

NOTE: Day attribute would still win with the Gain Ratio, Nevertheless: Gain ratio is more reliable than Information Gain

<figure><img src="../.gitbook/assets/gimg-03fde770bac0.png" alt=""><figcaption><p>Gain Ratio as information gain with less multi-value bias.</p><p>Credit: <a href="https://lh3.googleusercontent.com/iGhWawPGmntKD_u8zSa0IPkDggMDrKh6NupAR_acmknUxDWiFfJIfOuZTtXYuMAJq6wX7-lCLBAxVXkqQFbVAElFpoXd1WZfGlZgpch0aeBU87EQxQMf8g3RrFOGL8fuYtrrxBX0">copied from the original hosted image</a>.</p></figcaption></figure>


Therefore, we define the alternative, which is the GINI INDEX. It measures impurity, we define the average Gini, and the Gini Gain.

<figure><img src="../.gitbook/assets/gimg-0c4b7812badf.png" alt=""><figcaption><p>Gini index, average Gini, and Gini Gain.</p><p>Credit: <a href="https://lh4.googleusercontent.com/RbRnfwnEtsIcgYsZah90PVP-DoX0E2qEqBImKmyQGxEMMegWenzsMa2rNa18_F_jXTsscGVFK5X_FX9Vs6pWizuiXOgzSvCxy57a5_ny_48XzB09CWARY7wvbl6O3tYoho_ykza8">copied from the original hosted image</a>.</p></figcaption></figure>


FINALLY, further reading about decision trees and examples of INFOGAIN and GINI here used to point at the same pdf tutorial, which is kept at the end of the page.

Beyond trees, mutual information itself is hard to estimate, and [Variational bounds on mutual information](https://arxiv.org/abs/1905.06922v1) is the arXiv paper "On Variational Bounds of Mutual Information".

## Time series entropy

The tree measures treat each row on its own; a time series asks how predictable the sequence is.

The same notes are in [Timeseries](../predictive-ml/forecasting.md).

[entroPY](https://raphaelvallat.com/entropy/build/html/index.html) - EntroPy is a Python 3 package providing several time-efficient algorithms for computing the complexity of one-dimensional time-series. It can be used for example to extract features from EEG signals. The [Approximate entropy paper](https://journals.physiology.org/doi/pdf/10.1152/ajpheart.2000.278.6.H2039) is behind one of its measures, approximate entropy. The calls below compute each of them on a series x:

```python
print(perm_entropy(x, order=3, normalize=True)) # Permutation entropy
print(spectral_entropy(x, 100, method='welch', normalize=True)) # Spectral entropy
print(svd_entropy(x, order=3, delay=1, normalize=True)) # Singular value decomposition entropy
print(app_entropy(x, order=2, metric='chebyshev')) # Approximate entropy
print(sample_entropy(x, order=2, metric='chebyshev')) # Sample entropy
print(lziv_complexity('01111000011001', normalize=True)) # Lempel-Ziv complexity
```

For measures beyond those, the [PyInform](https://elife-asu.github.io/PyInform/index.html) 0.2.0 documentation lists the information-theoretic measures shown in the two figures below.

<figure><img src="../.gitbook/assets/gimg-15c13866e888.png" alt=""><figcaption><p>PyInform information-theoretic measures.</p><p>Credit: <a href="https://lh3.googleusercontent.com/2XcbUSTQe6BCTd2Hgmj-VU_ErIDRzSbfUucWtiqXRSaPdoYVKtcEs4AwvIjKYoFteF_Ndl5yhdvy24vFX-4x24Bap21_hAyYwDeX0Xh0u5PHUqj9Jc2KacINx6HtckWwNAHEcsMM">copied from the original hosted image</a>.</p></figcaption></figure>

<figure><img src="../.gitbook/assets/gimg-13adb68b8f41.png" alt=""><figcaption><p>PyInform measures continued.</p><p>Credit: <a href="https://lh4.googleusercontent.com/_bAbXFL9VqcqZHmyR8z_MJvV_u6PD_7_AOollUOFLHACmDegc-NeseoJcoBbZw6rBXZJx0NLDqYFwGk6wSs1WBfZ3QWuRN5J_Mq9hL-aSD-UuQi-depGzdPFNqOE07QHGAZ4SAdy">copied from the original hosted image</a>.</p></figcaption></figure>


## Complement Objective Training

Cross-entropy, from earlier on the page, only pushes up the correct class; complement objective training also uses the wrong ones. Complement Objective Training is a simple way to use incorrect-class probabilities and get more from labeled data in PyTorch Lightning. Article by [LightTag](https://www.lighttag.io/blog/complement-objective-training-with-pytorch-lightning/), and the method's [paper](https://arxiv.org/pdf/1903.01182.pdf) is on arXiv.

COT is a technique to effectively provide explicit negative feedback to our model. The technique gives us non-zero gradients with respect to incorrect classes, which are used to update the model's parameters.

COT doesn't replace cross-entropy. It's used as a second training step as follows: We run cross-entropy, and then we do a COT step. We minimize the cross-entropy between our target distribution. That's equivalent to maximizing the likelihood of the correct class. During the COT step, we maximize the entropy of the complement distribution. We pretend that the correct class isn't an option and make the remaining classes equally likely.

But, since the true class is an option, and we're training for it explicitly, maximizing the true classes probability and pushing the remaining classes to be equally likely is actually pushing their probabilities to 0 explicitly, which provides explicit gradients to propagate through our model.

## Deprecated links

{% hint style="warning" %}
These links and images no longer work. The original wording is kept here. A same-resource copy, when one was checked, is used above.
{% endhint %}

- Entropy. This address no longer opens: https://www.techleer.com/articles/496-a-short-introduction-to-entropy-cross-entropy-and-kl-divergence-aurelien-geron/
- awesome pdf tutorial. This address no longer opens: http://www.ke.tu-darmstadt.de/lehre/archiv/ws0809/mldm/dt.pdf
- FINALLY, further reading about decision trees and examples of INFOGAIN and GINI here.. This address no longer opens: http://www.ke.tu-darmstadt.de/lehre/archiv/ws0809/mldm/dt.pdf
