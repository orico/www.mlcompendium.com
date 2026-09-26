# Weakly Supervised

When labels are incomplete, inaccurate, or inexact rather than fully supervised, the model has to learn from weak signals. The page moves from a survey of that setting, to Snorkel and programmatic labeling, to papers on test-time training, noisy labels, and multiple instance learning (MIL), and ends with semi-weak supervision at scale.

The same notes are in [Label Propagation / Spreading](label-algorithms.md#label-propagation--spreading) and [Semi Supervised](semi-supervised.md).

The smallest case shows why weak supervision matters. Text classification with extremely small datasets relies heavily on feature engineering methods such as number of hashtags, number of punctuations and other insights that are really good for this type of text; the article behind that note is kept at the end of the page.

The map of the whole setting is a great [review paper](https://pdfs.semanticscholar.org/3adc/fd254b271bcc2fb7e2a62d750db17e6c2c08.pdf) from Nanjing University's National Key Laboratory for Novel Software Technology, great for weakly supervision. It discusses incomplete supervision, inaccurate, and inexact labels, and active learning.

With the survey's three kinds of weakness named, Stanford's answer is to write labels as programs. Stanford on weakly supervised learning had an older post, kept at the end of the page. [Stanford ai on snorkel](http://ai.stanford.edu/blog/weak-supervision/) is "Weak Supervision: A New Programming Paradigm for Machine Learning", which starts from deep learning models reaching state-of-the-art scores without hand-engineered features and asks where their training labels come from. [Intro to Snorkel](https://medium.com/@towardsai/data-centric-ai-with-snorkel-ai-the-enterprise-ai-platform-a8ed0803c24c) is "Data-Centric AI With Snorkel AI: The Enterprise AI Platform", which treats the data rather than the model specifics as the arbiter of whether AI deployment succeeds. Hazy research on weak and snorkel also had a post, kept at the end of the page.

Labels can also be weak at test time, noisy in training, or given only for groups. [Out of distribution generalization using test-time training](https://arxiv.org/abs/1909.13231) — the first case — "Test-time training turns a single unlabeled test instance into a self-supervised learning problem, on which we update the model parameters before making a prediction on this instance. "

[Learning Deep Networks from Noisy Labels with Dropout Regularization](https://arxiv.org/pdf/1705.03419.pdf) — the noisy-label case — "Large datasets often have unreliable labels—such as those obtained from Amazon’s Mechanical Turk or social media platforms—and classifiers trained on mislabeled datasets often exhibit poor performance. We present a simple, effective technique for accounting for label noise when training deep neural networks. We augment a standard deep network with a softmax layer that models the label noise statistics. Then, we train the deep network and noise model jointly via end-to-end stochastic gradient descent on the (perhaps mislabeled) dataset. The augmented model is overdetermined, so in order to encourage the learning of a non-trivial noise model, we apply dropout regularization to the weights of the noise model during training. Numerical experiments on noisy versions of the CIFAR-10 and MNIST datasets show that the proposed dropout technique outperforms state-of-the-art methods."

[Distill to label weakly supervised instance labeling using knowledge distillation](https://arxiv.org/pdf/1907.12926.pdf) — the inexact case, where only image-level labels exist — “Weakly supervised instance labeling using only image-level labels, in lieu of expensive fine-grained pixel annotations, is crucial in several applications including medical image analysis. In contrast to conventional instance segmentation scenarios in computer vision, the problems that we consider are characterized by a small number of training images and non-local patterns that lead to the diagnosis. In this paper, we explore the use of multiple instance learning (MIL) to design an instance label generator under this weakly supervised setting. Motivated by the observation that an MIL model can handle bags of varying sizes, we propose to repurpose an MIL model originally trained for bag-level classification to produce reliable predictions for single instances, i.e., bags of size 1. To this end, we introduce a novel regularization strategy based on virtual adversarial training for improving MIL training, and subsequently develop a knowledge distillation technique for repurposing the trained MIL model. Using empirical studies on colon cancer and breast cancer detection from histopathological images, we show that the proposed approach produces high-quality instance-level prediction and significantly outperforms state-of-the MIL methods.”

The last step joins weak labels with unlabeled data at scale. Researchers from Facebook AI Research (FAIR) have announced their work on a new concept in deep learning called Semi-weakly Supervised Learning; [Yet another article summarising FAIR](https://neurohive.io/en/state-of-the-art/semi-weakly-supervised-learning-increasing-classification-accuracy-with-billion-scale-unlabeled-images/) is Dane Mitrev's summary of how it increases classification accuracy with billion-scale unlabeled images.

## Deprecated links

{% hint style="warning" %}
These links and images no longer work. The original wording is kept here. A same-resource copy, when one was checked, is used above.
{% endhint %}


- Text classification with extremely small datasets. This address no longer opens: https://towardsdatascience.com/text-classification-with-extremely-small-datasets-333d322caee2
- Stanford on. This address no longer opens: https://dawn.cs.stanford.edu/2017/07/16/weak-supervision/
- Hazy research on weak and snorkel. This address no longer opens: https://hazyresearch.github.io/snorkel/blog/ws_blog_post.html
