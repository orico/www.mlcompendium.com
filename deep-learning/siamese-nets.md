# SIAMESE NETWORKS (one shot)

When there are only one or a few examples per class, a network cannot learn a flat class label, so it learns a similarity between inputs instead. The page follows that idea from the siamese setup and its losses to one-shot learning and then to multi networks.

The same notes are in [N-Shot Learning](../problem-framing/n-shot-learning.md) and [SIAMESE NETWORKS](deep-neural-nets.md#siamese-networks).

The starting point is a Siamese CNN that learns a similarity between images, not to classify; the face-recognition-from-scratch source for that note no longer opens and is kept at the end of the page. The mechanism is explained in [Visual tracking, explains contrastive and triplet loss](https://medium.com/intel-student-ambassadors/siamese-networks-for-visual-tracking-96262eaaba77), Param Popat's post on siamese networks for visual tracking: two or more identical networks share weights and work on two different input vectors to compute comparable output vectors.

With the loss in hand, the author's next notes were One shot learning, very thorough, baseline vs siamese, and What is triplet loss; both sources are kept at the end of the page.

The same sharing idea extends to MULTI NETWORKS. [Google whitening black boxes using multi nets, segmentation and classification](https://medium.com/health-ai/google-deepmind-might-have-just-solved-the-black-box-problem-in-medical-ai-3ed8bc21f636) is Susan Ruyu Qi on Google DeepMind improving the interpretability of the “Black Box” in medical imaging, where it is hard to understand why a model makes a certain diagnosis or recommendation.

## Deprecated links

{% hint style="warning" %}
These links and images no longer work. The original wording is kept here. A same-resource copy, when one was checked, is used above.
{% endhint %}
- Towards Data Science: one-shot-learning-with-siamese-networks-using-keras-17f34e75bb3d. This address no longer opens: https://towardsdatascience.com/one-shot-learning-with-siamese-networks-using-keras-17f34e75bb3d
- Towards Data Science: siamese-network-triplet-loss-b4ca82c1aec8. This address no longer opens: https://towardsdatascience.com/siamese-network-triplet-loss-b4ca82c1aec8
- Siamese CNN, learns a similarity between images, not to classify. This address no longer opens: https://medium.com/predict/face-recognition-from-scratch-using-siamese-networks-and-tensorflow-df03e32f8cd0
