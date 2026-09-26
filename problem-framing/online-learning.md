# Online Learning

Some models cannot wait for a finished dataset: they must update as labeled examples arrive in a stream, including non-stationary settings. The page starts with classical online learning, its tutorials, theory, and tools, and then moves to online deep learning.

The same notes are in [Active Learning Algorithms](../decision-intelligence/active-learning-algorithms.md), [Follow the regularized leader](../decision-intelligence/follow-the-regularized-leader.md), [Incremental Learning](../decision-intelligence/incremental-learning.md), and [Training Strategies](../evals/training-strategies.md).

### Online (Classical) Learning

Classical online learning updates the model as labeled examples arrive in a stream, instead of building one model on the whole dataset at once. If you want to start with OL, [start here](https://dziganto.github.io/data%20science/online%20learning/python/scikit-learn/An-Introduction-To-Online-Machine-Learning/): "An Introduction To Online Machine Learning" begins from batch, or offline, learning, the standard approach you already know, and contrasts it with the online version. Analytics Vidhya's "Introduction to Online Machine Learning: Simplified" is the second introduction, [here](https://www.analyticsvidhya.com/blog/2015/01/introduction-online-machine-learning-simplified-2/).

The theory is Shay Shalev's. [A thesis about online learning](http://ttic.uchicago.edu/~shai/papers/ShalevThesis07.pdf) is his thesis, opening with “How much better to get wisdom than gold, to choose understanding rather than silver”. [Some answers about what is OL,](https://www.quora.com/What-is-the-best-way-to-learn-online-machine-learning) are the Quora answers to "What is the best way to learn online machine learning?", which frame it as models that learn incrementally from data streams and point at production concerns such as latency, concept drift, and evaluation under streaming; the first one actually talks about S.Shalev’s [other paper.](http://www.cs.huji.ac.il/~shais/papers/OLsurvey.pdf), his online learning survey. Online learning — Andrew Ng — coursera is the lecture version, kept at the end of the page.

From theory to production: [Chip Huyen on online prediction & learning](https://huyenchip.com/2020/12/27/real-time-machine-learning.html) argues that machine learning is going real-time, with a 2022 follow-up on its challenges and solutions. For code, [River](https://github.com/online-ml/river/) is a Python library for [online machine learning](https://www.wikiwand.com/en/Online_machine_learning), and the figure below sketches the setting.

<figure><img src="../.gitbook/assets/image (7).png" alt=""><figcaption><p>Online machine learning.</p></figcaption></figure>

### Online Deep Learning (ODL)

After the classical setting, online deep learning adapts neural models from streaming or non-stationary data. The notes here name Hedge back propagation (HDP), Autonomous DL, Qactor — online AL for noisy labeled stream data. The article that covered them is kept at the end of the page.

## Deprecated links

{% hint style="warning" %}
These links and images no longer work. The original wording is kept here. A same-resource copy, when one was checked, is used above.
{% endhint %}


- Hedge back propagation (HDP), Autonomous DL, Qactor. This address no longer opens: https://towardsdatascience.com/online-deep-learning-odl-and-hedge-back-propagation-277f338a14b2
- coursera. This address no longer opens: https://www.coursera.org/learn/machine-learning/lecture/ABO2q/online-learning
