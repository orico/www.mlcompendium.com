# Active Learning Algorithms

This page is about active-style learning when labels arrive in large streams.
It starts with the Passive Aggressive classifier for massive data streams such as Twitter-scale text.

The same notes are in [Active Learning](../problem-framing/active-learning.md) and [Online Learning](../problem-framing/online-learning.md).

### Passive Aggressive classifier

This section is the Passive Aggressive (PA) classifier for massive data streams.

1. [The Passive Aggressive](https://www.quora.com/Classification-machine-learning-What-is-an-intuitive-explanation-of-the-Passive-Aggressive-classifier) (PA) algorithm is perfect for classifying massive streams of data (e.g. Twitter). It is easy to implement and very fast, but does not provide global guarantees like the support-vector machine (SVM).
- Text Classification 3: Passive Aggressive Algorithm, by Victor Lavrenko. [YouTube, seems like active learning in stream..?](https://www.youtube.com/watch?v=TJU8NfDdqNQ)
