# Active Learning Algorithms

When labels arrive in large streams, the learner has to update as they come instead of waiting for a full dataset. The page starts, and for now ends, with the Passive Aggressive classifier for massive data streams such as Twitter-scale text.

The same notes are in [Active Learning](../problem-framing/active-learning.md) and [Online Learning](../problem-framing/online-learning.md).

### Passive Aggressive classifier

For a stream that never stops, the first algorithm to reach for is a cheap one. [The Passive Aggressive](https://www.quora.com/Classification-machine-learning-What-is-an-intuitive-explanation-of-the-Passive-Aggressive-classifier) (PA) algorithm is perfect for classifying massive streams of data (e.g. Twitter). It is easy to implement and very fast, but does not provide global guarantees like the support-vector machine (SVM). To see it worked on text, Victor Lavrenko's Text Classification 3: Passive Aggressive Algorithm is the [YouTube, seems like active learning in stream..?](https://www.youtube.com/watch?v=TJU8NfDdqNQ) lecture.
