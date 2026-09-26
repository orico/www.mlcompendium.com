# Multi Armed Bandits

An A/B test waits until the end to act on what it learned; a multi-armed bandit keeps allocating trials while learning which arm pays off. The page starts with a book, then bandits in recommender systems, then the business case against conventional A/B testing, and ends with Thompson sampling.

The same notes are in [A/B Testing](a-b-testing.md), [Contextual Bandits](contextual-bandits.md), and [Reinforcement Learning](reinforcement-learning.md).
The same notes are in [Recommender Systems](../ai-product/recommender-systems.md).

The reference to start from is the [Book — bandit algorithms](https://tor-lattimore.com/downloads/book/book.pdf), the free online edition of Bandit Algorithms; its content is the same as the Cambridge University Press print edition, with minor typos corrected. Where the theory meets product, [Bandits for recommender systems](https://eugeneyan.com/writing/bandits/) is Eugene Yan on industry examples, exploration strategies, warm-starting, off-policy evaluation, and more.

The same notes are in [Recommender Systems](../ai-product/recommender-systems.md).

The business argument for bandits is speed. [Rolling out multi armed bandits for fast adaptive experimentation](https://www.bain.com/insights/rolling-out-multiarmed-bandits-for-fast-adaptive-experimentation/) is about how to outperform conventional A/B testing when scaling up personalized messages and services. The algorithm most of that rests on is [Thompson sampling for multi-armed bandit](https://medium.com/@iqra.bismi/thompson-sampling-a-powerful-algorithm-for-multi-armed-bandit-problems-95c15f63a180), a walkthrough of Thompson Sampling for decision-making under uncertainty; it was originally introduced by William R. Thompson in 1933 and is used in online advertising, clinical trials, and recommendation systems.
