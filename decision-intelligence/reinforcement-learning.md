# Reinforcement Learning

A bandit decides one step at a time; reinforcement learning gives the agent a sequence of states and lets it learn by interacting for reward. The page starts with books, platforms, and lectures, then sketches Q-learning, moves to deep RL, and ends with RLHF.

The same notes are in [Contextual Bandits](contextual-bandits.md) and [Multi Armed Bandits](multi-armed-bandits.md).

## Introduction

The entry point is the textbook in both of its states. [Reinforcement Learning: An Introduction second edition WIP](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf) is the second edition while it was still in progress, and the completed [book](http://incompleteideas.net/book/RLbook2020.pdf) is the finished version.

Beside the book, Vidhya has two Faizan Shaikh articles. Vidhya on [Getting ready for AI based gaming agents – Overview of Open Source Reinforcement Learning Platforms](https://www.analyticsvidhya.com/blog/2016/12/getting-ready-for-ai-based-gaming-agents-overview-of-open-source-reinforcement-learning-platforms/) explains reinforcement learning with examples and tours the major platforms such as Deepmind Lab and OpenAI Gym. Vidhya on [Simple Beginner’s guide to Reinforcement Learning & its implementation](https://www.analyticsvidhya.com/blog/2017/01/introduction-to-reinforcement-learning-implementation/) defines reinforcement learning as a type of machine learning where an agent learns to maximize reward by interacting with an environment.

The reward idea also works outside games. ZipRecruiter on [Classifying Job Titles With Noisy Labels Using REINFORCE](https://medium.com/@ziprecruiter.engineering/classifying-job-titles-with-noisy-labels-using-reinforce-ce1a4bde05e2) — Fine-grained job title classification with noisy labels using the REINFORCE algorithm and multi-task learning.

The same notes are in [Unbalanced labels](../problem-framing/label-algorithms.md#unbalanced-labels).

The reason it is here: this article has a very nice trick in adding a reward component to the loss function in order to mitigate for unbalanced class label problem, instead of the usual balancing.

For a full course, David Silver — [Home Page](https://www.davidsilver.uk/teaching/) is his Teaching page, and the Google DeepMind recordings of his RL Course are the lectures in order:

1. [1](https://www.youtube.com/watch?v=2pWv7GOvuf0): Introduction to Reinforcement Learning
2. [2](https://www.youtube.com/watch?v=lfHX2hHRMVQ): Markov Decision Process
3. [3](https://www.youtube.com/watch?v=Nd1-UUMVfz4): Planning by Dynamic Programming
4. [4](https://www.youtube.com/watch?v=PnHCvfgC_ZA): Model-Free Prediction
5. [5](https://www.youtube.com/watch?v=0g4j2k_Ggc4): Model Free Control
6. [6](https://www.youtube.com/watch?v=UoPei5o4fps): Value Function Approximation
7. [7](https://www.youtube.com/watch?v=KHZVXao4qXs): Policy Gradient Methods
8. [8](https://www.youtube.com/watch?v=ItMutbeOHtc): Integrating Learning and Planning
9. [9](https://www.youtube.com/watch?v=sGuiWX07sKw): Exploration and Exploitation
10. [10](https://www.youtube.com/watch?v=kZ_AUmFcZtk): Classic Games

The figure is from those David Silver teaching materials.

<figure><img src="../.gitbook/assets/image (3).png" alt=""><figcaption><p>David Silver teaching materials.</p></figcaption></figure>

The last introductory source frames RL as sequential decisions in general. [Sequential Decision Analytics and Modeling](https://castle.princeton.edu/sdamodeling/) is the second edition from CASTLE, by Warren B. Powell, Professor Emeritus at Princeton University.

### Q-LEARN

With the lectures behind you, Q-learning is the first algorithm to sketch. The problem is a Markov chain problem, (state, action, new state, reward). There is lots of exploration in the beginning, then exploitation, and the method returns optimal policy. For what Q itself is, the Udacity video What is Q is on youtube [here](https://www.youtube.com/watch?v=9m_6q_KECTk).

### Deep Learning

From Q-learning, the next step is RL combined with deep learning, deep Q-learning first. [A review paper about RL in DL](https://arxiv.org/pdf/1701.07274.pdf) is Deep Reinforcement Learning: An Overview on arXiv. [deep Q-learning](https://www.analyticsvidhya.com/blog/2019/04/introduction-deep-q-learning-python/) is Ankit Choudhary's An Introduction To Deep Reinforcement Learning, which builds a deep Q-learning model in Python using keras and gym. The Pytorch tutorials then give runnable versions: [DQN](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html) is the Reinforcement Learning (DQN) Tutorial, [PPO](https://pytorch.org/tutorials/intermediate/reinforcement_ppo.html) is Reinforcement Learning (PPO) with TorchRL, and the [Mario example](https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html) trains a Mario-playing RL agent.

## RLHF

After the RL introduction above, the same reward loop is used to tune language models from human feedback. The same notes are in [Reinforcement Learning for LLM](../generative-ai/large-language-models-llms.md#reinforcement-learning-for-llm).

[illustrated RLHF by Huggingface](https://huggingface.co/blog/rlhf) is Illustrating Reinforcement Learning from Human Feedback (RLHF), from a site that describes itself as on a journey to advance and democratize artificial intelligence through open source and open science.
