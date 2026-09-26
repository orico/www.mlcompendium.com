# Reinforcement Learning

This page is an introduction to reinforcement learning: agents that learn by interacting for reward.
It covers Q-learning sketches, deep RL pointers, and RLHF after the introductory reading list.

The same notes are in [Contextual Bandits](contextual-bandits.md) and [Multi Armed Bandits](multi-armed-bandits.md).

## Introduction

This section lists books, platforms, lectures, and a noisy-label REINFORCE example as the RL entry point.

- # Second edition, in progress Richard S. [Reinforcement Learning: An Introduction second edition WIP](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)
- & completed [book](http://incompleteideas.net/book/RLbook2020.pdf)
- This article explains reinforcement learning and its examples, by Faizan Shaikh. Vidhya on [Getting ready for AI based gaming agents – Overview of Open Source Reinforcement Learning Platforms](https://www.analyticsvidhya.com/blog/2016/12/getting-ready-for-ai-based-gaming-agents-overview-of-open-source-reinforcement-learning-platforms/)
- Reinforcement learning is a type of machine learning where an agent learns to maximize reward by interacting with an environment, by Faizan Shaikh. Vidhya on [Simple Beginner’s guide to Reinforcement Learning & its implementation](https://www.analyticsvidhya.com/blog/2017/01/introduction-to-reinforcement-learning-implementation/)
4. ZipRecruiter on [Classifying Job Titles With Noisy Labels Using REINFORCE](https://medium.com/@ziprecruiter.engineering/classifying-job-titles-with-noisy-labels-using-reinforce-ce1a4bde05e2) — Fine-grained job title classification with noisy labels using the REINFORCE algorithm and multi-task learning

The same notes are in [Unbalanced labels](../problem-framing/label-algorithms.md#unbalanced-labels).

 -> this article has a very nice trick in adding a reward component to the loss function in order to mitigate for unbalanced class label problem, instead of the usual balancing.
- Teaching – David Silver. Teaching – David Silver. David Silver — [Home Page](https://www.davidsilver.uk/teaching/)
- RL Course by David Silver - Lecture 1: Introduction to Reinforcement Learning, by Google DeepMind. — [1](https://www.youtube.com/watch?v=2pWv7GOvuf0)
- RL Course by David Silver - Lecture 2: Markov Decision Process, by Google DeepMind. [2](https://www.youtube.com/watch?v=lfHX2hHRMVQ)
- RL Course by David Silver - Lecture 3: Planning by Dynamic Programming, by Google DeepMind. [3](https://www.youtube.com/watch?v=Nd1-UUMVfz4)
- RL Course by David Silver - Lecture 4: Model-Free Prediction, by Google DeepMind. [4](https://www.youtube.com/watch?v=PnHCvfgC_ZA)
- RL Course by David Silver - Lecture 5: Model Free Control, by Google DeepMind. [5](https://www.youtube.com/watch?v=0g4j2k_Ggc4)
- RL Course by David Silver - Lecture 6: Value Function Approximation, by Google DeepMind. [6](https://www.youtube.com/watch?v=UoPei5o4fps)
- RL Course by David Silver - Lecture 7: Policy Gradient Methods, by Google DeepMind. [7](https://www.youtube.com/watch?v=KHZVXao4qXs)
- RL Course by David Silver - Lecture 8: Integrating Learning and Planning, by Google DeepMind. [8](https://www.youtube.com/watch?v=ItMutbeOHtc)
- RL Course by David Silver - Lecture 9: Exploration and Exploitation, by Google DeepMind. [9](https://www.youtube.com/watch?v=sGuiWX07sKw)
- RL Course by David Silver - Lecture 10: Classic Games, by Google DeepMind. [10](https://www.youtube.com/watch?v=kZ_AUmFcZtk)

<figure><img src="../.gitbook/assets/image (3).png" alt=""><figcaption><p>David Silver teaching materials.</p></figcaption></figure>

6. [Sequential Decision Analytics and Modeling](https://castle.princeton.edu/sdamodeling/)

### Q-LEARN

This section is the Q-learning sketch: Markov chain, exploration then exploitation, optimal policy.

- Markov chain problem, (state, action, new state, reward)
- Lots of Exploration in the beginning, then exploitation
- Returns optimal policy.
- What is Q, by Udacity. Refer to youtube [here](https://www.youtube.com/watch?v=9m_6q_KECTk)

### Deep Learning

This section points at deep RL reviews, deep Q-learning, and PyTorch tutorials.

1. [A review paper about RL in DL](https://arxiv.org/pdf/1701.07274.pdf)
- An Introduction To Deep Reinforcement Learning, by Ankit Choudhary. [deep Q-learning](https://www.analyticsvidhya.com/blog/2019/04/introduction-deep-q-learning-python/)
3. Pytorch
 - Reinforcement Learning (DQN) Tutorial — PyTorch Tutorials 2.14.0+cu130 documentation. [DQN](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
 - Reinforcement Learning (PPO) with TorchRL Tutorial — PyTorch Tutorials 2.14.0+cu130 documentation. [PPO](https://pytorch.org/tutorials/intermediate/reinforcement_ppo.html)
 - Train a Mario-playing RL Agent — PyTorch Tutorials 2.14.0+cu130 documentation. [Mario example](https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html)

## RLHF

This section links Hugging Face’s illustrated RLHF post after the RL introduction above.

The same notes are in [Reinforcement Learning for LLM](../generative-ai/large-language-models-llms.md#reinforcement-learning-for-llm).

- We’re on a journey to advance and democratize artificial intelligence through open source and open science. [illustrated RLHF by Huggingface](https://huggingface.co/blog/rlhf)
