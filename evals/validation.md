---
description: Or why we shouldn't trust models
---

# Datasets Reliability & Correctness

This page covers dataset ablation, shortcut cues, and behavioral testing so we do not over-trust models or benchmarks. It is about whether a dataset actually requires the reasoning you think it does.


1. [Clever Hans effect](https://thegradient.pub/nlps-clever-hans-moment-has-arrived) - in relations to cues left in the dataset that models find, instead of actually solving the defined task!
 - Ablating, i.e. removing, part of a model and observing the impact this has on performance is a common method for verifying that the part in question is useful. If performance doesn't go down, then the part is useless and should be removed. Carrying this method over to datasets, it should become common practice to perform dataset ablations, as well, for example:
 - Provide only incomplete input (as done in the reviewed paper): This verifies that the complete input is required. If not, the dataset contains cues that allow taking shortcuts.
 - Shuffle the input: This verifies the importance of word (or sentence) order. If a bag-of-words/sentences gives similar results, even though the task requires sequential reasoning, then the model has not learned sequential reasoning and the dataset contains cues that allow the model to "solve" the task without it.
 - Assign random labels: How much does performance drop if ten percent of instances are relabeled randomly? How much with all random labels? If scores don't change much, the model probably didn't learning anything interesting about the task.
 - Randomly replace content words: How much does performance drop if all noun phrases and/or verb phrases are replaced with random noun phrases and verbs? If not much, the dataset may provide unintended non-content cues, such as sentence length or distribution of function words.
 - Datasets need more love
 - Datasets ablation and public beta
 - Inter-prediction agreement
- Towards Debiasing Fact Verification Models. Towards Debiasing Fact Verification Models. [Paper](https://arxiv.org/abs/1908.05267)
3. Behavioral testing and CHECKLIST
 - An overview of the “CheckList” framework for fine-grained evaluation of NLP models, by Amit Chaudhary. [Blog](https://amitness.com/2020/07/checklist/)
 - CheckList Explained! (ACL 2020 Best Paper), by Connor Shorten. [Youtube](https://www.youtube.com/watch?v=L3gaWctPg6E)
 - [paper](https://arxiv.org/pdf/2005.04118.pdf)
 - Beyond Accuracy: Behavioral Testing of NLP models with CheckList - marcotcr/checklist. [git](https://github.com/marcotcr/checklist)
 - ע Beyond Accuracy: Behavioral Testing of NLP models with CheckList

https://arxiv.org/abs/2005.04118

מאמר מעניין ומאוד יישומי שזכה בbest paper award בACL 2020. [Yonatan hadar on the subject in hebrew](https://www.facebook.com/groups/MDLI1/permalink/1627671704063538/)
