---
description: Or why we shouldn't trust models
---

# Datasets Reliability & Correctness

A high score only means something if the dataset actually requires the reasoning you think it does, so this page is about not over-trusting models or benchmarks. It starts with shortcut cues and the dataset ablations that expose them, then a debiasing example from fact verification, then behavioral testing with CheckList.

The warning sign is the [Clever Hans effect](https://thegradient.pub/nlps-clever-hans-moment-has-arrived), in relations to cues left in the dataset that models find, instead of actually solving the defined task! The defence is borrowed from model work. Ablating, i.e. removing, part of a model and observing the impact this has on performance is a common method for verifying that the part in question is useful. If performance doesn't go down, then the part is useless and should be removed. Carrying this method over to datasets, it should become common practice to perform dataset ablations, as well, for example:

- Provide only incomplete input (as done in the reviewed paper): This verifies that the complete input is required. If not, the dataset contains cues that allow taking shortcuts.
- Shuffle the input: This verifies the importance of word (or sentence) order. If a bag-of-words/sentences gives similar results, even though the task requires sequential reasoning, then the model has not learned sequential reasoning and the dataset contains cues that allow the model to "solve" the task without it.
- Assign random labels: How much does performance drop if ten percent of instances are relabeled randomly? How much with all random labels? If scores don't change much, the model probably didn't learning anything interesting about the task.
- Randomly replace content words: How much does performance drop if all noun phrases and/or verb phrases are replaced with random noun phrases and verbs? If not much, the dataset may provide unintended non-content cues, such as sentence length or distribution of function words.

The lesson is that datasets need more love: datasets ablation and public beta before a benchmark is trusted, and inter-prediction agreement as a check on what the models are actually keying on.

Fact verification is a worked case of such a cue. Towards Debiasing Fact Verification Models ([Paper](https://arxiv.org/abs/1908.05267)) starts from the claim that fact verification requires validating a claim in the context of evidence, and shows that in the popular FEVER dataset this might not necessarily be the case: claim-only classifiers perform competitively with top evidence-aware models. The paper investigates the cause and identifies strong cues for predicting labels solely based on the claim, without considering any evidence.

Ablation removes pieces; behavioral testing and CHECKLIST go the other way and probe what the model can do. The [Blog](https://amitness.com/2020/07/checklist/) is Amit Chaudhary's overview of the “CheckList” framework for fine-grained evaluation of NLP models. The [Youtube](https://www.youtube.com/watch?v=L3gaWctPg6E) video is Connor Shorten's CheckList Explained! (ACL 2020 Best Paper). The [paper](https://arxiv.org/pdf/2005.04118.pdf) itself is Beyond Accuracy: Behavioral Testing of NLP models with CheckList, and the [git](https://github.com/marcotcr/checklist) repo, marcotcr/checklist, is its code.

The same paper, ע Beyond Accuracy: Behavioral Testing of NLP models with CheckList, was discussed in Hebrew with its abstract page at [https://arxiv.org/abs/2005.04118](https://arxiv.org/abs/2005.04118):

מאמר מעניין ומאוד יישומי שזכה בbest paper award בACL 2020. [Yonatan hadar on the subject in hebrew](https://www.facebook.com/groups/MDLI1/permalink/1627671704063538/) is that post in the Machine & Deep learning Israel group.
