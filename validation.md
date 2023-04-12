---
description: Or why we shouldn't trust models
---

# Datasets Reliability & Correctness

1. [Clever Hans effect](https://thegradient.pub/nlps-clever-hans-moment-has-arrived/?fbclid=IwAR3vSx9EjXcSPhXU3Jyf7aWpTqhbVDARnh3qGpSw0rysv9rLeGyZFFCPnJA) - in relations to cues left in the dataset that models find, instead of actually solving the defined task!
   1. Ablating, i.e. removing, part of a model and observing the impact this has on performance is a common method for verifying that the part in question is useful. If performance doesn't go down, then the part is useless and should be removed. Carrying this method over to datasets, it should become common practice to perform dataset ablations, as well, for example:
   2. Provide only incomplete input (as done in the reviewed paper): This verifies that the complete input is required. If not, the dataset contains cues that allow taking shortcuts.
   3. Shuffle the input: This verifies the importance of word (or sentence) order. If a bag-of-words/sentences gives similar results, even though the task requires sequential reasoning, then the model has not learned sequential reasoning and the dataset contains cues that allow the model to "solve" the task without it.
   4. Assign random labels: How much does performance drop if ten percent of instances are relabeled randomly? How much with all random labels? If scores don't change much, the model probably didn't learning anything interesting about the task.
   5. Randomly replace content words: How much does performance drop if all noun phrases and/or verb phrases are replaced with random noun phrases and verbs? If not much, the dataset may provide unintended non-content cues, such as sentence length or distribution of function words.
   6. Datasets need more love
   7. Datasets ablation and public beta
   8. Inter-prediction agreement
2. [Paper](https://arxiv.org/abs/1908.05267?fbclid=IwAR1xOHxCF3gewyijMYAfZJSysu88Y8lgRIT2OiG-jWQav4zcbPqGYSoFFkk)
3. Behavioral testing and CHECKLIST
   1. [Blog](https://amitness.com/2020/07/checklist/), [Youtube](https://www.youtube.com/watch?v=L3gaWctPg6E), [paper](https://arxiv.org/pdf/2005.04118.pdf), [git](https://github.com/marcotcr/checklist)
   2. [Yonatan hadar on the subject in hebrew](https://www.facebook.com/groups/MDLI1/permalink/1627671704063538/)
