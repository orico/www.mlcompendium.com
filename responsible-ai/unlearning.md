# Unlearning

Machine unlearning (MU) refers to the challenge of erasing a data point's influence on the input-output mapping of an ML model.
Privacy and fairness sometimes demand that a model forget, and retraining from scratch is rarely an option. The page moves from the papers that define the problem, to Medium posts that explain it, to the GitHub code that implements it, and ends with the community competition that tests it.

The same notes are in [Large Language Models (LLMs)](../generative-ai/large-language-models-llms.md).

## Papers

The field is young enough that a survey is the right first read. (really good) A [survey](https://arxiv.org/abs/2209.02299) of [MU](https://arxiv.org/pdf/2209.02299.pdf), "A Survey of Machine Unlearning", starts from the fact that computer systems hold large amounts of personal data that can threaten user privacy and weaken trust between humans and AI, and that recent regulations now require private information about a user to be removed on request. From there, [Existing literature on MU](https://github.com/jjbrophy47/machine_unlearning) collects the papers, and the Kaggle notebook [Machine Unlearning: The Right to be Forgotten](https://www.kaggle.com/code/tamlhp/machine-unlearning-the-right-to-be-forgotten#sec:algorithms), built on the 2023 Kaggle AI Report, walks the algorithms.

The broadest map is (really good) [Awesome MU on Github](https://github.com/tamlhp/awesome-machine-unlearning?tab=readme-ov-file#type-image) ([website](https://awesome-machine-unlearning.github.io/))- a collection of academic articles, published methodology, and datasets on the subject of machine unlearning. model agnostic, intrinsic, and data-driven approaches, evaluation metrics, and datasets. The figure below is its taxonomy.

<figure><img src="../.gitbook/assets/image (1) (1).png" alt=""><figcaption><p>Awesome machine unlearning.</p></figcaption></figure>

For language models, the reference case is "Who's Harry Potter? Approximate Unlearning in LLMs" on [Arxiv](https://arxiv.org/abs/2310.02238): LLMs trained on internet corpora often contain copyrighted content, and the paper proposes unlearning a subset of the training data without retraining from scratch. The paper is described by [Microsoft](https://www.microsoft.com/en-us/research/project/physics-of-agi/articles/whos-harry-potter-making-llms-forget-2/) Research, and Jesus Rodriguez's [medium](https://pub.towardsai.net/who-is-harry-potter-inside-microsoft-researchs-fine-tuning-method-for-unlearning-concepts-in-llms-33dfe8e742a9) post looks inside that fine-tuning method for unlearning concepts in LLMs.

Other methods trade exactness for speed. [fast yet effective MU](https://arxiv.org/pdf/2111.08947.pdf) is "Fast Yet Effective Machine Unlearning", and [one shot MU](https://arxiv.org/pdf/2201.05629.pdf) is "Zero-Shot Machine Unlearning". Two Springer reviews widen the view: [A review on MU](https://link.springer.com/article/10.1007/s42979-023-01767-4) and [Machine Un-learning: An Overview of Techniques, Applications, and Future Directions](https://link.springer.com/article/10.1007/s12559-023-10219-3). The exact approach is [Machine Unlearning](https://arxiv.org/abs/1912.03817), which starts from the point that once users share data it is hard to revoke, that any model trained on it may have memorized it, and introduces SISA training to make unlearning tractable.

## Medium

The papers are dense; the Medium posts explain the same ideas with practical angles. [A fresh perspective on machine unlearning, with a real-world solution!](https://medium.com/@aliborji/a-fresh-perspective-on-machine-unlearning-with-a-real-world-solution-203821dd01c0) a solution that uses three of the following approaches. Data Augmentation, Weight Decay, Fine-Tuning, Selective Retraining, and Neural Architecture Modifications.

Christopher Choquette's series asks What is MU? Part [1](https://medium.com/@choquette.christopher/what-is-machine-unlearning-pt-1-933ff53dc9a6) starts from why privacy regulation such as the ‘right to be forgotten’ matters and defines the key terms, and part [2](https://medium.com/@choquette.christopher/how-to-do-machine-unlearning-pt-2-ae32cb6ca2f1) compares the different methods of unlearning.

## GitHub

Once a method is chosen, the code is on GitHub. A plain repository search gives the [search results](https://github.com/search?q=unlearning&type=repositories) for unlearning. For the exact method, this [repository](https://github.com/cleverhans-lab/machine-unlearning) contains the core code used in the SISA experiments of our [Machine Unlearning](https://arxiv.org/abs/1912.03817) paper along with some example scripts. Implementations of various data deletion methods also used to be here; that repository is kept at the end of the page.

Measuring whether a model really forgot is its own problem, and the [Evaluation Doc.](https://docs.google.com/document/d/14B_aLihLTNE7a2yRQHNRRVwvSOkttakVYFhlayZBNkE/edit) collects related work on unlearning evaluation, from data deletion in k-means to certified and approximate data removal. This [repository](https://github.com/shash42/Evaluating-Inexact-Unlearning/tree/master) contains the code used in our experiments of our paper on [Evaluating Machine Unlearning](https://arxiv.org/abs/2201.06640) in the src/ folder along with some sample scripts in the scripts/ folder; the paper is "Towards Adversarial Evaluations for Inexact Machine Unlearning".

Two more entries close the list, the SCRUB repo for unbounded machine unlearning and the Chris Waites data-deletion repository:

5. #### [This](https://github.com/meghdadk/SCRUB) is a Python implementation of "Towards Unbounded Machine Unlearning"

6. #### data deletion

## Community

The code is tested in public. NeurIPS 2023 [Kaggle](https://www.kaggle.com/competitions/neurips-2023-machine-unlearning/leaderboard) - Machine Unlearning Erase the influence of requested samples without hurting accuracy.

## Deprecated links

{% hint style="warning" %}
These links and images no longer work. The original wording is kept here. A same-resource copy, when one was checked, is used above.
{% endhint %}


- paper. This address no longer opens: https://browse.arxiv.org/pdf/2310.02238
- Implementations of various data deletion methods.. This address no longer opens: https://github.com/ChrisWaites/data-deletion?tab=readme-ov-file
