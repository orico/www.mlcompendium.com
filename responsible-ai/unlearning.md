# Unlearning

Machine unlearning (MU) refers to the challenge of erasing a data point's influence on the input-output mapping of an ML model.
The sections below collect papers, medium posts, GitHub projects, and community notes on that unlearning problem.

The same notes are in [Large Language Models (LLMs)](../generative-ai/large-language-models-llms.md).

## Papers

These entries are surveys, methods, and notable unlearning papers.

- A Survey of Machine Unlearning. (really good) A [survey](https://arxiv.org/abs/2209.02299)
- of [MU](https://arxiv.org/pdf/2209.02299.pdf)
- Existing Literature about Machine Unlearning. [Existing literature on MU](https://github.com/jjbrophy47/machine_unlearning)
- Explore and run AI code with Kaggle Notebooks | Using data from 2023 Kaggle AI Report. [Machine Unlearning: The Right to be Forgotten](https://www.kaggle.com/code/tamlhp/machine-unlearning-the-right-to-be-forgotten#sec:algorithms)
4. (really good) [Awesome MU on Github](https://github.com/tamlhp/awesome-machine-unlearning?tab=readme-ov-file#type-image) ([website](https://awesome-machine-unlearning.github.io/))- a collection of academic articles, published methodology, and datasets on the subject of machine unlearning. model agnostic, intrinsic, and data-driven approaches, evaluation metrics, and datasets.

<figure><img src="../.gitbook/assets/image (1) (1).png" alt=""><figcaption><p>Awesome machine unlearning.</p></figcaption></figure>

- Who's Harry Potter? Approximate Unlearning in LLMs. [Arxiv](https://arxiv.org/abs/2310.02238)
- The page covers microsoft. The page covers microsoft. paper [Microsoft](https://www.microsoft.com/en-us/research/project/physics-of-agi/articles/whos-harry-potter-making-llms-forget-2/)
- [medium](https://pub.towardsai.net/who-is-harry-potter-inside-microsoft-researchs-fine-tuning-method-for-unlearning-concepts-in-llms-33dfe8e742a9)
6. [fast yet effective MU](https://arxiv.org/pdf/2111.08947.pdf)
7. [one shot MU](https://arxiv.org/pdf/2201.05629.pdf)
- Client Challenge. Client Challenge. [A review on MU](https://link.springer.com/article/10.1007/s42979-023-01767-4)
- Client Challenge. Client Challenge. [Machine Un-learning: An Overview of Techniques, Applications, and Future Directions](https://link.springer.com/article/10.1007/s12559-023-10219-3)
- The page covers machine Unlearning. The page covers machine Unlearning. [Machine Unlearning](https://arxiv.org/abs/1912.03817)

## Medium

These Medium posts give overviews and practical angles on machine unlearning.

1. [A fresh perspective on machine unlearning, with a real-world solution!](https://medium.com/@aliborji/a-fresh-perspective-on-machine-unlearning-with-a-real-world-solution-203821dd01c0) a solution that uses three of the following approaches. Data Augmentation, Weight Decay, Fine-Tuning, Selective Retraining, and Neural Architecture Modifications.
- What is MU? Part [1](https://medium.com/@choquette.christopher/what-is-machine-unlearning-pt-1-933ff53dc9a6)
- [2](https://medium.com/@choquette.christopher/how-to-do-machine-unlearning-pt-2-ae32cb6ca2f1)

## GitHub

These repositories implement or catalog machine unlearning and data deletion.

- Build software better, together. [search results](https://github.com/search?q=unlearning&type=repositories)
2. This [repository](https://github.com/cleverhans-lab/machine-unlearning) contains the core code used in the SISA experiments of our [Machine Unlearning](https://arxiv.org/abs/1912.03817) paper along with some example scripts.
3. Implementations of various data deletion methods.

[Evaluation Doc.](https://docs.google.com/document/d/14B_aLihLTNE7a2yRQHNRRVwvSOkttakVYFhlayZBNkE/edit)

4. This [repository](https://github.com/shash42/Evaluating-Inexact-Unlearning/tree/master) contains the code used in our experiments of our paper on [Evaluating Machine Unlearning](https://arxiv.org/abs/2201.06640) in the src/ folder along with some sample scripts in the scripts/ folder.
5. #### [This](https://github.com/meghdadk/SCRUB) is a Python implementation of "Towards Unbounded Machine Unlearning"

 This list item points at the SCRUB repo for unbounded machine unlearning.

6. #### data deletion

 This list item links again to the Chris Waites data-deletion repository.

## Community

Community competitions and leaderboards on machine unlearning.

1. NeurIPS 2023 [Kaggle](https://www.kaggle.com/competitions/neurips-2023-machine-unlearning/leaderboard) - Machine Unlearning Erase the influence of requested samples without hurting accuracy

## Deprecated links

{% hint style="warning" %}
These links and images no longer work. The original wording is kept here. A same-resource copy, when one was checked, is used above.
{% endhint %}


- paper. This address no longer opens: https://browse.arxiv.org/pdf/2310.02238
- Implementations of various data deletion methods.. This address no longer opens: https://github.com/ChrisWaites/data-deletion?tab=readme-ov-file
