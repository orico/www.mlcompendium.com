# N-Shot Learning

This page covers zero-, one-, and few-shot learning and how large language models use prompts as shots.

The same notes are in [SIAMESE NETWORKS](../deep-learning/deep-neural-nets.md#siamese-networks) and [SIAMESE NETWORKS (one shot)](../deep-learning/deep-learning-models.md#siamese-networks-one-shot).

### N-SHOT LEARNING

N-shot learning means training or adapting with only a handful of labeled examples per class.

1. [Zero shot, one shot, few shot](https://floydhub.github.io/n-shot-learning/) (siamese is one shot)

### ZERO SHOT LEARNING

Zero-shot learning classifies new classes using side information instead of labeled examples for those classes.

1. [Instead of using class labels](https://www.youtube.com/watch?v=jBnCcr-3bXc), we use some kind of vector representation for the classes, taken from a co-occurrence-after-svd or word2vec. — quite clever. This enables us to figure out if a new unseen class is near one of the known supervised classes. KNN can be used or some other distance-based classifier. Can we use word2vec for similarity measurements of new classes?

The same notes are in [WORD2VEC](../deep-learning/embedding.md#word2vec).

<figure><img src="../.gitbook/assets/gimg-21a7ab06ebb7.png" alt=""><figcaption><p>Image by Dr. Timothy Hospedales, Yandex</p><p>Credit: <a href="https://www.youtube.com/watch?v=jBnCcr-3bXc">Dr. Timothy Hospedales, Yandex</a>.</p></figcaption></figure>

For classification, we can use nearest neighbour or manifold-based labeling propagation.

<figure><img src="../.gitbook/assets/gimg-82e21102e9eb.png" alt=""><figcaption><p>Image by Dr. Timothy Hospedales, Yandex</p><p>Credit: <a href="https://www.youtube.com/watch?v=jBnCcr-3bXc">Dr. Timothy Hospedales, Yandex</a>.</p></figcaption></figure>

Multiple category vectors? Multilabel zero-shot also in the video

2. [with siamese networks](https://medium.com/data-science/zero-shot-intent-classification-with-siamese-networks-35900471c7fd)

#### GPT3 is ZERO, ONE, FEW

GPT-3 style prompting treats zero-, one-, and few-shot use as how many examples you put in the prompt.

The same notes are in [GPT3](../deep-learning/attention.md#gpt3), [Large Language Models (LLMs)](../generative-ai/large-language-models-llms.md), and [Prompt](../generative-ai/prompt.md).

- [Prompt Engineering Tips & Tricks](https://blog.andrewcantino.com/blog/2021/04/21/prompt-engineering-tips-and-tricks/)
- [Open GPT3 prompt engineering](https://medium.com/swlh/openai-gpt-3-and-prompt-engineering-dcdc2c5fcd29)

## Deprecated links

{% hint style="warning" %}
These links and images no longer work. The original wording is kept here. A same-resource copy, when one was checked, is used above.
{% endhint %}

- Zero shot, one shot, few shot (siamese is one shot). This address no longer opens: https://blog.floydhub.com/n-shot-learning/
- with siamese networks. This address no longer opens: https://towardsdatascience.com/zero-shot-intent-classification-with-siamese-networks-35900471c7fd
