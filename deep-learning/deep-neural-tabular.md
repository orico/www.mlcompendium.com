# Deep Neural Tabular

Tables are where tree ensembles usually win, so the question here is when a deep network is worth it on tabular data. The page starts with TabNet, one architecture built for tables, and then turns to the surveys that weigh deep learning against the usual tabular models.

The same notes are in [ENTITY EMBEDDINGS](representations.md#entity-embeddings).

Tabnet is the first stop. Its [papers with code](https://paperswithcode.com/paper/tabnet-attentive-interpretable-tabular/review/) page now lands on Hugging Face's Trending Papers, and the [paper](https://arxiv.org/abs/1908.07442) itself is "TabNet: Attentive Interpretable Tabular Learning". The [pytorch](https://github.com/dreamquark-ai/tabnet) code is the PyTorch implementation of TabNet paper : https://arxiv.org/pdf/1908.07442.pdf - dreamquark-ai/tabnet. A medium walkthrough used to sit here and is kept at the end of the page.

One architecture does not settle the question, so the surveys come next. [Survey on DNN and tabular data](https://arxiv.org/abs/2110.01889) is "Deep Neural Networks and Tabular Data: A Survey". The counterweight is [Tabular data: deep learning is NOT all you need](https://arxiv.org/pdf/2106.03253.pdf), which starts from the fact that tree ensemble models such as XGBoost are usually recommended for classification and regression on tabular data, and tests the deep learning models recently proposed as alternatives.

## Deprecated links

{% hint style="warning" %}
These links and images no longer work. The original wording is kept here. A same-resource copy, when one was checked, is used above.
{% endhint %}

- medium. This address no longer opens: https://towardsdatascience.com/tabnet-deep-neural-network-for-structured-tabular-data-39eb4b27a9e4
