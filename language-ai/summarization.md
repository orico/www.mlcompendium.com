# Summarization

This page collects extractive and abstractive summarization papers, code, and TextRank notes.
The linked notes cover extractive and abstractive methods, including TextRank-style approaches.

The same notes are in [NLP for hackers tutorials](foundation-nlp.md#nlp-for-hackers-tutorials).

<figure><img src="../.gitbook/assets/gimg-24182903c3d9.png" alt=""><figcaption><p>Jatana unsupervised text summarization overview.</p><p>Credit: <a href="https://lh4.googleusercontent.com/eoFe8uZJHAZ8cil1x7TZ-rENzkfkQE3wVr5fHGbeS17h2GlsSMJcFzZ4plUDHd7TN1gsZ6OKKp-WelNVaHmFhOVXxPltjxSN_USk3s5Ro_L1Ct-yLiST1q7ST5k5W80CkyHZj7eM">copied from the original hosted image</a>.</p></figcaption></figure>

- [Email summarization but with a great intro (see image above)](https://medium.com/jatana/unsupervised-text-summarization-using-sentence-embeddings-adb15ce83db1) using sent2vec clustering picking rank
2. [With nltk](https://stackabuse.com/text-summarization-with-nltk-in-python/) — words assigned weighted frequency, summed up in sentences and then selected based on the top K scored sentences.
- A curated list of resources dedicated to text summarization - mathsyouth/awesome-text-summarization. [Awesome-text-summarization on github](https://github.com/mathsyouth/awesome-text-summarization#abstractive-text-summarization)
- [Methodical review of abstractive summarization](https://medium.com/@madrugado/interesting-stuff-at-emnlp-part-ii-ce92ac928f16)
- [Medium on extractive and abstractive - overview with the abstractive code](https://medium.com/data-science/data-scientists-guide-to-summarization-fc0db952e363)
- A Neural Attention Model for Abstractive Sentence Summarization. [NAMAS](https://arxiv.org/abs/1509.00685)
- Neural Attention Model for Abstractive Summarization - facebookarchive/NAMAS. — [Neural attention model for abstractive summarization](https://github.com/facebookarchive/NAMAS)
- — [Neural Attention Model for Abstractive Sentence Summarization](https://www.aclweb.org/anthology/D/D15/D15-1044.pdf)
- Neural Attention Model for Abstractive Summarization - facebookarchive/NAMAS. — summarizes single sentences quite well [github](https://github.com/facebookarchive/NAMAS)
- Artificial Intelligence. [Abstractive vs extractive, blue intro](https://www.salesforce.com/products/einstein/ai-research/tl-dr-reinforced-model-abstractive-summarization/)
- [Intro to text summarization](https://medium.com/data-science/a-quick-introduction-to-text-summarization-in-machine-learning-3d27ccf18a9f)
- [Paper: survey on text summ](https://arxiv.org/pdf/1707.02268.pdf)
- Text Summarization Techniques: A Brief Survey. [arxiv](https://arxiv.org/abs/1707.02268)
- [Very short intro](https://medium.com/@stephenhky/summarizing-text-summarization-5d83ff2863a2)
- [Intro on encoder decoder](https://medium.com/@social_20188/text-summarization-cfdbbd6fb800)
- [Unsupervised methods using sentence emebeddings (long and good)](https://medium.com/jatana/unsupervised-text-summarization-using-sentence-embeddings-adb15ce83db1)
- [Abstractive summarization using bert for sota](https://medium.com/data-science/summarization-has-gotten-commoditized-thanks-to-bert-9bb73f2d6922)
14. Abstractive
 - PyTorch implementation/experiments on Abstractive Text Summarization using Sequence-to-sequence RNNs and Beyond paper. [Git1: uses pytorch 0.7, fails to work no matter what i did](https://github.com/alesee/abstractive-text-summarization)
 - Automatically generate headlines to short articles - udibr/headlines. [Git2, keras code for headlines, missing dataset](https://github.com/udibr/headlines)
 - <em>Learn how to summarize text in this article by Rajdeep Dua who currently leads the developer relations team at Salesforce India, and Manpreet Singh Ghotra who is currently working at Salesforce developing a machine learning platform/APIs.</em>. [Encoder decoder in keras using rnn, claims cherry picked results, the majority is probably not as good](https://hackernoon.com/text-summarization-using-keras-models-366b002408d9)
 - Text summarization using seq2seq in Keras. [A lot of Text summarization algos on git, using seq2seq, using many methods, glove, etc -](https://github.com/chen0040/keras-text-summarization)
 - Code for the ACL 2017 paper "Get To The Point: Summarization with Pointer-Generator Networks" (Python3) - becxer/pointer-generator. [Summarization with point generator networks](https://github.com/becxer/pointer-generator/)
 6. Summarization based on gigaword claims SOTA
 - Neural Attention Model for Abstractive Summarization - facebookarchive/NAMAS. [Facebooks neural attention network](https://github.com/facebookarchive/NAMAS)
 - HackerNoon - read, write and learn about any technology. [Medium on summarization with tensor flow on news articles from cnn](https://hackernoon.com/how-to-run-text-summarization-with-tensorflow-d4472587602d)
15. Keywords extraction
 1. [The best text rank presentation](http://ai.fon.bg.ac.rs/wp-content/uploads/2017/01/Topic_modeling_and_graph-based_keywords_extraction_2017.pdf)
 - [Text rank by gensim on medium](https://medium.com/@shivangisareen/text-summarisation-with-gensim-textrank-46bbb3401289)
 - Automatic Text Summarization with Python – Text Analytics Techniques. [Text rank 2](http://ai.intelligentonlinetools.com/ml/text-summarization/)
 4. [Text rank - custom code, extractive vs abstractive, how to use, some more theoretical info and page rank intuition.](https://nlpforhackers.io/textrank-text-summarization/)
 5. [Text rank paper](https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf)
 - Companies of any size have to manage and access huge amounts of data, providing advanced services for their end-users or to handle their internal processes. [Improving textrank using adjectival and noun compound modifiers](https://graphaware.com/neo4j/2017/10/03/efficient-unsupervised-topic-extraction-nlp-neo4j.html)
 7. [New similarity function paper for textrank](https://arxiv.org/pdf/1602.03606.pdf): This paper proposes a new similarity function for TextRank-style keyword and sentence ranking, and reports whether that change improves graph-based summarization quality over the usual overlap or TF-IDF edge weights.
16. Extractive summarization
 - [Text rank with glove vectors instead of tf-idf as in the paper](https://medium.com/analytics-vidhya/an-introduction-to-text-summarization-using-the-textrank-algorithm-with-python-implementation-2370c39d0c60)
 - [Medium with code on extractive using word occurrence similarity + cosine, pick top based on rank](https://medium.com/data-science/understand-text-summarization-and-create-your-own-summarizer-in-python-b26a9f09fc70)
 - [Medium on methods, freq, LSA, linking words, sentences,bayesian, graph ranking, hmm, crf,](https://medium.com/sciforce/towards-automatic-text-summarization-extractive-methods-e8439cd54715)
 - Automatic summarization. Automatic summarization - Wikipedia. [Wiki on automatic summarization, abstractive vs extractive,](https://en.wikipedia.org/wiki/Automatic_summarization#TextRank_and_LexRank)
 5. [Pyteaser, textteaset, lexrank, pytextrank summarization models & rouge-1/n and blue metrics to determine quality of summarization models](https://rare-technologies.com/text-summarization-in-python-extractive-vs-abstractive-techniques-revisited/) Bottom line is that textrank is competitive to sumy_lex

The same notes are in [Metrics](../generative-ai/large-language-models-llms.md#metrics).

 - Module for automatic summarization of text documents and HTML pages. [Sumy](https://github.com/miso-belica/sumy)
 - GitHub - xiaoxu193/PyTeaser: Summarizes news articles. [Pyteaser](https://github.com/xiaoxu193/PyTeaser)
 - Python implementation of TextRank algorithms ("textgraphs") for phrase extraction - DerwenAI/pytextrank. [Pytextrank](https://github.com/ceteri/pytextrank)
 - LexRank: Graph-based Lexical Centrality as Salience in Text Summarization. [Lexrank](https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume22/erkan04a-html/erkan04a.html)
 - Gensim is billed as a Natural Language Processing package that does 'Topic Modeling for Humans', by Selva Prabhakaran. [Gensim tutorial on textrank](https://www.machinelearningplus.com/nlp/gensim-tutorial/)
 - A module for E-mail Summarization which uses clustering of skip-thought sentence embeddings. [Email summarization](https://github.com/jatana-research/email-summarization)

## Deprecated links

{% hint style="warning" %}
These links and images no longer work. The original wording is kept here. A same-resource copy, when one was checked, is used above.
{% endhint %}

- Medium on extractive and abstractive - overview with the abstractive code This address no longer opens: https://towardsdatascience.com/data-scientists-guide-to-summarization-fc0db952e363
- Intro to text summarization This address no longer opens: https://towardsdatascience.com/a-quick-introduction-to-text-summarization-in-machine-learning-3d27ccf18a9f
- Abstractive summarization using bert for sota This address no longer opens: https://towardsdatascience.com/summarization-has-gotten-commoditized-thanks-to-bert-9bb73f2d6922
- Summarization based on gigaword claims SOTA This address no longer opens: https://github.com/tensorflow/models/tree/master/research/textsum
- Medium with code on extractive using word occurrence similarity + cosine, pick top based on rank This address no longer opens: https://towardsdatascience.com/understand-text-summarization-and-create-your-own-summarizer-in-python-b26a9f09fc70
