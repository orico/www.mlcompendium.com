# Named Entity Recognition (NER)

1. [**State of the art LSTM architectures using NN**](https://blog.paralleldots.com/data-science/named-entity-recognition-milestone-models-papers-and-technologies/)
2. **Medium:** [**Ner free datasets**](https://towardsdatascience.com/deep-learning-for-ner-1-public-datasets-and-annotation-methods-8b1ad5e98caf) **and** [**bilstm implementation**](https://towardsdatascience.com/deep-learning-for-named-entity-recognition-2-implementing-the-state-of-the-art-bidirectional-lstm-4603491087f1) **using glove embeddings**
3. **Easy to implement in keras! They are based on the following** [**paper**](https://arxiv.org/abs/1511.08308)
4. [**Medium**](https://medium.com/district-data-labs/named-entity-recognition-and-classification-for-entity-extraction-6f23342aa7c5)**: NLTK entities, polyglot entities, sner entities, finally an ensemble method wins all!**

![](https://lh5.googleusercontent.com/Z\_R1r2x4UbKloRvR46EthJ-3I38Kj4TM2VfXsGzcEsQCNJ75BpS0xMbEeCtxueTHp3jbweC2ti2Y\_2dopekm\_qP4Vks4v6suZ\_buGnFlOA1I6gdUwMYWsKWOD4eV38JVCcYQ0mes)

* [**Comparison between spacy and SNER**](https://medium.com/@dudsdu/named-entity-recognition-for-unstructured-documents-c325d47c7e3a) **- for terms.**
* **\*\*\*** [**Unsupervised NER using Bert**](https://towardsdatascience.com/unsupervised-ner-using-bert-2d7af5f90b8a)
* [**Custom NER using spacy**](https://towardsdatascience.com/custom-named-entity-recognition-using-spacy-7140ebbb3718)
* [**Spacy Ner with custom data**](https://medium.com/@manivannan\_data/how-to-train-ner-with-custom-training-data-using-spacy-188e0e508c6)

![](https://lh4.googleusercontent.com/L1nTdlSIQmOBa91u5HomKen0QlT3lWaKQjNv86ar2-cTuiKzI4y3oSdQGmJacjnJ28scacsfyvBDI4\_Y15M1i-eQ02CKAe0O7zNyJOwfrv0TiiP2ExWx9wrciCxnEGMqmvHGM2kd)

* [**How to create a NER from scratch using kaggle data, using crf, and analysing crf weights using external package**](https://towardsdatascience.com/named-entity-recognition-and-classification-with-scikit-learn-f05372f07ba2)
* [**Another comparison between spacy and SNER - both are the same, for many classes.**](https://towardsdatascience.com/a-review-of-named-entity-recognition-ner-using-automatic-summarization-of-resumes-5248a75de175)

![](https://lh5.googleusercontent.com/LOc8elLlxDHhro4Isd3NZwQQtlEdIYmS\_N3N1R8N2aEESRQnOYc5TANm2GMKKZF6r0ZDqfr34W\_47ti3JU\_mTtJPwxVDpQbztP7zdkRViby8hE\_RDPfKrWHX3XgOiKJ5ODneGvj6)

* [**Vidhaya on spacy vs ner**](https://www.analyticsvidhya.com/blog/2017/04/natural-language-processing-made-easy-using-spacy-%E2%80%8Bin-python/) **- tutorial + code on how to use spacy for pos, dep, ner, compared to nltk/corenlp (sner etc). The results reflect a global score not specific to LOC for example.**

![](https://lh6.googleusercontent.com/z1n0cTOVDdW-NRozFyUhTE4RjAf6MVtnMFp-4CZ0Y\_3VYFZirMz34wSK0bj66ViejWlfno\_Bjyqvenc7KevaFGt8gIBR7RmUjP5BrCM8mkfC5g3C9MiMux7myDm5Qh\_HzsXR2tSX)

**Stanford NER (SNER)**

* [**SNER presentation - combines HMM and MaxEnt features, distributional features, NER has** ](https://nlp.stanford.edu/software/jenny-ner-2007.pdf)
* [**many applications.**](https://nlp.stanford.edu/software/jenny-ner-2007.pdf)
* [**How to train SNER, a FAQ with many other answers (read first before doing anything with SNER)**](https://nlp.stanford.edu/software/crf-faq.shtml#a)
* [**SNER demo - capital letters matter, a minimum of one.**](http://nlp.stanford.edu:8080/ner/process)&#x20;
* [**State of the art NER benchmark**](https://github.com/magizbox/underthesea/wiki/TASK-CONLL-2003)
* [**Review paper, SNER, spacy, stanford wins**](http://www.aclweb.org/anthology/W16-2703)
* [**Review paper SNER, others on biographical text, stanford wins**](https://arxiv.org/ftp/arxiv/papers/1308/1308.0661.pdf)
* [**Another NER DL paper, 90%+**](https://openreview.net/forum?id=ry018WZAZ)

**Spacy & Others**

* [**Spacy - using prodigy and spacy to train a NER classifier using active learning**](https://www.youtube.com/watch?v=l4scwf8KeIA)
* [**Ner using DL BLSTM, using glove embeddings, using CRF layer against another CRF**](http://nlp.town/blog/ner-and-the-road-to-deep-learning/)**.**
* [**Another medium paper on the BLSTM CRF with guillarue’s code**](https://medium.com/intro-to-artificial-intelligence/entity-extraction-using-deep-learning-8014acac6bb8)
* [**Guillaume blog post, detailed explanation**](https://guillaumegenthial.github.io/sequence-tagging-with-tensorflow.html)
* [**For Italian**](https://www.qcri.org/app/media/4916)
* [**Another 90+ proposed solution**](https://arxiv.org/pdf/1603.01360.pdf)
* [**A promising python implementation based on one or two of the previous papers**](https://github.com/deepmipt/ner)
* [**Quora advise, the first is cool, the second is questionable**](https://www.quora.com/How-can-I-perform-named-entity-recognition-using-deep-learning-RNN-LSTM-Word2vec-etc)
* [**Off the shelf solutions benchmark**](https://www.programmableweb.com/news/performance-comparison-10-linguistic-apis-entity-recognition/elsewhere-web/2016/11/03)
* [**Parallel api talk about bilstm and their 2mil tagged ner model (washington passes)**](https://blog.paralleldots.com/data-science/named-entity-recognition-milestone-models-papers-and-technologies/)
