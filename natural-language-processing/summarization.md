# Summarization

![Jatana](https://lh4.googleusercontent.com/eoFe8uZJHAZ8cil1x7TZ-rENzkfkQE3wVr5fHGbeS17h2GlsSMJcFzZ4plUDHd7TN1gsZ6OKKp-WelNVaHmFhOVXxPltjxSN\_USk3s5Ro\_L1Ct-yLiST1q7ST5k5W80CkyHZj7eM)

1. [**Email summarization but with a great intro (see image above)**](https://medium.com/jatana/unsupervised-text-summarization-using-sentence-embeddings-adb15ce83db1)
2. [**With nltk**](https://stackabuse.com/text-summarization-with-nltk-in-python/) **- words assigned weighted frequency, summed up in sentences and then selected based on the top K scored sentences.**
3. [**Awesome-text-summarization on github**](https://github.com/mathsyouth/awesome-text-summarization#abstractive-text-summarization)
4. [**Methodical review of abstractive summarization**](https://medium.com/@madrugado/interesting-stuff-at-emnlp-part-ii-ce92ac928f16)
5. [**Medium on extractive and abstractive - overview with the abstractive code** ](https://towardsdatascience.com/data-scientists-guide-to-summarization-fc0db952e363)
6. [**NAMAS**](https://arxiv.org/abs/1509.00685) **-** [**Neural attention model for abstractive summarization**](https://github.com/facebookarchive/NAMAS)**, -**[**Neural Attention Model for Abstractive Sentence Summarization**](https://www.aclweb.org/anthology/D/D15/D15-1044.pdf) **- summarizes single sentences quite well,** [**github**](https://github.com/facebookarchive/NAMAS)
7. [**Abstractive vs extractive, blue intro**](https://www.salesforce.com/products/einstein/ai-research/tl-dr-reinforced-model-abstractive-summarization/)
8. [**Intro to text summarization**](https://towardsdatascience.com/a-quick-introduction-to-text-summarization-in-machine-learning-3d27ccf18a9f)
9. [**Paper: survey on text summ**](https://arxiv.org/pdf/1707.02268.pdf)**,** [**arxiv**](https://arxiv.org/abs/1707.02268)
10. [**Very short intro**](https://medium.com/@stephenhky/summarizing-text-summarization-5d83ff2863a2)
11. [**Intro on encoder decoder**](https://medium.com/@social\_20188/text-summarization-cfdbbd6fb800)
12. [**Unsupervised methods using sentence emebeddings (long and good)**](https://medium.com/jatana/unsupervised-text-summarization-using-sentence-embeddings-adb15ce83db1) **- using sent2vec, clustering, picking by rank**&#x20;
13. [**Abstractive summarization using bert for sota**](https://towardsdatascience.com/summarization-has-gotten-commoditized-thanks-to-bert-9bb73f2d6922)
14. **Abstractive**
    1. [**Git1: uses pytorch 0.7, fails to work no matter what i did**](https://github.com/alesee/abstractive-text-summarization)
    2. [**Git2, keras code for headlines, missing dataset**](https://github.com/udibr/headlines)
    3. [**Encoder decoder in keras using rnn, claims cherry picked results, the majority is prob not as good**](https://hackernoon.com/text-summarization-using-keras-models-366b002408d9)
    4. [**A lot of Text summarization algos on git, using seq2seq, using many methods, glove, etc -** ](https://github.com/chen0040/keras-text-summarization)
    5. [**Summarization with point generator networks**](https://github.com/becxer/pointer-generator/) **on git**
    6. [**Summarization based on gigaword claims SOTA**](https://github.com/tensorflow/models/tree/master/research/textsum)
    7. [**Facebooks neural attention network**](https://github.com/facebookarchive/NAMAS) **NAMAS on git**
    8. [**Medium on summarization with tensor flow on news articles from cnn**](https://hackernoon.com/how-to-run-text-summarization-with-tensorflow-d4472587602d)
15. **Keywords extraction**
    1. [**The best text rank presentation**](http://ai.fon.bg.ac.rs/wp-content/uploads/2017/01/Topic\_modeling\_and\_graph-based\_keywords\_extraction\_2017.pdf)
    2. [**Text rank by gensim on medium**](https://medium.com/@shivangisareen/text-summarisation-with-gensim-textrank-46bbb3401289)
    3. [**Text rank 2**](http://ai.intelligentonlinetools.com/ml/text-summarization/)
    4. [**Text rank - custom code, extractive vs abstractive, how to use, some more theoretical info and page rank intuition.**](https://nlpforhackers.io/textrank-text-summarization/)
    5. [**Text rank paper**](https://web.eecs.umich.edu/\~mihalcea/papers/mihalcea.emnlp04.pdf)
    6. [**Improving textrank using adjectival and noun compound modifiers**](https://graphaware.com/neo4j/2017/10/03/efficient-unsupervised-topic-extraction-nlp-neo4j.html)
    7. **Unread -** [**New similarity function paper for textrank**](https://arxiv.org/pdf/1602.03606.pdf)
16. **Extractive summarization**
    1. [**Text rank with glove vectors instead of tf-idf as in the paper**](https://medium.com/analytics-vidhya/an-introduction-to-text-summarization-using-the-textrank-algorithm-with-python-implementation-2370c39d0c60) **(sam)**
    2. [**Medium with code on extractive using word occurrence similarity + cosine, pick top based on rank**](https://towardsdatascience.com/understand-text-summarization-and-create-your-own-summarizer-in-python-b26a9f09fc70)
    3. [**Medium on methods, freq, LSA, linking words, sentences,bayesian, graph ranking, hmm, crf,** ](https://medium.com/sciforce/towards-automatic-text-summarization-extractive-methods-e8439cd54715)
    4. [**Wiki on automatic summarization, abstractive vs extractive,** ](https://en.wikipedia.org/wiki/Automatic\_summarization#TextRank\_and\_LexRank)
    5. [**Pyteaser, textteaset, lexrank, pytextrank summarization models &**\
       **rouge-1/n and blue metrics to determine quality of summarization models**\
       ](https://rare-technologies.com/text-summarization-in-python-extractive-vs-abstractive-techniques-revisited/)**Bottom line is that textrank is competitive to sumy\_lex**
    6. [**Sumy**](https://github.com/miso-belica/sumy)
    7. [**Pyteaser**](https://github.com/xiaoxu193/PyTeaser)
    8. [**Pytextrank**](https://github.com/ceteri/pytextrank)
    9. [**Lexrank**](https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume22/erkan04a-html/erkan04a.html)
    10. [**Gensim tutorial on textrank**](https://www.machinelearningplus.com/nlp/gensim-tutorial/)
    11. [**Email summarization**](https://github.com/jatana-research/email-summarization)
