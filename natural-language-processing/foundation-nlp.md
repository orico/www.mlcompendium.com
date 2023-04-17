# Foundation NLP

## **Basic nlp**

1. [**Benchmarking tokenizers for optimalprocessing speed**](https://towardsdatascience.com/benchmarking-python-nlp-tokenizers-3ac4735100c5)
2. [**Using nltk with gensim** ](https://www.scss.tcd.ie/\~munnellg/projects/visualizing-text.html)
3. [**Multiclass text classification with svm/nb/mean w2v/**](https://towardsdatascience.com/multi-class-text-classification-model-comparison-and-selection-5eb066197568)**d2v - tutorial with code and notebook.**
4. [**Basic pipeline for keyword extraction**](https://medium.com/analytics-vidhya/automated-keyword-extraction-from-articles-using-nlp-bfd864f41b34)
5. [**DL for text classification**](https://ahmedbesbes.com/overview-and-benchmark-of-traditional-and-deep-learning-models-in-text-classification.html)
   1. **Logistic regression with word ngrams**
   2. **Logistic regression with character ngrams**
   3. **Logistic regression with word and character ngrams**
   4. **Recurrent neural network (bidirectional GRU) without pre-trained embeddings**
   5. **Recurrent neural network (bidirectional GRU) with GloVe pre-trained embeddings**
   6. **Multi channel Convolutional Neural Network**
   7. **RNN (Bidirectional GRU) + CNN model**
6. **LexNLP -** [**glorified regex extractor**](https://towardsdatascience.com/lexnlp-library-for-automated-text-extraction-ner-with-bafd0014a3f8)

## **Chunking**

1. [**Coding Chunkers as Taggers: IO, BIO, BMEWO, and BMEWO+**](https://lingpipe-blog.com/2009/10/14/coding-chunkers-as-taggers-io-bio-bmewo-and-bmewo/)

## **NLP for hackers tutorials**

1. [**How to convert between verb/noun/adjective/adverb forms using Wordnet**](https://nlpforhackers.io/convert-words-between-forms/)
2. [**Complete guide for training your own Part-Of-Speech Tagger -**](https://nlpforhackers.io/training-pos-tagger/) **using** [**Penn Treebank tagset**](https://www.ling.upenn.edu/courses/Fall\_2003/ling001/penn\_treebank\_pos.html)**. Using nltk or stanford pos taggers, creating features from actual words (manual stemming, etc0 using the tags as labels, on a random forest, thus creating a classifier for POS on our own. Not entirely sure why we need to create a classifier from a “classifier”.**
3. [**Word net introduction**](https://nlpforhackers.io/starting-wordnet/) **- POS, lemmatize, synon, antonym, hypernym, hyponym**
4. [**Sentence similarity using wordnet**](https://nlpforhackers.io/wordnet-sentence-similarity/) **- using synonyms cumsum for comparison. Today replaced with w2v mean sentence similarity.**
5. [**Stemmers vs lemmatizers**](https://nlpforhackers.io/stemmers-vs-lemmatizers/) **- stemmers are faster, lemmatizers are POS / dictionary based, slower, converting to base form.**
6. [**Chunking**](https://nlpforhackers.io/text-chunking/) **- shallow parsing, compared to deep, similar to NER**
7. [**NER -**](https://nlpforhackers.io/named-entity-extraction/) **using nltk chunking as a labeller for a classifier, training one of our own. Using IOB features as well as others to create a new ner classifier which should be better than the original by using additional features. Aso uses a new english dataset GMB.**
8. [**Building nlp pipelines, functions coroutines etc..**](https://nlpforhackers.io/building-a-nlp-pipeline-in-nltk/)
9. [**Training ner using generators**](https://nlpforhackers.io/training-ner-large-dataset/)
10. [**Metrics, tp/fp/recall/precision/micro/weighted/macro f1**](https://nlpforhackers.io/classification-performance-metrics/)
11. [**Tf-idf**](https://nlpforhackers.io/tf-idf/)
12. [**Nltk for beginners**](https://nlpforhackers.io/introduction-nltk/)
13. [**Nlp corpora**](https://nlpforhackers.io/corpora/) **corpuses**
14. [**bow/bigrams**](https://nlpforhackers.io/language-models/)
15. [**Textrank**](https://nlpforhackers.io/textrank-text-summarization/)
16. [**Word cloud**](https://nlpforhackers.io/word-clouds/)
17. [**Topic modelling using gensim, lsa, lsi, lda,hdp**](https://nlpforhackers.io/topic-modeling/)
18. [**Spacy full tutorial**](https://nlpforhackers.io/complete-guide-to-spacy/)
19. [**POS using CRF**](https://nlpforhackers.io/crf-pos-tagger/)

## **Synonyms**&#x20;

1. **Python Module to get Meanings, Synonyms and what not for a given word using vocabulary (also a comparison against word net)** [**https://vocabulary.readthedocs.io/en/…**](https://vocabulary.readthedocs.io/en/latest/)

**For a given word, using Vocabulary, you can get its**

* **Meaning**
* **Synonyms**
* **Antonyms**
* **Part of speech : whether the word is a noun, interjection or an adverb et el**
* **Translate : Translate a phrase from a source language to the desired language.**
* **Usage example : a quick example on how to use the word in a sentence**
* **Pronunciation**
* **Hyphenation : shows the particular stress points(if any)**

### **Swiss army knife libraries**

1. [**textacy**](https://chartbeat-labs.github.io/textacy/) **is a Python library for performing a variety of natural language processing (NLP) tasks, built on the high-performance spacy library. With the fundamentals — tokenization, part-of-speech tagging, dependency parsing, etc. — delegated to another library, textacy focuses on the tasks that come before and follow after.**

## **Collocation**&#x20;

1. **What is collocation? - “the habitual juxtaposition of a particular word with another word or words with a frequency greater than chance.”Medium** [**tutorial**](https://medium.com/@nicharuch/collocations-identifying-phrases-that-act-like-individual-words-in-nlp-f58a93a2f84a)**, quite good, comparing freq/t-test/pmi/chi2 with github code**
2. **A website dedicated to** [**collocations**](http://www.collocations.de/)**, methods, references, metrics.**
3. [**Text analysis for sentiment, doing feature selection**](https://streamhacker.com/tag/chi-square/) **a tutorial with chi2(IG?),** [**part 2 with bi-gram collocation in ntlk**](https://streamhacker.com/2010/05/24/text-classification-sentiment-analysis-stopwords-collocations/)
4. [**Text2vec**](http://text2vec.org/collocations.html) **in R - has ideas on how to use collocations, for downstream tasks, LDA, W2V, etc. also explains about PMI and other metrics, note that gensim metric is unsupervised and probablistic.**
5. **NLTK on** [**collocations**](http://www.nltk.org/howto/collocations.html)
6. **A** [**blog post**](https://graus.nu/tag/gensim/) **about keeping or removing stopwords for collocation, usefull but no firm conclusion. Imo we should remove it before**
7. **A** [**blog post**](http://n-chandra.blogspot.com/2014/06/collocation-extraction-using-nltk.html) **with code of using nltk-based collocation**
8. **Small code for using nltk** [**collocation**](http://compling.hss.ntu.edu.sg/courses/hg2051/week09.html)
9. **Another code / score example for nltk** [**collocation**](https://stackoverflow.com/questions/8683588/understanding-nltk-collocation-scoring-for-bigrams-and-trigrams)
10. **Jupyter notebook on** [**manually finding collocation**](https://github.com/sgsinclair/alta/blob/a482d343142cba12030fea4be8f96fb77579b3ab/ipynb/utilities/Collocates.ipynb) **- not useful**
11. **Paper:** [**Ngram2Vec**](http://www.aclweb.org/anthology/D17-1023) **-** [**Github**](https://github.com/zhezhaoa/ngram2vec) **We introduce ngrams into four representation methods. The experimental results demonstrate ngrams’ effectiveness for learning improved word representations. In addition, we find that the trained ngram embeddings are able to reflect their semantic meanings and syntactic patterns. To alleviate the costs brought by ngrams, we propose a novel way of building co-occurrence matrix, enabling the ngram-based models to run on cheap hardware**
12. **Youtube on** [**bigrams**](https://www.youtube.com/watch?v=3i5QEmaOtkU\&list=PLjTSKEJpqIeANubEWBo-z5TO89m7VtfG\_)**,** [**collocation**](https://www.youtube.com/watch?v=QvrbsjwErMA)**, mutual info and** [**collocation**](http://www.let.rug.nl/nerbonne/teach/rema-stats-meth-seminar/presentations/Suster-2011-MI-Coll.pdf)

## **Language detection**

1. [**Using google lang detect**](https://github.com/Mimino666/langdetect) **- 55 languages af, ar, bg, bn, ca, cs, cy, da, de, el, en, es, et, fa, fi, fr, gu, he,**\
   **hi, hr, hu, id, it, ja, kn, ko, lt, lv, mk, ml, mr, ne, nl, no, pa, pl,**\
   **pt, ro, ru, sk, sl, so, sq, sv, sw, ta, te, th, tl, tr, uk, ur, vi, zh-cn, zh-tw**

## **Stemming**

**How to measure a stemmer?**

1. **References \[**[**1**](https://files.eric.ed.gov/fulltext/EJ1020841.pdf) [**2**](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.68.2870\&rep=rep1\&type=pdf)**(apr11)** [**3**](http://www.informationr.net/ir/19-1/paper605.html)**(Index compression factor ICF)** [**4**](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.16.8310\&rep=rep1\&type=pdf) [**5**](https://pdfs.semanticscholar.org/1c0c/0fa35d4ff8a2f925eb955e48d655494bd167.pdf)**]**

## **Phrase modelling**

1. [**Phrase Modeling**](https://github.com/explosion/spacy-notebooks/blob/master/notebooks/conference\_notebooks/modern\_nlp\_in\_python.ipynb) **- using gensim and spacy**

**Phrase modeling is another approach to learning combinations of tokens that together represent meaningful multi-word concepts. We can develop phrase models by looping over the the words in our reviews and looking for words that co-occur (i.e., appear one after another) together much more frequently than you would expect them to by random chance. The formula our phrase models will use to determine whether two tokens AA and BB constitute a phrase is:**

**count(A B)−countmincount(A)∗count(B)∗N>threshold**

1. [ **SO on PE.**](https://www.quora.com/Whats-the-best-way-to-extract-phrases-from-a-corpus-of-text-using-Python)
2.

## **Document classification**

1. [**Using hierarchical attention network**](https://www.cs.cmu.edu/\~hovy/papers/16HLT-hierarchical-attention-networks.pdf)

## **Hebrew NLP tools**

1. [**HebMorph**](https://github.com/synhershko/HebMorph.CorpusSearcher) **last update 7y ago**
2. [**Hebmorph elastic search**](https://github.com/synhershko/elasticsearch-analysis-hebrew/wiki/Getting-Started) [**Hebmorph blog post**](https://code972.com/blog/2013/12/673-hebrew-search-done-right)**, and other** [**blog posts**](https://code972.com/hebmorph)**,** [**youtube**](https://www.youtube.com/watch?v=v8w32wC6ppI)
3. [**Awesome hebrew nlp git**](https://github.com/iddoberger/awesome-hebrew-nlp)**,** [**git**](https://github.com/synhershko/HebMorph/blob/master/dotNet/HebMorph/HSpell/Constants.cs)
4. [**Hebrew-nlp service**](https://hebrew-nlp.co.il/) [**docs**](https://docs.hebrew-nlp.co.il/#/README) [**the features**](https://hebrew-nlp.co.il/features) **(morphological analysis, normalization etc),** [**git**](https://github.com/HebrewNLP)
5. [**Apache solr stop words (dead)**](https://wiki.apache.org/solr/LanguageAnalysis#Hebrew)
6. [**SO on hebrew analyzer/stemming**](https://stackoverflow.com/questions/1063856/lucene-hebrew-analyzer)**,** [**here too**](https://stackoverflow.com/questions/20953495/is-there-a-good-stemmer-for-hebrew)
7. [**Neural sentiment benchmark using two algorithms, for character and word level lstm/gru**](https://github.com/omilab/Neural-Sentiment-Analyzer-for-Modern-Hebrew) **-** [**the paper**](http://aclweb.org/anthology/C18-1190)
8. [**Hebrew word embeddings**](https://github.com/liorshk/wordembedding-hebrew)
9. [**Paper for rich morphological datasets for comparison - rivlin**](https://aclweb.org/anthology/C18-1190)

## **Semantic roles:**

1. [**http://language.worldofcomputing.net/semantics/semantic-roles.html**](http://language.worldofcomputing.net/semantics/semantic-roles.html)
