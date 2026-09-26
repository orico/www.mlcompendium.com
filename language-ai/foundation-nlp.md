# Foundation NLP

This page collects foundational NLP notes: chunking, collocations, stemming, Hebrew tools, and related tutorials.
The sections below gather basic pipelines, stemming, chunking, collocations, phrase models, and related tool lists.

## Basic nlp

This section lists basic NLP pipelines, classification benchmarks, and keyword extraction.

- [Benchmarking tokenizers for optimalprocessing speed](https://medium.com/data-science/benchmarking-python-nlp-tokenizers-3ac4735100c5)
2. Using nltk with gensim
- [Multiclass text classification with svm/nb/mean w2v/](https://medium.com/data-science/multi-class-text-classification-model-comparison-and-selection-5eb066197568) d2v tutorial with code and notebook
- [Basic pipeline for keyword extraction](https://medium.com/analytics-vidhya/automated-keyword-extraction-from-articles-using-nlp-bfd864f41b34)
5. DL for text classification
 1. Logistic regression with word ngrams
 2. Logistic regression with character ngrams
 3. Logistic regression with word and character ngrams
 4. Recurrent neural network (bidirectional GRU) without pre-trained embeddings
 5. Recurrent neural network (bidirectional GRU) with GloVe pre-trained embeddings
 6. Multi channel Convolutional Neural Network
 7. RNN (Bidirectional GRU) + CNN model
- LexNLP - [glorified regex extractor](https://medium.com/data-science/lexnlp-library-for-automated-text-extraction-ner-with-bafd0014a3f8)

## Stemming

This section asks how to measure a stemmer and lists reference papers.

How to measure a stemmer?

- References \ [[1](https://files.eric.ed.gov/fulltext/EJ1020841.pdf)
- 2(apr11) [3](http://www.informationr.net/ir/19-1/paper605.html)
- (Index compression factor ICF) [4](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.16.8310&rep=rep1&type=pdf)
- [5](https://pdfs.semanticscholar.org/1c0c/0fa35d4ff8a2f925eb955e48d655494bd167.pdf)

## Chunking

This section is about chunking tag schemes such as IO, BIO, and BMEWO.

- I’ve finished up the first order linear-chain CRF tagger implementation and a bunch of associated generalizations in the tagging interface. [Coding Chunkers as Taggers: IO, BIO, BMEWO, and BMEWO+](https://web.archive.org/web/20160315214621/http://lingpipe-blog.com/2009/10/14/coding-chunkers-as-taggers-io-bio-bmewo-and-bmewo/)

## Collocation

This section is about collocations: word pairs that co-occur more than chance.

The same notes are in [Information Theory](../data/information-theory.md).

1. What is collocation? - “the habitual juxtaposition of a particular word with another word or words with a frequency greater than chance.”Medium [tutorial](https://medium.com/@nicharuch/collocations-identifying-phrases-that-act-like-individual-words-in-nlp-f58a93a2f84a), quite good, comparing freq/t-test/pmi/chi2 with github code
- A website dedicated to [collocations](http://www.collocations.de/) website dedicated methods references metrics
- [Text analysis for sentiment, doing feature selection](https://streamhacker.com/tag/chi-square/)
- Text Classification for Sentiment Analysis – Stopwords and Collocations | StreamHacker. a tutorial with chi2(IG?) [part 2 with bi-gram collocation in ntlk](https://streamhacker.com/2010/05/24/text-classification-sentiment-analysis-stopwords-collocations/)
4. [Text2vec](http://text2vec.org/collocations.html) in R - has ideas on how to use collocations, for downstream tasks, LDA, W2V, etc. also explains about PMI and other metrics, note that gensim metric is unsupervised and probablistic.
- NLTK :: Sample usage for collocations. NLTK on [collocations](http://www.nltk.org/howto/collocations.html)
6. A [blog post](https://graus.nu/tag/gensim/) about keeping or removing stopwords for collocation, usefull but no firm conclusion. Imo we should remove it before
7. A [blog post](http://n-chandra.blogspot.com/2014/06/collocation-extraction-using-nltk.html) with code of using nltk-based collocation
8. Small code for using nltk collocation
- Another code / score example for nltk. [collocation](https://stackoverflow.com/questions/8683588/understanding-nltk-collocation-scoring-for-bigrams-and-trigrams)
- Jupyter notebook on [manually finding collocation](https://github.com/sgsinclair/alta/blob/a482d343142cba12030fea4be8f96fb77579b3ab/ipynb/utilities/Collocates.ipynb) Jupyter notebook not useful
11. Paper: [Ngram2Vec](http://www.aclweb.org/anthology/D17-1023) - [Github](https://github.com/zhezhaoa/ngram2vec) We introduce ngrams into four representation methods. The experimental results demonstrate ngrams’ effectiveness for learning improved word representations. In addition, we find that the trained ngram embeddings are able to reflect their semantic meanings and syntactic patterns. To alleviate the costs brought by ngrams, we propose a novel way of building co-occurrence matrix, enabling the ngram-based models to run on cheap hardware
- language modeling: trigrams, by Francisco Iacobelli. Youtube on [bigrams](https://www.youtube.com/watch?v=3i5QEmaOtkU&list=PLjTSKEJpqIeANubEWBo-z5TO89m7VtfG_)
- PMI, by Francisco Iacobelli. PMI, by Francisco Iacobelli. [collocation](https://www.youtube.com/watch?v=QvrbsjwErMA)
- mutual info and [collocation](http://www.let.rug.nl/nerbonne/teach/rema-stats-meth-seminar/presentations/Suster-2011-MI-Coll.pdf) Youtube mutual info and

## Phrase modelling

This section explains phrase modeling with gensim and the co-occurrence score.

- 💫  Jupyter notebooks for spaCy examples and tutorials - spacy-notebooks/notebooks/conference_notebooks/modern_nlp_in_python.ipynb at master · explosion/spacy-notebooks. - using gensim and spacy [Phrase Modeling](https://github.com/explosion/spacy-notebooks/blob/master/notebooks/conference_notebooks/modern_nlp_in_python.ipynb)

Phrase modeling is another approach to learning combinations of tokens that together represent meaningful multi-word concepts. We can develop phrase models by looping over the the words in our reviews and looking for words that co-occur (i.e., appear one after another) together much more frequently than you would expect them to by random chance. The formula our phrase models will use to determine whether two tokens AA and BB constitute a phrase is:

$$\frac{\mathrm{count}(A\,B)-\mathrm{count}_{\min}}{\mathrm{count}(A)\cdot\mathrm{count}(B)}\cdot N>\mathrm{threshold}$$

1. [ SO on PE.](https://www.quora.com/Whats-the-best-way-to-extract-phrases-from-a-corpus-of-text-using-Python)

## Synonyms

This section points at a vocabulary library for meanings, synonyms, and related lookup features.

The same notes are in [Augmentation](augmentation.md).

1. Python Module to get Meanings, Synonyms and what not for a given word using vocabulary (also a comparison against word net) [https://vocabulary.readthedocs.io/en/latest/](https://vocabulary.readthedocs.io/en/latest/)

For a given word, using Vocabulary, you can get its

- Meaning
- Synonyms
- Antonyms
- Part of speech : whether the word is a noun, interjection or an adverb et el
- Translate : Translate a phrase from a source language to the desired language.
- Usage example : a quick example on how to use the word in a sentence
- Pronunciation
- Hyphenation : shows the particular stress points(if any)

## Semantic roles:

This section points at a note on semantic roles.

1. http://language.worldofcomputing.net/semantics/semantic-roles.html

- Multiclass text classification with svm/nb/mean w2v/ [https://towardsdatascience.com/multi-class-text-classification-model-comparison-and-selection-5eb066197568](https://towardsdatascience.com/multi-class-text-classification-model-comparison-and-selection-5eb066197568)
- http://language.worldofcomputing.net/semantics/semantic-roles.html. [http://language.worldofcomputing.net/semantics/semantic-roles.html](http://language.worldofcomputing.net/semantics/semantic-roles.html)
- Not Acceptable! Not Acceptable! [http://language.worldofcomputing.net/semantics/semantic-roles.html](http://language.worldofcomputing.net/semantics/semantic-roles.html)

## Document classification

This section links hierarchical attention networks for document classification.

The same notes are in [Intent Recognition](intent-recognition.md).

1. [Using hierarchical attention network](https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf)

## Language detection

This section links a Google langdetect wrapper and its language list.

The same notes are in [LANGUAGE DETECTION / IDENTIFICATION](language-detection-identification-generation-nld-nli-nlg.md#language-detection--identification).

1. [Using google lang detect](https://github.com/Mimino666/langdetect) - 55 languages af, ar, bg, bn, ca, cs, cy, da, de, el, en, es, et, fa, fi, fr, gu, he,

 hi, hr, hu, id, it, ja, kn, ko, lt, lv, mk, ml, mr, ne, nl, no, pa, pl,

 pt, ro, ru, sk, sl, so, sq, sv, sw, ta, te, th, tl, tr, uk, ur, vi, zh-cn, zh-tw

## Hebrew NLP tools

This section collects Hebrew NLP tools, embeddings, and morphology resources.

- Sample app for using HebMorph, more details soon. [HebMorph](https://github.com/synhershko/HebMorph.CorpusSearcher)
- Hebrew analyzer plugin for elasticsearch. [Hebmorph elastic search](https://github.com/synhershko/elasticsearch-analysis-hebrew/wiki/Getting-Started)
- Hebrew search done right. Hebrew search done right. [Hebmorph blog post](https://code972.com/blog/2013/12/673-hebrew-search-done-right)
- HebMorph is an open-source solution (free for non-commercial use) enabling Hebrew search, via Elasticsearch, Solr and Lucene analysis plugins. and other [blog posts](https://code972.com/hebmorph)
- Open Source / HebMorph - Hebrew made searchable / Itamar Syn-Hershko, by Reversim. [youtube](https://www.youtube.com/watch?v=v8w32wC6ppI)
- :book: A curated list of resources for NLP (Natural Language Processing) for Hebrew - iddoberger/awesome-hebrew-nlp. [Awesome hebrew nlp git](https://github.com/iddoberger/awesome-hebrew-nlp)
- HebMorph/dotNet/HebMorph/HSpell/Constants.cs at master · synhershko/HebMorph. [git](https://github.com/synhershko/HebMorph/blob/master/dotNet/HebMorph/HSpell/Constants.cs)
- אנו מספקים יכולות עיבוד שפה טבעית לעברית כשרות ענן. אנו מספקים יכולות עיבוד שפה טבעית לעברית כשרות ענן. [Hebrew-nlp service](https://hebrew-nlp.co.il/)
- HebrewNLP API Documentation. HebrewNLP API Documentation. [docs](https://docs.hebrew-nlp.co.il/#/README)
- שרותי ענן לשימוש פשוט וקל בתוכנות והבאפליקציות שלכם. שרותי ענן לשימוש פשוט וקל בתוכנות והבאפליקציות שלכם. [the features](https://hebrew-nlp.co.il/features)
- (morphological analysis, normalization etc) [git](https://github.com/HebrewNLP) morphological analysis normalization etc
- LanguageAnalysis - Solr - Apache Software Foundation. LanguageAnalysis - Solr - Apache Software Foundation. [Apache solr stop words (dead)](https://wiki.apache.org/solr/LanguageAnalysis#Hebrew)
- [SO on hebrew analyzer/stemming](https://stackoverflow.com/questions/1063856/lucene-hebrew-analyzer)
- [here too](https://stackoverflow.com/questions/20953495/is-there-a-good-stemmer-for-hebrew)
- Neural Sentiment Analyzer for Modern Hebrew. [Neural sentiment benchmark using two algorithms, for character and word level lstm/gru](https://github.com/omilab/Neural-Sentiment-Analyzer-for-Modern-Hebrew)
- Not Acceptable! Not Acceptable! - [the paper](http://aclweb.org/anthology/C18-1190)
- The code behind the blog post: https://www.oreilly.com/learning/capturing-semantic-meanings-using-deep-learning - liorshk/wordembedding-hebrew. [Hebrew word embeddings](https://github.com/liorshk/wordembedding-hebrew)
- Not Acceptable! Not Acceptable! [Paper for rich morphological datasets for comparison - rivlin](https://aclweb.org/anthology/C18-1190)

### Swiss army knife libraries

This section points at textacy as a spaCy-based Swiss-army-knife NLP library.

The same notes are in [SPACY](nlp.md#spacy).

1. [textacy](https://textacy.readthedocs.io/en/latest/) is a Python library for performing a variety of natural language processing (NLP) tasks, built on the high-performance spacy library. With the fundamentals — tokenization, part-of-speech tagging, dependency parsing, etc. — delegated to another library, textacy focuses on the tasks that come before and follow after.

## NLP for hackers tutorials

This section is a reading list of NLP-for-hackers tutorials, from WordNet through spaCy.

1. [How to convert between verb/noun/adjective/adverb forms using Wordnet](https://nlpforhackers.io/convert-words-between-forms/)
2. [Complete guide for training your own Part-Of-Speech Tagger -](https://nlpforhackers.io/training-pos-tagger/) using [Penn Treebank tagset](https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html). Using nltk or stanford pos taggers, creating features from actual words (manual stemming, etc0 using the tags as labels, on a random forest, thus creating a classifier for POS on our own. Not entirely sure why we need to create a classifier from a “classifier”.
3. [Word net introduction](https://nlpforhackers.io/starting-wordnet/) - POS, lemmatize, synon, antonym, hypernym, hyponym
4. [Sentence similarity using wordnet](https://nlpforhackers.io/wordnet-sentence-similarity/) - using synonyms cumsum for comparison. Today replaced with w2v mean sentence similarity.
5. [Stemmers vs lemmatizers](https://nlpforhackers.io/stemmers-vs-lemmatizers/) - stemmers are faster, lemmatizers are POS / dictionary based, slower, converting to base form.
6. [Chunking](https://nlpforhackers.io/text-chunking/) - shallow parsing, compared to deep, similar to NER
7. [NER -](https://nlpforhackers.io/named-entity-extraction/) using nltk chunking as a labeller for a classifier, training one of our own. Using IOB features as well as others to create a new ner classifier which should be better than the original by using additional features. Aso uses a new english dataset GMB.
8. [Building nlp pipelines, functions coroutines etc..](https://nlpforhackers.io/building-a-nlp-pipeline-in-nltk/)
9. [Training ner using generators](https://nlpforhackers.io/training-ner-large-dataset/)
10. [Metrics, tp/fp/recall/precision/micro/weighted/macro f1](https://nlpforhackers.io/classification-performance-metrics/)
11. [Tf-idf](https://nlpforhackers.io/tf-idf/)
12. [Nltk for beginners](https://nlpforhackers.io/introduction-nltk/)
- URL Source: https://nlpforhackers.io/corpora/, by GoDaddy.com LLC. [Nlp corpora](https://nlpforhackers.io/corpora/)
14. [bow/bigrams](https://nlpforhackers.io/language-models/)
15. [Textrank](https://nlpforhackers.io/textrank-text-summarization/)

The same notes are in [Summarization](summarization.md).

16. [Word cloud](https://nlpforhackers.io/word-clouds/)
17. [Topic modelling using gensim, lsa, lsi, lda,hdp](https://nlpforhackers.io/topic-modeling/)
18. [Spacy full tutorial](https://nlpforhackers.io/complete-guide-to-spacy/)
19. [POS using CRF](https://nlpforhackers.io/crf-pos-tagger/)

## Deprecated links

{% hint style="warning" %}
These links and images no longer work. The original wording is kept here. A same-resource copy, when one was checked, is used above.
{% endhint %}

- Benchmarking tokenizers for optimalprocessing speed This address no longer opens: https://towardsdatascience.com/benchmarking-python-nlp-tokenizers-3ac4735100c5
- Using nltk with gensim This address no longer opens: https://www.scss.tcd.ie/~munnellg/projects/visualizing-text.html
- DL for text classification This address no longer opens: https://ahmedbesbes.com/overview-and-benchmark-of-traditional-and-deep-learning-models-in-text-classification.html
- glorified regex extractor This address no longer opens: https://towardsdatascience.com/lexnlp-library-for-automated-text-extraction-ner-with-bafd0014a3f8
- Coding Chunkers as Taggers: IO, BIO, BMEWO, and BMEWO+ This address no longer opens: https://lingpipe-blog.com/2009/10/14/coding-chunkers-as-taggers-io-bio-bmewo-and-bmewo/
- textacy This address no longer opens: https://chartbeat-labs.github.io/textacy/
- https://vocabulary.readthedocs.io/en/… This address no longer opens: https://vocabulary.readthedocs.io/en/…
- collocation This address no longer opens: http://compling.hss.ntu.edu.sg/courses/hg2051/week09.html
- 2 This address no longer opens: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.68.2870&rep=rep1&type=pdf
