# Foundation NLP

Before any neural model, text has to be tokenized, cut into chunks, stemmed, and turned into phrases a counter can see. This page walks that ground in order: a basic pipeline and its classifiers, then stemming, chunking, collocations, and phrase models, then word-level lookups such as synonyms and semantic roles, and finally document classification, language detection, Hebrew tools, and the NLP-for-hackers tutorial run.

## Basic nlp

The first step of any pipeline is splitting a string into tokens, and it is also the step people wait on. [Benchmarking tokenizers for optimalprocessing speed](https://medium.com/data-science/benchmarking-python-nlp-tokenizers-3ac4735100c5) is Andrew Long's benchmark of Python NLP tokenizers, written because a tokenizer, the function that breaks a string into a list of words, is slow when there is a lot of text for a bag-of-words classifier. A second item, using nltk with gensim, used to sit here; that address no longer opens and is kept at the end of the page.

Once text is tokens, the obvious job is classification. [Multiclass text classification with svm/nb/mean w2v/](https://medium.com/data-science/multi-class-text-classification-model-comparison-and-selection-5eb066197568) d2v tutorial with code and notebook is Susan Li's comparison and selection of models for multi-class text classification, the same try-several-algorithms search used in any supervised problem. The same pipeline also yields keywords: [Basic pipeline for keyword extraction](https://medium.com/analytics-vidhya/automated-keyword-extraction-from-articles-using-nlp-bfd864f41b34) is Sowmya Vivek's automated keyword extraction from articles, where keywords give a concise representation of the content and help retrieval, bibliographic search, and categorizing the article.

The DL for text classification overview and benchmark that compared traditional and deep models is gone (it is kept at the end of the page), but its ladder of models is still the useful part:

 1. Logistic regression with word ngrams
 2. Logistic regression with character ngrams
 3. Logistic regression with word and character ngrams
 4. Recurrent neural network (bidirectional GRU) without pre-trained embeddings
 5. Recurrent neural network (bidirectional GRU) with GloVe pre-trained embeddings
 6. Multi channel Convolutional Neural Network
 7. RNN (Bidirectional GRU) + CNN model

When the target is structured fields rather than a label, a rule layer is often enough. LexNLP is a [glorified regex extractor](https://medium.com/data-science/lexnlp-library-for-automated-text-extraction-ner-with-bafd0014a3f8): ContraxSuite's library for automated text extraction and NER, introduced through leasing forms with entity names, addresses, dates, amounts, and conditions.

## Stemming

Classifiers and extractors both work better when word forms collapse to a common root, which raises the question the author asks here: how to measure a stemmer?

How to measure a stemmer?

The references are the evaluation papers. References \ [[1](https://files.eric.ed.gov/fulltext/EJ1020841.pdf) is the first paper, from the ERIC full-text archive. Reference 2(apr11) is the one whose address is kept at the end of the page, and [3](http://www.informationr.net/ir/19-1/paper605.html) is the next paper in that run. The measure called the (Index compression factor ICF) comes from [4](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.16.8310&rep=rep1&type=pdf), David A. Hull's "Stemming Algorithms - A Case Study for Detailed Evaluation". [5](https://pdfs.semanticscholar.org/1c0c/0fa35d4ff8a2f925eb955e48d655494bd167.pdf) is "A Comparative Study of Stemming Algorithms".

## Chunking

After single words are normalized, the next unit is a chunk, a run of words that belongs together, and chunks have to be encoded as tags before a tagger can learn them. [Coding Chunkers as Taggers: IO, BIO, BMEWO, and BMEWO+](https://web.archive.org/web/20160315214621/http://lingpipe-blog.com/2009/10/14/coding-chunkers-as-taggers-io-bio-bmewo-and-bmewo/) is the LingPipe post written after finishing a first-order linear-chain CRF tagger and its generalizations in the tagging interface, turning to coding chunkers with CRFs through the IO, BIO, and BMEWO schemes.

## Collocation

Chunks come from a tagger; collocations come from counting. They are word pairs that co-occur more than chance.

The same notes are in [Information Theory](../data/information-theory.md).

What is collocation? - “the habitual juxtaposition of a particular word with another word or words with a frequency greater than chance.” The Medium [tutorial](https://medium.com/@nicharuch/collocations-identifying-phrases-that-act-like-individual-words-in-nlp-f58a93a2f84a) is the place to start, quite good, comparing freq/t-test/pmi/chi2 with github code. For the full set of scores, a website dedicated to [collocations](http://www.collocations.de/) is a website dedicated methods references metrics.

Collocations also serve as features. [Text analysis for sentiment, doing feature selection](https://streamhacker.com/tag/chi-square/) is the StreamHacker thread on chi-square selection, and "Text Classification for Sentiment Analysis – Stopwords and Collocations" on StreamHacker is a tutorial with chi2(IG?), the [part 2 with bi-gram collocation in ntlk](https://streamhacker.com/2010/05/24/text-classification-sentiment-analysis-stopwords-collocations/). [Text2vec](http://text2vec.org/collocations.html) in R - has ideas on how to use collocations, for downstream tasks, LDA, W2V, etc. also explains about PMI and other metrics, note that gensim metric is unsupervised and probablistic.

For code, NLTK's own sample usage page on [collocations](http://www.nltk.org/howto/collocations.html) is the reference. Whether to strip stopwords first is the question in a [blog post](https://graus.nu/tag/gensim/) about keeping or removing stopwords for collocation, usefull but no firm conclusion. Imo we should remove it before. Another [blog post](http://n-chandra.blogspot.com/2014/06/collocation-extraction-using-nltk.html) comes with code of using nltk-based collocation, next to the small code for using nltk collocation. Another code / score example for nltk is the Stack Overflow question on understanding NLTK [collocation](https://stackoverflow.com/questions/8683588/understanding-nltk-collocation-scoring-for-bigrams-and-trigrams) scoring for bigrams and trigrams, asked by someone comparing which word pair is more likely to occur in US English. The Jupyter notebook on [manually finding collocation](https://github.com/sgsinclair/alta/blob/a482d343142cba12030fea4be8f96fb77579b3ab/ipynb/utilities/Collocates.ipynb) is from The Art of Literary Text Analysis; Jupyter notebook not useful.

Collocations can also go into the embedding itself. Paper: [Ngram2Vec](http://www.aclweb.org/anthology/D17-1023), "Learning Improved Word Representations from Ngram Co-occurrence Statistics" by Zhe Zhao, Tao Liu, Shen Li, Bofang Li, and Xiaoyong Du at EMNLP 2017, with its [Github](https://github.com/zhezhaoa/ngram2vec) of four word embedding models supporting arbitrary context features. We introduce ngrams into four representation methods. The experimental results demonstrate ngrams’ effectiveness for learning improved word representations. In addition, we find that the trained ngram embeddings are able to reflect their semantic meanings and syntactic patterns. To alleviate the costs brought by ngrams, we propose a novel way of building co-occurrence matrix, enabling the ngram-based models to run on cheap hardware

The scores behind all of this are in lectures. For language modeling: trigrams, by Francisco Iacobelli, the Youtube on [bigrams](https://www.youtube.com/watch?v=3i5QEmaOtkU&list=PLjTSKEJpqIeANubEWBo-z5TO89m7VtfG_) is the start, and his PMI lecture is the video on [collocation](https://www.youtube.com/watch?v=QvrbsjwErMA). The mutual info and [collocation](http://www.let.rug.nl/nerbonne/teach/rema-stats-meth-seminar/presentations/Suster-2011-MI-Coll.pdf) slides are Simon Šuster's association measures talk from the RUG Seminar in Statistics and Methodology; Youtube mutual info and collocation is the video pair above.

## Phrase modelling

Collocation scores find pairs; phrase modeling turns them into tokens. [Phrase Modeling](https://github.com/explosion/spacy-notebooks/blob/master/notebooks/conference_notebooks/modern_nlp_in_python.ipynb) is the modern NLP in Python notebook from explosion's spaCy notebooks, using gensim and spacy.

Phrase modeling is another approach to learning combinations of tokens that together represent meaningful multi-word concepts. We can develop phrase models by looping over the the words in our reviews and looking for words that co-occur (i.e., appear one after another) together much more frequently than you would expect them to by random chance. The formula our phrase models will use to determine whether two tokens AA and BB constitute a phrase is:

$$\frac{\mathrm{count}(A\,B)-\mathrm{count}_{\min}}{\mathrm{count}(A)\cdot\mathrm{count}(B)}\cdot N>\mathrm{threshold}$$

For other ways to pull phrases out of a corpus in Python, the Quora thread [ SO on PE.](https://www.quora.com/Whats-the-best-way-to-extract-phrases-from-a-corpus-of-text-using-Python) asks exactly that.

## Synonyms

Phrases group words that appear together; synonyms group words that mean the same thing.

The same notes are in [Augmentation](augmentation.md).

Vocabulary is a Python Module to get Meanings, Synonyms and what not for a given word using vocabulary (also a comparison against word net), documented at [https://vocabulary.readthedocs.io/en/latest/](https://vocabulary.readthedocs.io/en/latest/). This is a genuine feature list: for a given word, using Vocabulary, you can get its

- Meaning
- Synonyms
- Antonyms
- Part of speech : whether the word is a noun, interjection or an adverb et el
- Translate : Translate a phrase from a source language to the desired language.
- Usage example : a quick example on how to use the word in a sentence
- Pronunciation
- Hyphenation : shows the particular stress points(if any)

## Semantic roles:

Synonyms describe a word alone; semantic roles describe what a word does in the sentence. The note on semantic roles is [http://language.worldofcomputing.net/semantics/semantic-roles.html](http://language.worldofcomputing.net/semantics/semantic-roles.html), and the same page is linked again as [http://language.worldofcomputing.net/semantics/semantic-roles.html](http://language.worldofcomputing.net/semantics/semantic-roles.html) and [http://language.worldofcomputing.net/semantics/semantic-roles.html](http://language.worldofcomputing.net/semantics/semantic-roles.html).

The multiclass text classification with svm/nb/mean w2v/ comparison from the basic pipeline also sits here under its older address, [https://towardsdatascience.com/multi-class-text-classification-model-comparison-and-selection-5eb066197568](https://towardsdatascience.com/multi-class-text-classification-model-comparison-and-selection-5eb066197568).

## Document classification

Word-level features stop working when documents are long, which is where hierarchy helps.

The same notes are in [Intent Recognition](intent-recognition.md).

[Using hierarchical attention network](https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf) is the paper by Zichao Yang, Diyi Yang, Chris Dyer, Xiaodong He, Alex Smola, and Eduard Hovy of Carnegie Mellon University and Microsoft Research.

## Language detection

Before any of the steps above, a pipeline has to know which language it is reading.

The same notes are in [LANGUAGE DETECTION / IDENTIFICATION](language-detection-identification-generation-nld-nli-nlg.md#language-detection--identification).

[Using google lang detect](https://github.com/Mimino666/langdetect) is the wrapper for that check, with 55 languages af, ar, bg, bn, ca, cs, cy, da, de, el, en, es, et, fa, fi, fr, gu, he, hi, hr, hu, id, it, ja, kn, ko, lt, lv, mk, ml, mr, ne, nl, no, pa, pl, pt, ro, ru, sk, sl, so, sq, sv, sw, ta, te, th, tl, tr, uk, ur, vi, zh-cn, zh-tw.

## Hebrew NLP tools

Hebrew is on that list, and it needs its own tools for search, morphology, stemming, and embeddings.

Search is where most of the Hebrew tooling started. [HebMorph](https://github.com/synhershko/HebMorph.CorpusSearcher) is the sample app for using HebMorph, and [Hebmorph elastic search](https://github.com/synhershko/elasticsearch-analysis-hebrew/wiki/Getting-Started) is the Hebrew analyzer plugin for elasticsearch. The [Hebmorph blog post](https://code972.com/blog/2013/12/673-hebrew-search-done-right) is "Hebrew search done right", and the other [blog posts](https://code972.com/hebmorph) describe HebMorph as an open-source solution (free for non-commercial use) enabling Hebrew search, via Elasticsearch, Solr and Lucene analysis plugins. The [youtube](https://www.youtube.com/watch?v=v8w32wC6ppI) talk is Itamar Syn-Hershko's "Open Source / HebMorph - Hebrew made searchable" at Reversim. In the code itself, the HSpell constants file in HebMorph is the [git](https://github.com/synhershko/HebMorph/blob/master/dotNet/HebMorph/HSpell/Constants.cs) link, from the effort to make Hebrew properly searchable by IR libraries while keeping recall, precision, and relevancy.

For a wider list, [Awesome hebrew nlp git](https://github.com/iddoberger/awesome-hebrew-nlp) is a curated list of resources for NLP for Hebrew. As a cloud service, the [Hebrew-nlp service](https://hebrew-nlp.co.il/) offers Hebrew natural language processing, starting with a morphological engine: אנו מספקים יכולות עיבוד שפה טבעית לעברית כשרות ענן. Its [docs](https://docs.hebrew-nlp.co.il/#/README) are the HebrewNLP API Documentation, and [the features](https://hebrew-nlp.co.il/features) page lists the services: שרותי ענן לשימוש פשוט וקל בתוכנות והבאפליקציות שלכם. The HebrewNLP API libraries and examples are on [git](https://github.com/HebrewNLP) (morphological analysis, normalization etc), morphological analysis normalization etc.

Stopwords and stemming are the recurring gap. The Solr LanguageAnalysis page was the reference for [Apache solr stop words (dead)](https://wiki.apache.org/solr/LanguageAnalysis#Hebrew). The [SO on hebrew analyzer/stemming](https://stackoverflow.com/questions/1063856/lucene-hebrew-analyzer) question asks whether a Lucene Hebrew analyzer exists at all, and [here too](https://stackoverflow.com/questions/20953495/is-there-a-good-stemmer-for-hebrew) asks for a good Hebrew stemmer and whether lemmas can stand in for stems, since for Semitic languages the two terms are used interchangeably.

Past search, the same morphology question shows up in sentiment and embeddings. [Neural sentiment benchmark using two algorithms, for character and word level lstm/gru](https://github.com/omilab/Neural-Sentiment-Analyzer-for-Modern-Hebrew) is the Neural Sentiment Analyzer for Modern Hebrew, and [the paper](http://aclweb.org/anthology/C18-1190) behind it is "Representations and Architectures in Neural Sentiment Analysis for Morphologically Rich Languages: A Case Study from Modern Hebrew" by Adam Amram, Anat Ben David, and Reut Tsarfaty. [Hebrew word embeddings](https://github.com/liorshk/wordembedding-hebrew) is the code behind the blog post: https://www.oreilly.com/learning/capturing-semantic-meanings-using-deep-learning. The same COLING 2018 paper is also the [Paper for rich morphological datasets for comparison - rivlin](https://aclweb.org/anthology/C18-1190).

### Swiss army knife libraries

Rather than assembling each of these steps by hand, a Swiss-army-knife library bundles them on top of spaCy.

The same notes are in [SPACY](nlp.md#spacy).

[textacy](https://textacy.readthedocs.io/en/latest/) is a Python library for performing a variety of natural language processing (NLP) tasks, built on the high-performance spacy library. With the fundamentals — tokenization, part-of-speech tagging, dependency parsing, etc. — delegated to another library, textacy focuses on the tasks that come before and follow after.

## NLP for hackers tutorials

The last section is a tutorial run that retraces most of this page, from WordNet through spaCy. Most of its addresses now point to an unrelated site and are kept at the end of the page; the author's notes on the ones that still open, and on the ones that do not, stay here.

The tutorial on converting between verb, noun, adjective, and adverb forms with WordNet is one of the lost ones. So is the complete guide for training your own Part-Of-Speech Tagger, which used the [Penn Treebank tagset](https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html). Using nltk or stanford pos taggers, creating features from actual words (manual stemming, etc0 using the tags as labels, on a random forest, thus creating a classifier for POS on our own. Not entirely sure why we need to create a classifier from a “classifier”. The WordNet introduction, which covered POS, lemmatize, synon, antonym, hypernym, hyponym, is also kept at the end.

The same tutorial series went on to four more notes, whose addresses now point to an unrelated site and are kept at the end of the page. Sentence similarity using wordnet - using synonyms cumsum for comparison. Today replaced with w2v mean sentence similarity. Stemmers vs lemmatizers - stemmers are faster, lemmatizers are POS / dictionary based, slower, converting to base form, the same trade-off as the stemming section above. Chunking - shallow parsing, compared to deep, similar to NER, the tutorial counterpart of the chunking tags above. NER - using nltk chunking as a labeller for a classifier, training one of our own. Using IOB features as well as others to create a new ner classifier which should be better than the original by using additional features. Aso uses a new english dataset GMB.

The rest of the run is kept at the end of the page: building nlp pipelines with functions and coroutines, training NER using generators, classification metrics, TF-IDF, NLTK for beginners, NLP corpora, bow/bigrams language models, and TextRank.

The same notes are in [Summarization](summarization.md).

After TextRank the run closed with word clouds, topic modelling using gensim, lsa, lsi, lda, hdp, a full spaCy tutorial, and POS using CRF, all of which are kept at the end of the page as well.

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
- How to convert between verb/noun/adjective/adverb forms using Wordnet. This address now points to an unrelated site: https://nlpforhackers.io/convert-words-between-forms/
- Complete guide for training your own Part-Of-Speech Tagger -. This address now points to an unrelated site: https://nlpforhackers.io/training-pos-tagger/
- Word net introduction - POS, lemmatize, synon, antonym, hypernym, hyponym. This address now points to an unrelated site: https://nlpforhackers.io/starting-wordnet/
- Building nlp pipelines, functions coroutines etc... This address now points to an unrelated site: https://nlpforhackers.io/building-a-nlp-pipeline-in-nltk/
- Training ner using generators. This address now points to an unrelated site: https://nlpforhackers.io/training-ner-large-dataset/
- Metrics, tp/fp/recall/precision/micro/weighted/macro f1. This address now points to an unrelated site: https://nlpforhackers.io/classification-performance-metrics/
- Tf-idf. This address now points to an unrelated site: https://nlpforhackers.io/tf-idf/
- Nltk for beginners. This address now points to an unrelated site: https://nlpforhackers.io/introduction-nltk/
- Nlp corpora. This address now points to an unrelated site: https://nlpforhackers.io/corpora/
- bow/bigrams. This address now points to an unrelated site: https://nlpforhackers.io/language-models/
- Textrank. This address now points to an unrelated site: https://nlpforhackers.io/textrank-text-summarization/
- Word cloud. This address now points to an unrelated site: https://nlpforhackers.io/word-clouds/
- Topic modelling using gensim, lsa, lsi, lda,hdp. This address now points to an unrelated site: https://nlpforhackers.io/topic-modeling/
- Spacy full tutorial. This address now points to an unrelated site: https://nlpforhackers.io/complete-guide-to-spacy/
- POS using CRF. This address now points to an unrelated site: https://nlpforhackers.io/crf-pos-tagger/
- Sentence similarity using wordnet. This address now points to an unrelated site: https://nlpforhackers.io/wordnet-sentence-similarity/
- Stemmers vs lemmatizers. This address now points to an unrelated site: https://nlpforhackers.io/stemmers-vs-lemmatizers/
- Chunking. This address now points to an unrelated site: https://nlpforhackers.io/text-chunking/
- NER -. This address now points to an unrelated site: https://nlpforhackers.io/named-entity-extraction/
