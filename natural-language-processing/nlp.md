# Natural Language Programming (Many Topics)

### **NLP - a reality check**

1. [**A powerful benchmark**](https://github.com/KevinMusgrave/powerful-benchmarke)**,** [**paper**](https://arxiv.org/pdf/2003.08505.pdf)**,** [**medium**](https://medium.com/@tkm45/updates-to-a-metric-learning-reality-check-730b6914dfe7) **- normalizing data sets allows us to see that there wasn't any advancement in terms of metrics in many NLP algorithms.**

### **TOOLS**

#### **SPACY**&#x20;

1. [**Vidhaya on spacy vs ner**](https://www.analyticsvidhya.com/blog/2017/04/natural-language-processing-made-easy-using-spacy-%E2%80%8Bin-python/) **- tutorial + code on how to use spacy for pos, dep, ner, compared to nltk/corenlp (sner etc). The results reflect a global score not specific to LOC for example.**
2. **The** [**spaCy course**](https://course.spacy.io)
3. **SPACY OPTIMIZATION -** [**LP using CYTHON and SPACY.**](https://medium.com/huggingface/100-times-faster-natural-language-processing-in-python-ee32033bdced)
4.

### **NLP embedding repositories**

1. [**Nlpl**](http://vectors.nlpl.eu/repository/)

### **NLP DATASETS**

1. [**The bid bad**](https://datasets.quantumstat.com) **600,** [**medium**](https://medium.com/towards-artificial-intelligence/600-nlp-datasets-and-glory-4b0080bf5ab)

### **NLP Libraries**

1. [**Has all the known libraries**](https://nlpforhackers.io/libraries/)
2. [**Comparison between spacy, pytorch, allenlp**](https://luckytoilet.wordpress.com/2018/12/29/deep-learning-for-nlp-spacy-vs-pytorch-vs-allennlp/?fbclid=IwAR236Mrg4J4pBGSLlvQ8xNbEw21lvMeLi6CfqRB2x6BL1U9vJm7\_mB7Q10E) **- very basic info**
3. [**Comparison spacy,nltk**](https://spacy.io/usage/facts-figures) **core nlp**
4. [**Comparing Production grade nlp libs**](https://www.oreilly.com/ideas/comparing-production-grade-nlp-libraries-accuracy-performance-and-scalability)
5. [**nltk vs spac**](https://blog.thedataincubator.com/2016/04/nltk-vs-spacy-natural-language-processing-in-python/)**y**

### **Multilingual models**

1. [**Fb’s laser**](https://engineering.fb.com/ai-research/laser-multilingual-sentence-embeddings/)
2. [**Xlm**](https://github.com/facebookresearch/XLM)**,** [**xlm-r**](https://ai.facebook.com/blog/-xlm-r-state-of-the-art-cross-lingual-understanding-through-self-supervision/)
3. **Google universal embedding space.**

### **Augmenting text in NLP**

1. [**Synonyms**](https://towardsdatascience.com/data-augmentation-in-nlp-2801a34dfc28)**, similar embedded words (w2v), back translation, contextualized word embeddings, text generation**
2. **Yonatan hadar also has a medium post about this**

### **TF-IDF**

[**TF-IDF**](http://www.tfidf.com) **- how important is a word to a document in a corpus**

**TF(t) = (Number of times term t appears in a document) / (Total number of terms in the document).**

**Frequency of word in doc / all words in document (normalized bcz docs have diff sizes)**

**IDF(t) = log\_e(Total number of documents / Number of documents with term t in it).**

**measures how important a term is**

**TF-IDF is TF\*IDF**\
****

1. [**A much clearer explanation plus python code**](https://stevenloria.com/tf-idf/)**,** [**part 2**](http://blog.christianperone.com/2011/10/machine-learning-text-feature-extraction-tf-idf-part-ii/)
2. [**Get top tfidf keywords**](https://stackoverflow.com/questions/34232190/scikit-learn-tfidfvectorizer-how-to-get-top-n-terms-with-highest-tf-idf-score)
3. [**Print top features**](https://gist.github.com/StevenMaude/ea46edc315b0f94d03b9)

**Data sets:**

1. [**Fast text multilingual**](https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md)
2. [**NLP embeddings**](http://vectors.nlpl.eu/repository/#)

### **Sparse textual content**

1. **mean(IDF(i) \* w2v word vectors (i)) with or without reducing PC1 from the whole w2 average (amir pupko)**\
   ****\
   ****

**def mean\_weighted\_embedding(model, words, idf=1.0):**

&#x20;   **if words:**

&#x20;       **return np.mean(idf \* model\[words], axis=0)a**

&#x20;   **else:**

&#x20;       **print('we have an empty list')**

&#x20;       **return \[]**\
****

**idf\_mapping = dict(zip(vectorizer.get\_feature\_names(), vectorizer.idf\_))**&#x20;

**logs\_sequences\_df\['idf\_vectors'] = logs\_sequences\_df.message.apply(lambda x: \[idf\_mapping\[token] for token in splitter(x)])**

**logs\_sequences\_df\['mean\_weighted\_idf\_w2v'] = \[mean\_weighted\_embedding(ft, splitter(logs\_sequences\_df\['message'].iloc\[i]), 1 / np.array(logs\_sequences\_df\['idf\_vectors'].iloc\[i]).reshape(-1,1)) for i in range(logs\_sequences\_df.shape\[0])]**\
****\
****

1. [**Multiply by TFIDF**](https://towardsdatascience.com/supercharging-word-vectors-be80ee5513d)
2. **Enriching using lstm-next word (char or word-wise)**
3. **Using external wiktionary/pedia data for certain words, phrases**
4. **Finding clusters of relevant data and figuring out if you can enrich based on the content of the clusters**
5. [**Applying deep nlp methods without big data, i.e., sparseness**](https://towardsdatascience.com/lessons-learned-from-applying-deep-learning-for-nlp-without-big-data-d470db4f27bf?\_branch\_match\_id=584170448791192656)

### **Basic nlp**

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

### **Chunking**

1. [**Coding Chunkers as Taggers: IO, BIO, BMEWO, and BMEWO+**](https://lingpipe-blog.com/2009/10/14/coding-chunkers-as-taggers-io-bio-bmewo-and-bmewo/)

### **NLP for hackers tutorials**

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

### **Synonyms**&#x20;

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

**Swiss army knife libraries**

1. [**textacy**](https://chartbeat-labs.github.io/textacy/) **is a Python library for performing a variety of natural language processing (NLP) tasks, built on the high-performance spacy library. With the fundamentals — tokenization, part-of-speech tagging, dependency parsing, etc. — delegated to another library, textacy focuses on the tasks that come before and follow after.**

**Collocation**&#x20;

1. **What is collocation? - “the habitual juxtaposition of a particular word with another word or words with a frequency greater than chance.”Medium** [**tutorial**](https://medium.com/@nicharuch/collocations-identifying-phrases-that-act-like-individual-words-in-nlp-f58a93a2f84a)**, quite good, comparing freq/t-test/pmi/chi2 with github code**
2. **A website dedicated to** [**collocations**](http://www.collocations.de)**, methods, references, metrics.**
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

**Language detection**

1. [**Using google lang detect**](https://github.com/Mimino666/langdetect) **- 55 languages af, ar, bg, bn, ca, cs, cy, da, de, el, en, es, et, fa, fi, fr, gu, he,**\
   **hi, hr, hu, id, it, ja, kn, ko, lt, lv, mk, ml, mr, ne, nl, no, pa, pl,**\
   **pt, ro, ru, sk, sl, so, sq, sv, sw, ta, te, th, tl, tr, uk, ur, vi, zh-cn, zh-tw**

**Stemming**

**How to measure a stemmer?**

1. **References \[**[**1**](https://files.eric.ed.gov/fulltext/EJ1020841.pdf) **** [**2**](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.68.2870\&rep=rep1\&type=pdf)**(apr11)** [**3**](http://www.informationr.net/ir/19-1/paper605.html)**(Index compression factor ICF)** [**4**](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.16.8310\&rep=rep1\&type=pdf) **** [**5**](https://pdfs.semanticscholar.org/1c0c/0fa35d4ff8a2f925eb955e48d655494bd167.pdf)**]**

**Phrase modelling**

1. [**Phrase Modeling**](https://github.com/explosion/spacy-notebooks/blob/master/notebooks/conference\_notebooks/modern\_nlp\_in\_python.ipynb) **- using gensim and spacy**

**Phrase modeling is another approach to learning combinations of tokens that together represent meaningful multi-word concepts. We can develop phrase models by looping over the the words in our reviews and looking for words that co-occur (i.e., appear one after another) together much more frequently than you would expect them to by random chance. The formula our phrase models will use to determine whether two tokens AA and BB constitute a phrase is:**

**count(A B)−countmincount(A)∗count(B)∗N>threshold**

1. [ **SO on PE.**](https://www.quora.com/Whats-the-best-way-to-extract-phrases-from-a-corpus-of-text-using-Python)
2.

**Document classification**

1. [**Using hierarchical attention network**](https://www.cs.cmu.edu/\~hovy/papers/16HLT-hierarchical-attention-networks.pdf)

**Hebrew NLP tools**

1. [**HebMorph**](https://github.com/synhershko/HebMorph.CorpusSearcher) **last update 7y ago**
2. [**Hebmorph elastic search**](https://github.com/synhershko/elasticsearch-analysis-hebrew/wiki/Getting-Started) **** [**Hebmorph blog post**](https://code972.com/blog/2013/12/673-hebrew-search-done-right)**, and other** [**blog posts**](https://code972.com/hebmorph)**,** [**youtube**](https://www.youtube.com/watch?v=v8w32wC6ppI)
3. [**Awesome hebrew nlp git**](https://github.com/iddoberger/awesome-hebrew-nlp)**,** [**git**](https://github.com/synhershko/HebMorph/blob/master/dotNet/HebMorph/HSpell/Constants.cs)
4. [**Hebrew-nlp service**](https://hebrew-nlp.co.il) **** [**docs**](https://docs.hebrew-nlp.co.il/#/README) **** [**the features**](https://hebrew-nlp.co.il/features) **(morphological analysis, normalization etc),** [**git**](https://github.com/HebrewNLP)
5. [**Apache solr stop words (dead)**](https://wiki.apache.org/solr/LanguageAnalysis#Hebrew)
6. [**SO on hebrew analyzer/stemming**](https://stackoverflow.com/questions/1063856/lucene-hebrew-analyzer)**,** [**here too**](https://stackoverflow.com/questions/20953495/is-there-a-good-stemmer-for-hebrew)
7. [**Neural sentiment benchmark using two algorithms, for character and word level lstm/gru**](https://github.com/omilab/Neural-Sentiment-Analyzer-for-Modern-Hebrew) **-** [**the paper**](http://aclweb.org/anthology/C18-1190)
8. [**Hebrew word embeddings**](https://github.com/liorshk/wordembedding-hebrew)
9. [**Paper for rich morphological datasets for comparison - rivlin**](https://aclweb.org/anthology/C18-1190)

**Semantic roles:**

1. [**http://language.worldofcomputing.net/semantics/semantic-roles.html**](http://language.worldofcomputing.net/semantics/semantic-roles.html)

### **ANNOTATION**

1. [**Snorkel**](https://www.snorkel.org/use-cases/) **- using weak supervision to create less noisy labelled datasets**
   1. [**Git**](https://github.com/snorkel-team/snorkel)
   2. [**Medium**](https://towardsdatascience.com/introducing-snorkel-27e4b0e6ecff)
2. [**Snorkel metal**](https://jdunnmon.github.io/metal\_deem.pdf) **weak supervision for multi-task learning.** [**Conversation**](https://spectrum.chat/snorkel/help/hierarchical-labelling-example\~aa4d8617-d287-43a6-865e-7c9034888363)**,** [**git**](https://github.com/HazyResearch/metal/blob/master/tutorials/Multitask.ipynb)
   1. **Yes, the Snorkel project has included work before on hierarchical labeling scenarios. The main papers detailing our results include the DEEM workshop paper you referenced (**[**https://dl.acm.org/doi/abs/10.1145/3209889.3209898**](https://dl.acm.org/doi/abs/10.1145/3209889.3209898)**) and the more complete paper presented at AAAI (**[**https://arxiv.org/abs/1810.02840**](https://arxiv.org/abs/1810.02840)**). Before the Snorkel and Snorkel MeTaL projects were merged in Snorkel v0.9, the Snorkel MeTaL project included an interface for explicitly specifying hierarchies between tasks which was utilized by the label model and could be used to automatically compile a multi-task end model as well (demo here:** [**https://github.com/HazyResearch/metal/blob/master/tutorials/Multitask.ipynb**](https://github.com/HazyResearch/metal/blob/master/tutorials/Multitask.ipynb)**). That interface is not currently available in Snorkel v0.9 (no fundamental blockers; just hasn't been ported over yet).**
   2. **There are, however, still a number of ways to model such situations. One way is to treat each node in the hierarchy as a separate task and combine their probabilities post-hoc (e.g., P(credit-request) = P(billing) \* P(credit-request | billing)). Another is to treat them as separate tasks and use a multi-task end model to implicitly learn how the predictions of some tasks should affect the predictions of others (e.g., the end model we use in the AAAI paper). A third option is to create a single task with all the leaf categories and modify the output space of the LFs you were considering for the higher nodes (the deeper your hierarchy is or the larger the number of classes, the less apppealing this is w/r/t to approaches 1 and 2).**
3. [**mechanical turk calculator**](https://morninj.github.io/mechanical-turk-cost-calculator/)
4. [**Mturk alternatives**](https://moneypantry.com/amazon-mechanical-turk-crowdsourcing-alternatives/)
   1. [**Workforce / onespace**](https://www.crowdsource.com/workforce/)
   2. [**Jobby**](https://www.jobboy.com)
   3. [**Shorttask**](http://www.shorttask.com)
   4. [**Samasource**](https://www.samasource.org/team)
   5. **Figure 8 -** [**pricing**](https://siftery.com/crowdflower/pricing) **-** [**definite guide**](https://www.earnonlineguys.com/figure-eight-tasks-guide/)
5. [**Brat nlp annotation tool**](http://brat.nlplab.org/?fbclid=IwAR1bDCM3j3nEQb3Hrf9dGCwyRvDVMBXoob4WtVLCWAMBgPraZmkSi123IrI)
6. [**Prodigy by spacy**](https://prodi.gy)**,**&#x20;
   1. [**seed-small sample, many sample tutorial on youtube by ines**](https://www.youtube.com/watch?v=5di0KlKl0fE)
   2. [**How to use prodigy, tutorial on medium plus notebook code inside**](https://medium.com/@david.campion/text-classification-be-lazy-use-prodigy-b0f9d00e9495)
7. [**Doccano**](https://github.com/chakki-works/doccano) **- prodigy open source alternative butwith users management & statistics out of the box**
8. **Medium** [**Lighttag - has some cool annotation metrics\tests**](https://medium.com/@TalPerry/announcing-lighttag-the-easy-way-to-annotate-text-afb7493a49b8)
9. [Loopr](https://loopr.ai/products/labeling-platform).ai - An AI powered semi-automated and automated annotation process for high quality data.object detection, analytics, nlp, active learning.
10. **Medium** [**Assessing annotator disagreement**](https://towardsdatascience.com/assessing-annotator-disagreements-in-python-to-build-a-robust-dataset-for-machine-learning-16c74b49f043)
11. [**A great python package for measuring disagreement on GH**](https://github.com/o-P-o/disagree)
12. [**Reliability is key, and not just mechanical turk**](https://www.youtube.com/watch?v=ktZLuXPXPEI)
13. [**7 myths about annotation**](https://www.aaai.org/ojs/index.php/aimagazine/article/viewFile/2564/2468)
14. [**Annotating twitter sentiment using humans, 3 classes, 55% accuracy using SVMs.**](http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0155036) **they talk about inter agreement etc. and their DS is** [**partially publicly available**](https://www.clarin.si/repository/xmlui/handle/11356/1054)**.**
15. [**Exploiting disagreement** ](https://s3.amazonaws.com/academia.edu.documents/8026932/10.1.1.2.8084.pdf?AWSAccessKeyId=AKIAIWOWYYGZ2Y53UL3A\&Expires=1534444363\&Signature=3dHHw3EmAjPXFxwutVbtsZWEIzw%3D\&response-content-disposition=inline%3B%20filename%3DExploiting\_agreement\_and\_disagreement\_of.pdf)
16. [**Vader annotation**](http://comp.social.gatech.edu/papers/icwsm14.vader.hutto.pdf)
    1. **They must pass an english exam**
    2. **They get control questions to establish their reliability**
    3. **They get a few sentences over and over again to establish inter disagreement**
    4. **Two or more people get a overlapping sentences to establish disagreement**
    5. **5 judges for each sentence (makes 4 useless)**
    6. **They dont know each other**
    7. **Simple rules to follow**
    8. **Random selection of sentences**
    9. **Even classes**
    10. **No experts**
    11. **Measuring reliability kappa/the other kappa.**
17. [**Label studio**\
    ****](https://labelstud.io)![](https://lh3.googleusercontent.com/X2kRKqlPnkMZyspKgiJYHR5vyE2NnRfkYJZMxBs\_rfFeGaMl0L07hqCO8VRGnTV\_E9qhroCDYLIlQ1e78EgraeE6wwPE3WJDkzVmR6kQTgv4I-npCh3UkKnuBE\_C1Lo9dQ3QxcEg)

**Ideas:**&#x20;

1. **Active learning for a group (or single) of annotators, we have to wait for all annotations to finish each big batch in order to retrain the model.**
2. **Annotate a small group, automatic labelling using knn**
3. **Find a nearest neighbor for out optimal set of keywords per “category,**&#x20;
4. **For a group of keywords, find their knn neighbors in w2v-space, alternatively find k clusters in w2v space that has those keywords. For a new word/mean sentence vector in the ‘category’ find the minimal distance to the new cluster (either one of approaches) and this is new annotation.**

#### [**7 myths of annotation**](https://www.aaai.org/ojs/index.php/aimagazine/article/viewFile/2564/2468)

1. **Myth One: One Truth Most data collection efforts assume that there is one correct interpretation for every input example.**&#x20;
2. **Myth Two: Disagreement Is Bad To increase the quality of annotation data, disagreement among the annotators should be avoided or reduced.**&#x20;
3. **Myth Three: Detailed Guidelines Help When specific cases continuously cause disagreement, more instructions are added to limit interpretations.**&#x20;
4. **Myth Four: One Is Enough Most annotated examples are evaluated by one person.**&#x20;
5. **Myth Five: Experts Are Better Human annotators with domain knowledge provide better annotated data.**&#x20;
6. **Myth Six: All Examples Are Created Equal The mathematics of using ground truth treats every example the same; either you match the correct result or not.**&#x20;
7. **Myth Seven: Once Done, Forever Valid Once human annotated data is collected for a task, it is used over and over with no update. New annotated data is not aligned with previous data.**

#### [**Crowd Sourcing** ](https://www.youtube.com/watch?v=ktZLuXPXPEI)

![](https://lh3.googleusercontent.com/CpbWZ2kVN\_c84uZnRgfBAxTVBxBQArQDbMhZj12n8n8zRZIB-1FwOyEx7Yn2P\_sZ6qclUnfimvkKUsmSTXC3eFFIM49oHGhwMctXkPZUGFGXTAO3LlhZJv7Gw1TGr\_pDjRsIiCSc)

![](https://lh3.googleusercontent.com/Xo5pBUmwOyqKqnZJvJc2kyjzPZYiZLY4acF\_oK6Su6WsYCVuJygvdgDgjLRhPWdbcVsxO8qs6C1pHuH0ZWVVZ5-Z-F1fRlojJ-MYcaMUx56tE0Z2OxzJ02ieMNEhIAHiLnMwZKPi)

![](https://lh3.googleusercontent.com/Hx9UzYlcDRUIpf9Pt-f4xI9M8EwPapcEcwwXcmKry8VC0OzyI4kbrp7h4E7nOXeMMdR1wdd\_Dwa54THEBpvcwZbjmWHBQQEAzBGtB8RyF40xbx6AV4L9BErGcbRFM-AMHuN7GTq\_)![](https://lh4.googleusercontent.com/1VEsT95na9TLGXNUBwAGMKOdTJDI4cJ5rCirq\_WYhCne-xBmDTjcpJ4Qmoyh7OHW5ilBCnjpJ4U1opy1TK7v6-i4AmsqAbUm42YGg1Ee\_90HFblseEd1K6PyfTA7NTow6B6WsZtE)****\
****

![](https://lh3.googleusercontent.com/m1MAdhxW1T3\_-s0i6PHH-xCBfBpQLCqtVpL-WfUvVyR3A\_NT274te37PLRYjfCELOS0YB4zUNCAswBcG0fY4fMDlWh-hmz9kMCVfiM5xqyyZDc5NEfkIYt57O105II8kU5ccVnIG)

![](https://lh4.googleusercontent.com/s8A8VcNA22GZ5FtBnQaAJvxyJmw7jgEIp4LFw28z5OxoZwAfuoShsSSDSRa7Loqud-caBFY9lQK1xhbUrlwyhox2btt7hLMfbb\_L59BzFGxxgX35p-5bJdInEIkuWf6vBmmioaWe)

* **Conclusions:**&#x20;
  * **Experts are the same as a crowd**
  * **Costs a lot less \$$$.**

#### **Inter agreement**

**\*\*\*** [**The best tutorial on agreements, cohen, david, kappa, krip etc.**](https://dkpro.github.io/dkpro-statistics/inter-rater-agreement-tutorial.pdf)

1. **Cohens kappa (two people)**

&#x20;**but you can use it to map a group by calculating agreement for each pair**

1. [**Why cohens kappa should be avoided as a performance measure in classification**](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0222916)
2. [**Why it should be used as a measure of classification**](https://thedatascientist.com/performance-measures-cohens-kappa-statistic/)
3. [**Kappa in plain english**](https://stats.stackexchange.com/questions/82162/cohens-kappa-in-plain-english)
4. [**Multilabel using kappa**](https://stackoverflow.com/questions/52272901/multi-label-annotator-agreement-with-cohen-kappa)
5. [**Kappa and the relation with accuracy**](https://gis.stackexchange.com/questions/110188/how-are-kappa-and-overall-accuracy-related-with-respect-to-thematic-raster-data) **(redundant, % above chance, should not be used due to other reasons researched here)**

**The Kappa statistic varies from 0 to 1, where.**

* **0 = agreement equivalent to chance.**
* **0.1 – 0.20 = slight agreement.**
* **0.21 – 0.40 = fair agreement.**
* **0.41 – 0.60 = moderate agreement.**
* **0.61 – 0.80 = substantial agreement.**
* **0.81 – 0.99 = near perfect agreement**
* **1 = perfect agreement.**

1. **Fleiss’ kappa, from 3 people and above.**

**Kappa ranges from 0 to 1, where:**

* **0 is no agreement (or agreement that you would expect to find by chance),**
* **1 is perfect agreement.**
* **Fleiss’s Kappa is an extension of Cohen’s kappa for three raters or more. In addition, the assumption with Cohen’s kappa is that your raters are deliberately chosen and fixed. With Fleiss’ kappa, the assumption is that your raters were chosen at random from a larger population.**
* [**Kendall’s Tau**](https://www.statisticshowto.datasciencecentral.com/kendalls-tau/) **is used when you have ranked data, like two people ordering 10 candidates from most preferred to least preferred.**
* **Krippendorff’s alpha is useful when you have multiple raters and multiple possible ratings.**

1. **Krippendorfs alpha**&#x20;

* [**Ignores missing data entirely**](https://deepsense.ai/multilevel-classification-cohen-kappa-and-krippendorff-alpha/)**.**
* **Can handle various sample sizes, categories, and numbers of raters.**
* **Applies to any** [**measurement level**](https://www.statisticshowto.datasciencecentral.com/scales-of-measurement/) **(i.e. (**[**nominal, ordinal, interval, ratio**](https://www.statisticshowto.datasciencecentral.com/nominal-ordinal-interval-ratio/)**).**
* **Values range from 0 to 1, where 0 is perfect disagreement and 1 is perfect agreement. Krippendorff suggests: “\[I]t is customary to require α ≥ .800. Where tentative conclusions are still acceptable, α ≥ .667 is the lowest conceivable limit (2004, p. 241).”**
* [**Supposedly multi label**](https://stackoverflow.com/questions/57256287/calculate-kappa-score-for-multi-label-image-classifcation)

1. **MACE - the new kid on the block. -**

&#x20;**learns in an unsupervised fashion to**&#x20;

1. **a) identify which annotators are trustworthy and**
2. &#x20;**b) predict the correct underlying labels. We match performance of more complex state-of-the-art systems and perform well even under adversarial conditions**
3. [**MACE**](https://www.isi.edu/publications/licensed-sw/mace/) **does exactly that. It tries to find out which annotators are more trustworthy and upweighs their answers.**
4. [**Git**](https://github.com/dirkhovy/MACE) **-**

**When evaluating redundant annotatio**

**ns (like those from Amazon's MechanicalTurk), we usually want to**

1. **aggregate annotations to recover the most likely answer**
2. **find out which annotators are trustworthy**
3. **evaluate item and task difficulty**

**MACE solves all of these problems, by learning competence estimates for each annotators and computing the most likely answer based on those competences.**

1.

**Calculating agreement**

1. **Compare against researcher-ground-truth**
2. **Self-agreement**
3. **Inter-agreement**
   1. [**Medium**](https://towardsdatascience.com/inter-rater-agreement-kappas-69cd8b91ff75)
   2. [**Kappa**](https://stats.stackexchange.com/questions/82162/cohens-kappa-in-plain-english) **cohen**
   3. [**Multi annotator with kappa (which isnt), is this okay?**](https://stackoverflow.com/questions/52272901/multi-label-annotator-agreement-with-cohen-kappa)
   4. **Github computer Fleiss Kappa** [**1**](https://gist.github.com/skylander86/65c442356377367e27e79ef1fed4adee)
   5. [**Fleiss Kappa Example**](https://www.wikiwand.com/en/Fleiss'\_kappa#/Worked\_example)
   6. [**GWET AC1**](https://stats.stackexchange.com/questions/235929/fleiss-kappa-alternative-for-ranking)**,** [**paper**](https://s3.amazonaws.com/sitesusa/wp-content/uploads/sites/242/2014/05/J4\_Xie\_2013FCSM.pdf)**: as an alternative to kappa, and why**
   7. [**Website, krippensorf vs fleiss calculator**](https://nlp-ml.io/jg/software/ira/)

**Machine Vision annotation**

1. [**CVAT**](https://venturebeat.com/2019/03/05/intel-open-sources-cvat-a-toolkit-for-data-labeling/)

**Troubling shooting agreement metrics**

1. **Imbalance data sets, i.e., why my** [**Why is reliability so low when percentage of agreement is high?**](https://www.researchgate.net/post/Why\_is\_reliability\_so\_low\_when\_percentage\_of\_agreement\_is\_high)
2. [**Interpretation of kappa values**](https://towardsdatascience.com/interpretation-of-kappa-values-2acd1ca7b18f)
3. [**Interpreting agreement**](http://web2.cs.columbia.edu/\~julia/courses/CS6998/Interrater\_agreement.Kappa\_statistic.pdf)**, Accuracy precision kappa**

### **CONVOLUTION NEURAL NETS (CNN)**

1. [**Cnn for text**](https://medium.com/@TalPerry/convolutional-methods-for-text-d5260fd5675f) **- tal perry**
2. [**1D CNN using KERAS**](https://blog.goodaudience.com/introduction-to-1d-convolutional-neural-networks-in-keras-for-time-sequences-3a7ff801a2cf)

### **KNOWLEDGE GRAPHS**

1. [**Automatic creation of KG using spacy**](https://towardsdatascience.com/auto-generated-knowledge-graphs-92ca99a81121) **and networx**\
   **Knowledge graphs can be constructed automatically from text using part-of-speech and dependency parsing. The extraction of entity pairs from grammatical patterns is fast and scalable to large amounts of text using NLP library SpaCy.**
2. [**Medium on Reconciling your data and the world of knowledge graphs**](https://towardsdatascience.com/reconciling-your-data-and-the-world-with-knowledge-graphs-bce66b377b14)
3. **Medium Series:**
   1. [**Creating kg**](https://towardsdatascience.com/knowledge-graphs-at-a-glance-c9119130a9f0)
   2. [**Building from structured sources**](https://towardsdatascience.com/building-knowledge-graphs-from-structured-sources-346c56c9d40e)
   3. [**Semantic models**](https://towardsdatascience.com/semantic-models-for-constructing-knowledge-graphs-38c0a1df316a)
4.
5.

### **SUMMARIZATION**

![](https://lh4.googleusercontent.com/eoFe8uZJHAZ8cil1x7TZ-rENzkfkQE3wVr5fHGbeS17h2GlsSMJcFzZ4plUDHd7TN1gsZ6OKKp-WelNVaHmFhOVXxPltjxSN\_USk3s5Ro\_L1Ct-yLiST1q7ST5k5W80CkyHZj7eM)

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
       ****](https://rare-technologies.com/text-summarization-in-python-extractive-vs-abstractive-techniques-revisited/)**Bottom line is that textrank is competitive to sumy\_lex**
    6. [**Sumy**](https://github.com/miso-belica/sumy)
    7. [**Pyteaser**](https://github.com/xiaoxu193/PyTeaser)
    8. [**Pytextrank**](https://github.com/ceteri/pytextrank)
    9. [**Lexrank**](https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume22/erkan04a-html/erkan04a.html)
    10. [**Gensim tutorial on textrank**](https://www.machinelearningplus.com/nlp/gensim-tutorial/)
    11. [**Email summarization**](https://github.com/jatana-research/email-summarization)

###

### **SENTIMENT ANALYSIS**

#### **Databases**

1. [**Sentiment databases**](https://medium.com/@datamonsters/sentiment-analysis-tools-overview-part-1-positive-and-negative-words-databases-ae35431a470c) ****&#x20;
2. **Movie reviews:** [**IMDB reviews dataset on Kaggle**](https://www.kaggle.com/c/word2vec-nlp-tutorial/data)
3. **Sentiwordnet – mapping wordnet senses to a polarity model:** [**SentiWordnet Site**](http://sentiwordnet.isti.cnr.it)
4. [**Twitter airline sentiment on Kaggle**](https://www.kaggle.com/crowdflower/twitter-airline-sentiment)
5. [**First GOP Debate Twitter Sentiment**](https://www.kaggle.com/crowdflower/first-gop-debate-twitter-sentiment)
6. [**Amazon fine foods reviews**](https://www.kaggle.com/snap/amazon-fine-food-reviews)

#### **Tools**

1. **\*\* Many** [**Sentiment tools,** ](https://medium.com/@datamonsters/sentiment-analysis-tools-overview-part-2-7f3a75c262a3)
2. [**NTLK sentiment analyzer**](http://www.nltk.org/api/nltk.sentiment.html)
3. **Vader (NTLK, standalone):**
   1. [**Vader/Sentiwordnet/etc python code examples - possibly good for ensembles**](https://nlpforhackers.io/sentiment-analysis-intro/)
   2. **\*\***[**Intro into Vader**](http://t-redactyl.io/blog/2017/04/using-vader-to-handle-sentiment-analysis-with-social-media-text.html)
   3. [**Why vader?**](https://www.quora.com/Which-is-the-superior-Sentiment-Analyzer-Vader-or-TextBlob)
   4. **\*\***[**Vader - a clear explanation about the paper’s methodology** ](https://www.ijariit.com/manuscripts/v4i1/V4I1-1307.pdf)
   5. **Simple Intro to** [**Vader**](https://medium.com/@aneesha/quick-social-media-sentiment-analysis-with-vader-da44951e4116)
   6. [**A very lengthy and overly complex explanation about using NTLK vader**](https://programminghistorian.org/en/lessons/sentiment-analysis)
   7. [**Vader tutorial, +-0.2 for neutrals.**](https://www.learndatasci.com/tutorials/sentiment-analysis-reddit-headlines-pythons-nltk/)
4. **Text BLob:**
   1. [**Text blob classification**](http://rwet.decontextualize.com/book/textblob/)
   2. [**Python code**](https://planspace.org/20150607-textblob\_sentiment/)
   3. [**More code**](https://textminingonline.com/getting-started-with-textblob)
   4. [**A lengthy tutorial**](https://www.analyticsvidhya.com/blog/2018/02/natural-language-processing-for-beginners-using-textblob/)
   5. **\*\***[**Text blob sentiment analysis tutorial on medium**](https://medium.com/@rahulvaish/textblob-and-sentiment-analysis-python-a687e9fabe96)
   6. [**A lengthy intro plus code about text blob**](https://aparrish.neocities.org/textblob.html)
5. [**Comparative opinion mining a review paper - has some info about unsupervised as well**](https://arxiv.org/pdf/1712.08941.pdf)
6. [**Another reference list, has some unsupervised.**](http://scholar.google.co.il/scholar\_url?url=http://www.nowpublishers.com/article/DownloadSummary/INR-011\&hl=en\&sa=X\&scisig=AAGBfm0NN0Pge4htltclF-D6H4BpxocqwA\&nossl=1\&oi=scholarr)
7. **Sentiwordnet3.0** [**paper**](https://www.researchgate.net/profile/Fabrizio\_Sebastiani/publication/220746537\_SentiWordNet\_30\_An\_Enhanced\_Lexical\_Resource\_for\_Sentiment\_Analysis\_and\_Opinion\_Mining/links/545fbcc40cf27487b450aa21.pdf)
8. [**presentation**](https://web.stanford.edu/class/cs124/lec/sentiment.pdf)
9.  [Hebrew Psychological Lexicons](https://github.com/natalieShapira/HebrewPsychologicalLexicons) by Natalie Shapira

    This is the official code accompanying a paper on the [Hebrew Psychological Lexicons](https://www.aclweb.org/anthology/2021.clpsych-1.6.pdf) was presented at CLPsych 2021.

![Summary Hebrew Psych Lexicon](<../.gitbook/assets/image (10).png>)

**Reference papers:**

1. [**Twitter as a corpus for SA and opinion mining**](http://crowdsourcing-class.org/assignments/downloads/pak-paroubek.pdf)

#### **Ground Truth**&#x20;

1. **For sentiment In Vader -**&#x20;
   1. **“Screening for English language reading comprehension – each rater had to individually score an 80% or higher on a standardized college-level reading comprehension test.**&#x20;
   2. **Complete an online sentiment rating training and orientation session, and score 90% or higher for matching the known (prevalidated) mean sentiment rating of lexical items which included individual words, emoticons, acronyms, sentences, tweets, and text snippets (e.g., sentence segments, or phrases).**&#x20;
   3. **Every batch of 25 features contained five “golden items” with a known (pre-validated) sentiment rating distribution. If a worker was more than one standard deviation away from the mean of this known distribution on three or more of the five golden items, we discarded all 25 ratings in the batch from this worker.**&#x20;
   4. **Bonus to incentivize and reward the highest quality work. Asked workers to select the valence score that they thought “most other people” would choose for the given lexical feature (early/iterative pilot testing revealed that wording the instructions in this manner garnered a much tighter standard deviation without significantly affecting the mean sentiment rating, allowing us to achieve higher quality (generalized) results while being more economical).**&#x20;
   5. **Compensated AMT workers $0.25 for each batch of 25 items they rated, with an additional $0.25 incentive bonus for all workers who successfully matched the group mean (within 1.5 standard deviations) on at least 20 of 25 responses in each batch. Using these four quality control methods, we achieved remarkable value in the data obtained from our AMT workers – we paid incentive bonuses for high quality to at least 90% of raters for most batches.**

![](https://lh3.googleusercontent.com/69nazHo5T9cGMIhgljIDJ4muIjo-fa3PGetGTJwMsktsM699NA2a212TbyqityPup5Q3mVztCO9ieDKSk8y\_qDUrTt4DNsCXkjK0Hg70JLyu-xzdqIQScsuc6Va2M2sH\_Bp0o8Z\_)

[**Multilingual Twitter Sentiment Classification: The Role of Human Annotators**](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0155036)

* **1.6 million tweets labelled**
* **13 languages**
* **Evaluated 6 pretrained classification models**
* **10 CFV**
* **SVM / NB**
* **Annotator agreements.**&#x20;
  * **about 15% were intentionally duplicated to be annotated twice,**
  * **by the same annotator**&#x20;
  * **by two different annotators**&#x20;
* **Self-agreement from multiple annotations of the same annotator**
* **Inter-agreement from multiple annotations by different annotators**&#x20;
* **The confidence intervals for the agreements are estimated by bootstrapping \[**[**12**](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0155036#pone.0155036.ref012)**].**&#x20;
* **It turns out that the self-agreement is a good measure to identify low quality annotators,**&#x20;
* **the inter-annotator agreement provides a good estimate of the objective difficulty of the task, unless it is too low.**

**Alpha was developed to measure the agreement between human annotators, but can also be used to measure the agreement between classification models and a gold standard. It generalizes several specialized agreement measures, takes ordering of classes into account, and accounts for the agreement by chance. Alpha is defined as follows:**&#x20;

![](https://lh4.googleusercontent.com/\_7WwUqxDoCvZwOyBlIUEe0k4IWAq1dlTS\_kgyBiddpOgIbUS-HcArQzOE3gHDurmR0pceyxF71PZU-NsY5Q65fe\_3cFpnak029I3RNnJ\_ofWTGjuHwIIYo-GacTF6bKpNSP50FPP)

[**Method cont here**](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0194317) **in a second paper**&#x20;

###

### **TOPIC MODELING**

****\
****[**A very good article about LSA (TFIDV X SVD), pLSA, LDA, and LDA2VEC.**](https://medium.com/nanonets/topic-modeling-with-lsa-psla-lda-and-lda2vec-555ff65b0b05) **Including code and explanation about dirichlet probability.** [**Lda2vec code**](http://nbviewer.jupyter.org/github/cemoody/lda2vec/blob/master/examples/twenty\_newsgroups/lda2vec/lda2vec.ipynb#)

[**A descriptive comparison for LSA pLSA and LDA**](https://www.reddit.com/r/MachineLearning/comments/10mdtf/lsa\_vs\_plsa\_vs\_lda/)

**A** [**great summation**](https://cs.stanford.edu/\~ppasupat/a9online/1140.html) **about topic modeling, Pros and Cons! (LSA, pLSA, LDA)**

[**Word cloud**](http://keyonvafa.com/inauguration-wordclouds/) **for topic modelling**

[**Sklearn LDA and NMF for topic modelling**](http://scikit-learn.org/stable/auto\_examples/applications/plot\_topics\_extraction\_with\_nmf\_lda.html#sphx-glr-auto-examples-applications-plot-topics-extraction-with-nmf-lda-py)

[**Topic modelling with sentiment per topic according to the data in the topic**](https://www.slideshare.net/jainayush91/topic-modelling-tutorial-on-usage-and-applications)

#### **(TopSBM) topic block modeling**

1. [**Topsbm** ](https://topsbm.github.io)

#### **(LDA) Latent Dirichlet Allocation**&#x20;

**LDA is already taken by the above algorithm!**

[**Latent Dirichlet allocation (LDA) -**](https://algorithmia.com/algorithms/nlp/LDA) **This algorithm takes a group of documents (anything that is made of up text), and returns a number of topics (which are made up of a number of words) most relevant to these documents.**&#x20;

* **LDA is an example of topic modelling**&#x20;
* **?- can this be used for any set of features, not just text?**

[**Medium Article about LDA and**](https://medium.com/ml2vec/topic-modeling-is-an-unsupervised-learning-approach-to-clustering-documents-to-discover-topics-fdfbf30e27df) **NMF (Non-negative Matrix factorization)+ code**

[**Medium article on LDA - a good one with pseudo algorithm and proof**](https://medium.com/@jonathan\_hui/machine-learning-latent-dirichlet-allocation-lda-1d9d148f13a4)****\
****

**In case LDA groups together two topics, we can influence the algorithm in a way that makes those two topics separable -** [**this is called Semi Supervised Guided LDA**](https://medium.freecodecamp.org/how-we-changed-unsupervised-lda-to-semi-supervised-guidedlda-e36a95f3a164)****\
****

* [**LDA tutorials plus code**](https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/)**, used this to build my own classes - using gensim mallet wrapper, doesn't work on pyLDAviz, so use** [**this**](http://jeriwieringa.com/2018/07/17/pyLDAviz-and-Mallet/#comment-4018495276) **to fix it**&#x20;
* [**Introduction to LDA topic modelling, really good,**](http://www.vladsandulescu.com/topic-prediction-lda-user-reviews/) **** [**plus git code**](https://github.com/vladsandulescu/topics)
* [**Sklearn examples using LDA and NMF**](http://scikit-learn.org/stable/auto\_examples/applications/plot\_topics\_extraction\_with\_nmf\_lda.html#sphx-glr-auto-examples-applications-plot-topics-extraction-with-nmf-lda-py)
* [**Tutorial on lda/nmf on medium**](https://medium.com/mlreview/topic-modeling-with-scikit-learn-e80d33668730) **- using tfidf matrix as input!**
* [**Gensim and sklearn LDA variants, comparison**](https://gist.github.com/aronwc/8248457)**,** [**python 3**](https://github.com/EricSchles/sklearn\_gensim\_example/blob/master/example.py)
* [**Medium article on lda/nmf with code**](https://medium.com/mlreview/topic-modeling-with-scikit-learn-e80d33668730)
* **One of the best explanation about** [**Tf-idf vs bow for LDA/NMF**](https://stackoverflow.com/questions/44781047/necessary-to-apply-tf-idf-to-new-documents-in-gensim-lda-model) **- tf for lda, tfidf for nmf, but tfidf can be used for top k selection in lda + visualization,** [**important paper**](http://www.cs.columbia.edu/\~blei/papers/BleiLafferty2009.pdf)
  * [**LDA is a probabilistic**](https://stackoverflow.com/questions/40171208/scikit-learn-should-i-fit-model-with-tf-or-tf-idf) **generative model that generates documents by sampling a topic for each word and then a word from the sampled topic. The generated document is represented as a bag of words.**
  * **NMF is in its general definition the search for 2 matrices W and H such that W\*H=V where V is an observed matrix. The only requirement for those matrices is that all their elements must be non negative.**
  * **From the above definitions it is clear that in LDA only bag of words frequency counts can be used since a vector of reals makes no sense. Did we create a word 1.2 times? On the other hand we can use any non negative representation for NMF and in the example tf-idf is used.**
  * **As far as choosing the number of iterations, for the NMF in scikit learn I don't know the stopping criterion although I believe it is the relative improvement of the loss function being smaller than a threshold so you 'll have to experiment. For LDA I suggest checking manually the improvement of the log likelihood in a held out validation set and stopping when it falls under a threshold.**
  * **The rest of the parameters depend heavily on the data so I suggest, as suggested by @rpd, that you do a parameter search.**
  * **So to sum up, LDA can only generate frequencies and NMF can generate any non negative matrix.**
*

**Very important:**&#x20;

* [**How to measure the variance for LDA and NMF, against PCA.**](https://stackoverflow.com/questions/48148689/how-to-compare-predictive-power-of-pca-and-nmf) **1. Variance score the transformation and inverse transformation of data, test for 1,2,3,4 PCs/LDs/NMs.**
* [**Matching lda mallet performance with gensim and sklearn lda via hyper parameters**](https://groups.google.com/forum/#!topic/gensim/bBHkGogNrfg)

1. [**What is LDA?**](https://www.quora.com/Is-LDA-latent-dirichlet-allocation-unsupervised-or-supervised-learning)
   1. **It is unsupervised natively; it uses joint probability method to find topics(user has to pass # of topics to LDA api). If “Doc X word” is size of input data to LDA, it transforms it to 2 matrices:**
   2. **Doc X topic**
   3. **Word X topic**
   4. **further if you want, you can feed “Doc X topic” matrix to supervised algorithm if labels were given.**
2. **Medium on** [**LDA**](https://medium.com/ml2vec/topic-modeling-is-an-unsupervised-learning-approach-to-clustering-documents-to-discover-topics-fdfbf30e27df)**, explains the random probabilistic nature of LDA**![](https://lh6.googleusercontent.com/-16nr83feu9UQzaIoi4CIMYwSHRhH99p49scg\_Mnk9PH7EmMh-Q6410FLxPtwZCapOrKkq3J9MK7njHPD21o1TYxZYZopSHAoWCKFuwCMU8Rcy0kLIacqWcPqtETr8ZuTaxN6BLn)
3. **Machinelearningplus on** [**LDA in sklearn**](https://www.machinelearningplus.com/nlp/topic-modeling-python-sklearn-examples/) **- a great read, dont forget to read the** [**mallet**](https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/) **article.**
4. **Medium on** [**LSA pLSA, LDA LDA2vec**](https://medium.com/nanonets/topic-modeling-with-lsa-psla-lda-and-lda2vec-555ff65b0b05)**, high level theoretical - not clear**
5. [**Medium on LSI vs LDA vs HDP, HDP wins..**](https://medium.com/square-corner-blog/topic-modeling-optimizing-for-human-interpretability-48a81f6ce0ed)
6. **Medium on** [**LDA**](https://medium.com/@samsachedina/effective-data-science-latent-dirichlet-allocation-a109742f7d1c)**, some historical reference and general high level how to use exapmles.**
7. [**Incredibly useful response**](https://www.quora.com/What-are-good-ways-of-evaluating-the-topics-generated-by-running-LDA-on-a-corpus) **on LDA grid search params and about LDA expectations. Must read.**
8. [**Lda vs pLSA**](https://stats.stackexchange.com/questions/155860/latent-dirichlet-allocation-vs-plsa)**, talks about the sampling from a distribution of distributions in LDA**
9. [**BLog post on topic modelling**](http://mcburton.net/blog/joy-of-tm/) **- has some text about overfitting - undiscussed in many places.**
10. [**Perplexity vs coherence on held out unseen dat**](https://stats.stackexchange.com/questions/182010/when-is-it-ok-to-not-use-a-held-out-set-for-topic-model-evaluation)**a, not okay and okay, respectively. Due to how we measure the metrics, ie., read the formulas.** [**Also this**](https://transacl.org/ojs/index.php/tacl/article/view/582/158) **and** [**this**](https://stackoverflow.com/questions/11162402/lda-topic-modeling-training-and-testing)
11. **LDA as** [**dimentionality reduction** ](https://stackoverflow.com/questions/46504688/lda-as-the-dimension-reduction-before-or-after-partitioning)
12. [**LDA on alpha and beta to control density of topics**](https://stats.stackexchange.com/questions/364494/lda-and-test-data-perplexity)
13. **Jupyter notebook on** [**hacknews LDA topic modelling**](http://nbviewer.jupyter.org/github/bmabey/hacker\_news\_topic\_modelling/blob/master/HN%20Topic%20Model%20Talk.ipynb#topic=55\&lambda=1\&term=) **- missing code?**
14. [**Jupyter notebook**](http://nbviewer.jupyter.org/github/dolaameng/tutorials/blob/master/topic-finding-for-short-texts/topics\_for\_short\_texts.ipynb) **for kmeans, lda, svd,nmf comparison - advice is to keep nmf or other as a baseline to measure against LDA.**
15. [**Gensim on LDA**](https://rare-technologies.com/what-is-topic-coherence/) **with** [**code** ](https://nbviewer.jupyter.org/github/dsquareindia/gensim/blob/280375fe14adea67ce6384ba7eabf362b05e6029/docs/notebooks/topic\_coherence\_tutorial.ipynb)
16. [**Medium on lda with sklearn**](https://medium.com/mlreview/topic-modeling-with-scikit-learn-e80d33668730)
17. **Selecting the number of topics in LDA,** [**blog 1**](https://cran.r-project.org/web/packages/ldatuning/vignettes/topics.html)**,** [**blog2**](http://www.rpubs.com/MNidhi/NumberoftopicsLDA)**,** [**using preplexity**](https://stackoverflow.com/questions/21355156/topic-models-cross-validation-with-loglikelihood-or-perplexity)**,** [**prep and aic bic**](https://stats.stackexchange.com/questions/322809/inferring-the-number-of-topics-for-gensims-lda-perplexity-cm-aic-and-bic)**,** [**coherence**](https://stackoverflow.com/questions/17421887/how-to-determine-the-number-of-topics-for-lda)**,** [**coherence2**](https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/#17howtofindtheoptimalnumberoftopicsforlda)**,** [**coherence 3 with tutorial**](https://datascienceplus.com/evaluation-of-topic-modeling-topic-coherence/)**, un**[**clear**](https://community.rapidminer.com/discussion/51283/what-is-the-best-number-of-topics-on-lda)**,** [**unclear with analysis of stopword % inclusion**](https://markhneedham.com/blog/2015/03/24/topic-modelling-working-out-the-optimal-number-of-topics/)**,** [**unread**](https://www.quora.com/What-are-the-best-ways-of-selecting-number-of-topics-in-LDA)**,** [**paper: heuristic approach**](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4597325/)**,** [**elbow method**](https://www.knime.com/blog/topic-extraction-optimizing-the-number-of-topics-with-the-elbow-method)**,** [**using cv**](http://freerangestats.info/blog/2017/01/05/topic-model-cv)**,** [**Paper: new stability metric**](https://github.com/derekgreene/topic-stability) **+ gh code,**&#x20;
18. [**Selecting the top K words in LDA**](https://stats.stackexchange.com/questions/199263/choosing-words-in-a-topic-which-cut-off-for-lda-topics)
19. [**Presentation: best practices for LDA**](http://www.phusewiki.org/wiki/images/c/c9/Weizhong\_Presentation\_CDER\_Nov\_9th.pdf)
20. [**Medium on guidedLDA**](https://medium.freecodecamp.org/how-we-changed-unsupervised-lda-to-semi-supervised-guidedlda-e36a95f3a164) **- switching from LDA to a variation of it that is guided by the researcher / data**&#x20;
21. **Medium on lda -** [**another introductory**](https://towardsdatascience.com/thats-mental-using-lda-topic-modeling-to-investigate-the-discourse-on-mental-health-over-time-11da252259c3)**,** [**la times**](https://medium.com/swiftworld/topic-modeling-of-new-york-times-articles-11688837d32f)
22. [**Topic modelling through time**](https://tedunderwood.com/category/methodology/topic-modeling/)
23. [**Mallet vs nltk**](https://stackoverflow.com/questions/7476180/topic-modelling-in-mallet-vs-nltk)**,** [**params**](https://github.com/RaRe-Technologies/gensim/issues/193)**,** [**params**](https://groups.google.com/forum/#!topic/gensim/tOoc1Q0Ump0)
24. [**Paper: improving feature models**](http://aclweb.org/anthology/Q15-1022)
25. [**Lda vs w2v (doesn't make sense to compare**](https://stats.stackexchange.com/questions/145485/lda-vs-word2vec/145488)**,** [**again here**](https://stats.stackexchange.com/questions/145485/lda-vs-word2vec)
26. [**Adding lda features to w2v for classification**](https://stackoverflow.com/questions/48140319/add-lda-topic-modelling-features-to-word2vec-sentiment-classification)
27. [**Spacy and gensim on 20 news groups**](https://www.shanelynn.ie/word-embeddings-in-python-with-spacy-and-gensim/)
28. **The best topic modelling explanation including** [**Usages**](https://nlpforhackers.io/topic-modeling/)**, insights,  a great read, with code  - shows how to find similar docs by topic in gensim, and shows how to transform unseen documents and do similarity using sklearn:**&#x20;
    1. **Text classification – Topic modeling can improve classification by grouping similar words together in topics rather than using each word as a feature**
    2. **Recommender Systems – Using a similarity measure we can build recommender systems. If our system would recommend articles for readers, it will recommend articles with a topic structure similar to the articles the user has already read.**
    3. **Uncovering Themes in Texts – Useful for detecting trends in online publications for example**
    4. **A Form of Tagging - If document classification is assigning a single category to a text, topic modeling is assigning multiple tags to a text. A human expert can label the resulting topics with human-readable labels and use different heuristics to convert the weighted topics to a set of tags.**
    5. [**Topic Modelling for Feature Selection**](https://www.analyticsvidhya.com/blog/2016/08/beginners-guide-to-topic-modeling-in-python/) **- Sometimes LDA can also be used as feature selection technique. Take an example of text classification problem where the training data contain category wise documents. If LDA is running on sets of category wise documents. Followed by removing common topic terms across the results of different categories will give the best features for a category.**
    6.

**How to interpret topics using pyldaviz:**

**Let’s interpret the topic visualization. Notice how topics are shown on the left while words are on the right. Here are the main things you should consider:**

1. **Larger topics are more frequent in the corpus.**
2. **Topics closer together are more similar, topics further apart are less similar.**
3. **When you select a topic, you can see the most representative words for the selected topic. This measure can be a combination of how frequent or how discriminant the word is. You can adjust the weight of each property using the slider.**
4. **Hovering over a word will adjust the topic sizes according to how representative the word is for the topic.**

**\*\*\*\***[**pyLDAviz paper\*\*\*!**](https://cran.r-project.org/web/packages/LDAvis/vignettes/details.pdf)****\
****

[**pyLDAviz - what am i looking at ?**](https://github.com/explosion/spacy-notebooks/blob/master/notebooks/conference\_notebooks/modern\_nlp\_in\_python.ipynb) **by spacy**

**There are a lot of moving parts in the visualization. Here's a brief summary:**

* **On the left, there is a plot of the "distance" between all of the topics (labeled as the Intertopic Distance Map)**
  * **The plot is rendered in two dimensions according a** [**multidimensional scaling (MDS)**](https://en.wikipedia.org/wiki/Multidimensional\_scaling) **algorithm. Topics that are generally similar should be appear close together on the plot, while dissimilar topics should appear far apart.**
  * **The relative size of a topic's circle in the plot corresponds to the relative frequency of the topic in the corpus.**
  * **An individual topic may be selected for closer scrutiny by clicking on its circle, or entering its number in the "selected topic" box in the upper-left.**
* **On the right, there is a bar chart showing top terms.**
  * **When no topic is selected in the plot on the left, the bar chart shows the top-30 most "salient" terms in the corpus. A term's saliency is a measure of both how frequent the term is in the corpus and how "distinctive" it is in distinguishing between different topics.**
  * **When a particular topic is selected, the bar chart changes to show the top-30 most "relevant" terms for the selected topic. The relevance metric is controlled by the parameter λλ, which can be adjusted with a slider above the bar chart.**
    * **Setting the λλ parameter close to 1.0 (the default) will rank the terms solely according to their probability within the topic.**
    * **Setting λλ close to 0.0 will rank the terms solely according to their "distinctiveness" or "exclusivity" within the topic — i.e., terms that occur only in this topic, and do not occur in other topics.**
    * **Setting λλ to values between 0.0 and 1.0 will result in an intermediate ranking, weighting term probability and exclusivity accordingly.**
* **Rolling the mouse over a term in the bar chart on the right will cause the topic circles to resize in the plot on the left, to show the strength of the relationship between the topics and the selected term.**

**A more detailed explanation of the pyLDAvis visualization can be found** [**here**](https://cran.r-project.org/web/packages/LDAvis/vignettes/details.pdf)**. Unfortunately, though the data used by gensim and pyLDAvis are the same, they don't use the same ID numbers for topics. If you need to match up topics in gensim's LdaMulticore object and pyLDAvis' visualization, you have to dig through the terms manually.**\
****

1. [**Another great article about LDA**](https://www.analyticsvidhya.com/blog/2016/08/beginners-guide-to-topic-modeling-in-python/)**, including algorithm, parameters!! And**

**Parameters of LDA**

1. **Alpha and Beta Hyperparameters – alpha represents document-topic density and Beta represents topic-word density. Higher the value of alpha, documents are composed of more topics and lower the value of alpha, documents contain fewer topics. On the other hand, higher the beta, topics are composed of a large number of words in the corpus, and with the lower value of beta, they are composed of few words.**
2. **Number of Topics – Number of topics to be extracted from the corpus. Researchers have developed approaches to obtain an optimal number of topics by using Kullback Leibler Divergence Score. I will not discuss this in detail, as it is too mathematical. For understanding, one can refer to this\[1] original paper on the use of KL divergence.**
3. **Number of Topic Terms – Number of terms composed in a single topic. It is generally decided according to the requirement. If the problem statement talks about extracting themes or concepts, it is recommended to choose a higher number, if problem statement talks about extracting features or terms, a low number is recommended.**
4. **Number of Iterations / passes – Maximum number of iterations allowed to LDA algorithm for convergence.**

**Ways to improve LDA:**

1. **Reduce dimentionality of document-term matrix**
2. **Frequency filter**
3. **POS filter**
4. **Batch wise LDA**
5. [**History of LDA**](http://qpleple.com/bib/#Newman10a) **- by the frech guy**
6. [**Diff between lda and mallet**](https://groups.google.com/forum/#!topic/gensim/\_VO4otCV6cU) **- The inference algorithms in Mallet and Gensim are indeed different. Mallet uses Gibbs Sampling which is more precise than Gensim's faster and online Variational Bayes. There is a way to get relatively performance by increasing number of passes.**
7. [**Mallet in gensim blog post**](https://rare-technologies.com/tutorial-on-mallet-in-python/)
8. **Alpha beta in mallet:** [**contribution**](https://datascience.stackexchange.com/questions/199/what-does-the-alpha-and-beta-hyperparameters-contribute-to-in-latent-dirichlet-a)
   1. [**The default for alpha is 5.**](https://stackoverflow.com/questions/44561609/how-does-mallet-set-its-default-hyperparameters-for-lda-i-e-alpha-and-beta)**0 divided by the number of topics. You can think of this as five "pseudo-words" of weight on the uniform distribution over topics. If the document is short, we expect to stay closer to the uniform prior. If the document is long, we would feel more confident moving away from the prior.**
   2. **With hyperparameter optimization, the alpha value for each topic can be different. They usually become smaller than the default setting.**
   3. **The default value for beta is 0.01. This means that each topic has a weight on the uniform prior equal to the size of the vocabulary divided by 100. This seems to be a good value. With optimization turned on, the value rarely changes by more than a factor of two.**
9. [**Multilingual - alpha is divided by topic count, reaffirms 7**](http://mallet.cs.umass.edu/topics-polylingual.php)
10. [**Topic modelling with lda and nmf on medium**](https://medium.com/ml2vec/topic-modeling-is-an-unsupervised-learning-approach-to-clustering-documents-to-discover-topics-fdfbf30e27df) **- has a very good simple example with probabilities**
11. **Code:** [**great for top docs, terms, topics etc.**](http://nbviewer.jupyter.org/github/bmabey/hacker\_news\_topic\_modelling/blob/master/HN%20Topic%20Model%20Talk.ipynb#topic=55\&lambda=1\&term=)
12. **Great article:** [**Many ways of evaluating topics by running LDA**](https://www.quora.com/What-are-good-ways-of-evaluating-the-topics-generated-by-running-LDA-on-a-corpus)
13. [**Youtube on LDAvis explained**](http://stat-graphics.org/movies/ldavis.html)
14. **Presentation:** [**More visualization options including ldavis**](https://speakerdeck.com/bmabey/visualizing-topic-models?slide=17)
15. [**A pointer to the ldaviz fix**](https://github.com/RaRe-Technologies/gensim/issues/2069) **->** [**fix**](http://jeriwieringa.com/2018/07/17/pyLDAviz-and-Mallet/#comment-4018495276)**,** [**git code**](https://github.com/jerielizabeth/Gospel-of-Health-Notebooks/blob/master/blogPosts/pyLDAvis\_and\_Mallet.ipynb)
16. [**Difference between lda in gensim and sklearn a post on rare**](https://github.com/RaRe-Technologies/gensim/issues/457)
17. [**The best code article on LDA/MALLET**](https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/)**, and using** [**sklearn**](https://www.machinelearningplus.com/nlp/topic-modeling-python-sklearn-examples/) **(using clustering for getting group of sentences in each topic)**
18. [**LDA in gensim, a tutorial by gensim**](https://nbviewer.jupyter.org/github/rare-technologies/gensim/blob/develop/docs/notebooks/atmodel\_tutorial.ipynb)
19. &#x20;**** [**Lda on medium**](https://towardsdatascience.com/topic-modelling-in-python-with-nltk-and-gensim-4ef03213cd21) ****&#x20;
20. &#x20;**** [**What are the pros and cons of LDA and NMF in topic modeling? Under what situations should we choose LDA or NMF? Is there comparison of two techniques in topic modeling?**](https://www.quora.com/What-are-the-pros-and-cons-of-LDA-and-NMF-in-topic-modeling-Under-what-situations-should-we-choose-LDA-or-NMF-Is-there-comparison-of-two-techniques-in-topic-modeling)
21. [**What is the difference between NMF and LDA? Why are the priors of LDA sparse-induced?**](https://www.quora.com/What-is-the-difference-between-NMF-and-LDA-Why-are-the-priors-of-LDA-sparse-induced)
22. [**Exploring Topic Coherence over many models and many topics**](http://aclweb.org/anthology/D/D12/D12-1087.pdf) **lda nmf svd, using umass and uci coherence measures**
23. **\*\*\*** [**Practical topic findings for short sentence text**](http://nbviewer.jupyter.org/github/dolaameng/tutorials/blob/master/topic-finding-for-short-texts/topics\_for\_short\_texts.ipynb) **code**
24. [**What's the difference between SVD/NMF and LDA as topic model algorithms essentially? Deterministic vs prob based**](https://www.quora.com/Whats-the-difference-between-SVD-NMF-and-LDA-as-topic-model-algorithms-essentially)
25. [**What is the difference between NMF and LDA? Why are the priors of LDA sparse-induced?**](https://www.quora.com/What-is-the-difference-between-NMF-and-LDA-Why-are-the-priors-of-LDA-sparse-induced)
26. [**What are the relationships among NMF, tensor factorization, deep learning, topic modeling, etc.?**](https://www.quora.com/What-are-the-relationships-among-NMF-tensor-factorization-deep-learning-topic-modeling-etc)
27. [**Code: lda nmf**](https://www.kaggle.com/rchawla8/topic-modeling-with-lda-and-nmf-algorithms)
28. [**Unread a comparison of lda and nmf**](https://wiki.ubc.ca/Course:CPSC522/A\_Comparison\_of\_LDA\_and\_NMF\_for\_Topic\_Modeling\_on\_Literary\_Themes)
29. [**Presentation: lda sparse coding matrix factorization**](https://www.cs.cmu.edu/\~epxing/Class/10708-15/slides/LDA\_SC.pdf)
30. [**An experimental comparison between NMF and LDA for active cross-situational object-word learning**](https://ieeexplore.ieee.org/abstract/document/7846822)
31. [**Topic coherence in gensom with jupyter code**](https://markroxor.github.io/gensim/static/notebooks/topic\_coherence\_tutorial.html)
32. [**Topic modelling dynamic presentation**](http://chdoig.github.io/pygotham-topic-modeling/#/)
33. **Paper:** [**Topic modelling and event identification from twitter data**](https://arxiv.org/ftp/arxiv/papers/1608/1608.02519.pdf)**, says LDA vs NMI (NMF?) and using coherence to analyze**
34. [**Just another medium article about ™**](https://medium.com/square-corner-blog/topic-modeling-optimizing-for-human-interpretability-48a81f6ce0ed)
35. [**What is Wrong with Topic Modeling? (and How to Fix it Using Search-based SE)**](https://www.researchgate.net/publication/307303102\_What\_is\_Wrong\_with\_Topic\_Modeling\_and\_How\_to\_Fix\_it\_Using\_Search-based\_SE) **LDADE's tunings dramatically reduces topic instability.**&#x20;
36. [**Talk about topic modelling**](https://tedunderwood.com/category/methodology/topic-modeling/)
37. [**Intro to topic modelling**](http://blog.echen.me/2011/08/22/introduction-to-latent-dirichlet-allocation/)
38. [**Detecting topics in twitter**](https://github.com/heerme/twitter-topics) **github code**
39. [**Another topic model tutorial**](https://github.com/derekgreene/topic-model-tutorial/blob/master/2%20-%20NMF%20Topic%20Models.ipynb)
40. **(didnt read) NTM -** [**neural topic modeling using embedded spaces**](https://github.com/elbamos/NeuralTopicModels) **with github code**
41. [**Another lda tutorial**](https://blog.intenthq.com/blog/automatic-topic-modelling-with-latent-dirichlet-allocation)
42. [**Comparing tweets using lda**](https://ink.library.smu.edu.sg/cgi/viewcontent.cgi?article=2374\&context=sis\_research)
43. [**Lda and w2v as features for some classification task**](https://www.kaggle.com/vukglisovic/classification-combining-lda-and-word2vec)
44. [**Improving ™ with embeddings**](https://github.com/datquocnguyen/LFTM)
45. [**w2v/doc2v for topic clustering - need to see the code to understand how they got clean topics, i assume a human rewrote it**](https://towardsdatascience.com/automatic-topic-clustering-using-doc2vec-e1cea88449c)
46.

**Topic coherence (lda/nmf)**

1. [**What is?**](https://www.quora.com/What-is-topic-coherence)**,** [**Wiki on pmi**](https://en.wikipedia.org/wiki/Pointwise\_mutual\_information#cite\_note-Church1990-1)
2. [**Datacamp on coherence metrics, a comparison, read me.**](https://datascienceplus.com/evaluation-of-topic-modeling-topic-coherence/)
3. **Paper:** [**explains what is coherence**](http://aclweb.org/anthology/J90-1003)

![](https://lh4.googleusercontent.com/Jw5TMIwMSsVYMPRQxe5ZWKC3IDdj8KBAhd4y7nr5nLQZsxdhzDFM8gUVXjVnfZnoqfX-G1t2JjrpxKz2-IyO4WU5VTIOHUJgavudWCaaA18j7bbOf\_nUpewy874W-a9SyaOWDSfQ)

1. [**Umass vs C\_v, what are the diff?** ](https://groups.google.com/forum/#!topic/gensim/CsscFah0Ax8)
2. **Paper: umass, uci, nmpi, cv, cp etv** [**Exploring the Space of Topic Coherence Measures**](http://svn.aksw.org/papers/2015/WSDM\_Topic\_Evaluation/public.pdf)
3. **Paper:** [**Automatic evaluation of topic coherence**](https://mimno.infosci.cornell.edu/info6150/readings/N10-1012.pdf) ****&#x20;
4. **Paper:** [**exploring the space of topic coherence methods**](https://dl.acm.org/citation.cfm?id=2685324)
5. **Paper:** [**Relation between mutial information / entropy and pmi**](https://svn.spraakdata.gu.se/repos/gerlof/pub/www/Docs/npmi-pfd.pdf)
6. **Stackexchange:** [**coherence / pmi how to calc**](https://stats.stackexchange.com/questions/158790/topic-similarity-semantic-pmi-between-two-words-wikipedia)
7. **Paper:** [**Machine Reading Tea Leaves: Automatically Evaluating Topic Coherence and Topic Model Quality**](http://www.aclweb.org/anthology/E14-1056) **- perplexity needs unseen data, coherence doesnt**
8. [**Evaluation of topic modelling techniques for twitter**](https://www.cs.toronto.edu/\~jstolee/projects/topic.pdf) **lda lda-u btm w2vgmm**
9. **Paper:** [**Topic coherence measures**](https://svn.aksw.org/papers/2015/WSDM\_Topic\_Evaluation/public.pdf)
10. [**topic modelling from different domains**](http://proceedings.mlr.press/v32/chenf14.pdf)
11. **Paper:** [**Optimizing Semantic Coherence in Topic Models**](https://mimno.infosci.cornell.edu/papers/mimno-semantic-emnlp.pdf)
12. **Paper:** [**L-EnsNMF: Boosted Local Topic Discovery via Ensemble of Nonnegative Matrix Factorization** ](http://www.joonseok.net/papers/lensnmf.pdf)
13. **Paper:** [**Content matching between TV shows and advertisements through Latent Dirichlet Allocation** ](http://arno.uvt.nl/show.cgi?fid=145381)
14. **Paper:** [**Full-Text or Abstract? Examining Topic Coherence Scores Using Latent Dirichlet Allocation**](http://www.saf21.eu/wp-content/uploads/2017/09/5004a165.pdf)
15. **Paper:** [**Evaluating topic coherence**](https://pdfs.semanticscholar.org/03a0/62fdcd13c9287a2d4e1d6d057fd2e083281c.pdf) **- Abstract: Topic models extract representative word sets—called topics—from word counts in documents without requiring any semantic annotations. Topics are not guaranteed to be well interpretable, therefore, coherence measures have been proposed to distinguish between good and bad topics. Studies of topic coherence so far are limited to measures that score pairs of individual words. For the first time, we include coherence measures from scientific philosophy that score pairs of more complex word subsets and apply them to topic scoring.**

**Conclusion: The results of the first experiment show that if we are using the one-any, any-any and one-all coherences directly for optimization they are leading to meaningful word sets. The second experiment shows that these coherence measures are able to outperform the UCI coherence as well as the UMass coherence on these generated word sets. For evaluating LDA topics any-any and one-any coherences perform slightly better than the UCI coherence. The correlation of the UMass coherence and the human ratings is not as high as for the other coherences.**

1. **Code:** [**Evaluating topic coherence, using gensim umass or cv parameter**](https://datascienceplus.com/evaluation-of-topic-modeling-topic-coherence/) **- To conclude, there are many other approaches to evaluate Topic models such as Perplexity, but its poor indicator of the quality of the topics.Topic Visualization is also a good way to assess topic models. Topic Coherence measure is a good way to compare difference topic models based on their human-interpretability.The u\_mass and c\_v topic coherences capture the optimal number of topics by giving the interpretability of these topics a number called coherence score.**
2. **Formulas:** [**UCI vs UMASS**\
   ****](http://qpleple.com/topic-coherence-to-evaluate-topic-models/)![](https://lh6.googleusercontent.com/aWrfeNX1FDBZYrIxAUSFw2ZcRQXyHTuxZ\_rgRXBhMPjvMY0sCQx-OlFKBRgId3Eynhv2532ZA5FWxB3Jz4Y8rjfAg5lnjwfxhRcmqfNq7d9rYrxWZrp146xarFHL6OkLSIVXPLEe)
3. [**Inferring the number of topics for gensim's LDA - perplexity, CM, AIC, and BIC**](https://stats.stackexchange.com/questions/322809/inferring-the-number-of-topics-for-gensims-lda-perplexity-cm-aic-and-bic)
4. [**Perplexity as a measure for LDA**](https://groups.google.com/forum/#!topic/gensim/tgJLVulf5xQ)
5. [**Finding number of topics using perplexity**](https://groups.google.com/forum/#!topic/gensim/TpuYRxhyIOc)
6. [**Coherence for tweets**](http://terrierteam.dcs.gla.ac.uk/publications/fang\_sigir\_2016\_examine.pdf)
7. **Presentation** [**Twitter DLA**](https://www.slideshare.net/akshayubhat/twitter-lda)**,** [**tweet pooling improvements**](http://users.cecs.anu.edu.au/\~ssanner/Papers/sigir13.pdf)**,** [**hierarchical summarization of tweets**](https://www.researchgate.net/publication/322359369\_Hierarchical\_Summarization\_of\_News\_Tweets\_with\_Twitter-LDA)**,** [**twitter LDA in java**](https://sites.google.com/site/lyangwww/code-data) **** [**on github**](https://github.com/minghui/Twitter-LDA)****\
   **Papers:** [**TM of twitter timeline**](https://medium.com/@alexisperrier/topic-modeling-of-twitter-timelines-in-python-bb91fa90d98d)**,** [**in twitter aggregation by conversatoin**](https://www.aaai.org/ocs/index.php/ICWSM/ICWSM16/paper/download/13162/12778)**,** [**twitter topics using LDA**](http://uu.diva-portal.org/smash/get/diva2:904196/FULLTEXT01.pdf)**,** [**empirical study**](https://snap.stanford.edu/soma2010/papers/soma2010\_12.pdf) **,**&#x20;

#### **COHERENCE**&#x20;

* [**Using regularization to improve PMI score and in turn coherence for LDA topics**](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.230.7738\&rep=rep1\&type=pdf)
* [**Improving model precision - coherence using turkers for LDA**](https://pdfs.semanticscholar.org/1d29/f7a9e3135bba0339b9d70ecbda9d106b01d2.pdf)
* [**Gensim**](https://radimrehurek.com/gensim/models/coherencemodel.html) **-** [ **paper about their algorithm and PMI/UCI etc.**](http://svn.aksw.org/papers/2015/WSDM\_Topic\_Evaluation/public.pdf)
* [**Advice for coherence,**](https://gist.github.com/dsquareindia/ac9d3bf57579d02302f9655db8dfdd55) **then** [**Good vs bad model (50 vs 1 iterations) measuring u\_mass coherence**](https://markroxor.github.io/gensim/static/notebooks/topic\_coherence\_tutorial.html) **-** [**2nd code**](https://gist.github.com/dsquareindia/ac9d3bf57579d02302f9655db8dfdd55) **- “In your data we can see that there is a peak between 0-100 and a peak between 400-500. What I would think in this case is that "does \~480 topics make sense for the kind of data I have?" If not, you can just do an np.argmax for 0-100 topics and trade-off coherence score for simpler understanding. Otherwise just do an np.argmax on the full set.”**
* [**Diff term weighting schemas for topic modelling, code plus paper**](https://github.com/cipriantruica/TM\_TESTS)
* [**Workaround for pyLDAvis using LDA-Mallet**](http://jeriwieringa.com/2018/07/17/pyLDAviz-and-Mallet/#comment-4018495276)
* [**pyLDAvis paper**](http://www.aclweb.org/anthology/W14-3110)
* [**Visualizing LDA topics results** ](https://de.dariah.eu/tatom/topic\_model\_visualization.html)
* [**Visualizing trends, topics, sentiment, heat maps, entities**](https://github.com/Lissy93/twitter-sentiment-visualisation) **- really good**
* **Topic stability Metric, a novel method, compared against jaccard, spearman, silhouette.:** [**Measuring LDA Topic Stability from Clusters of Replicated Runs**](https://arxiv.org/pdf/1808.08098.pdf)

#### **Non-negative Matrix factorization (NMF)**

[**Medium Article about LDA and**](https://medium.com/ml2vec/topic-modeling-is-an-unsupervised-learning-approach-to-clustering-documents-to-discover-topics-fdfbf30e27df) **NMF (Non-negative Matrix factorization)+ code**\
****

#### **LDA2VEC**

1. **“if you want to rework your own topic models that, say, jointly correlate an article’s topics with votes or predict topics over users then you might be interested in** [**lda2vec**](https://github.com/cemoody/lda2vec)**.”**
2. [**Datacamp intro**](https://www.datacamp.com/community/tutorials/lda2vec-topic-model)
3. [**Original blog**](https://multithreaded.stitchfix.com/blog/2016/05/27/lda2vec/#topic=38\&lambda=1\&term=) **- I just learned about these papers which are quite similar:** [**Gaussian LDA for Topic Word Embeddings**](http://www.aclweb.org/anthology/P15-1077) **and** [**Nonparametric Spherical Topic Modeling with Word Embeddings**](http://arxiv.org/abs/1604.00126)**.**
4. [**Moody’s Slide Share**](https://www.slideshare.net/ChristopherMoody3/word2vec-lda-and-introducing-a-new-hybrid-algorithm-lda2vec-57135994) **(excellent read)**
5. [**Docs**](http://lda2vec.readthedocs.io/en/latest/?badge=latest)
6. [**Original Git**](https://github.com/cemoody/lda2vec) **+** [**Excellent notebook example**](http://nbviewer.jupyter.org/github/cemoody/lda2vec/blob/master/examples/twenty\_newsgroups/lda2vec/lda2vec.ipynb#topic=0\&lambda=1\&term=)
7. [**Tf implementation**](https://github.com/meereeum/lda2vec-tf)**,** [**another more recent one tf 1.5**](https://github.com/nateraw/Lda2vec-Tensorflow)
8. [**Another blog explaining about lda etc**](https://datawarrior.wordpress.com/tag/lda2vec/)**,** [**post**](https://datawarrior.wordpress.com/2016/02/15/lda2vec-a-hybrid-of-lda-and-word2vec/)**,** [**post**](https://datawarrior.wordpress.com/2016/04/20/local-and-global-words-and-topics/)
9. [**Lda2vec in tf**](https://github.com/meereeum/lda2vec-tf)**,** [**tf 1.5**](https://github.com/nateraw/Lda2vec-Tensorflow)**,**&#x20;
10. [**Comparing lda2vec to lda**](https://medium.com/scaleabout/a-gentle-introduction-to-doc2vec-db3e8c0cce5e)
11. **Youtube:** [**lda/doc2vec with pca examples**](https://www.youtube.com/watch?v=i3Opb3-QNX4)
12. [**Example on gh**](https://github.com/BoPengGit/LDA-Doc2Vec-example-with-PCA-LDAvis-visualization/blob/master/Doc2Vec/Doc2Vec2.py) **on jupyter**

#### **TOP2VEC**

1. [**Git**](https://github.com/ddangelov/Top2Vec)**,** [**paper**](https://arxiv.org/pdf/2008.09470.pdf)
2. **Topic modeling with distillibert** [**on medium**](https://towardsdatascience.com/topic-modeling-with-bert-779f7db187e6)**,** [**bertTopic**](https://towardsdatascience.com/interactive-topic-modeling-with-bertopic-1ea55e7d73d8)**!, c-tfidf, umap, hdbscan, merging similar topics, visualization,** [**berTopic (same method as the above)**](https://github.com/MaartenGr/BERTopic)
3. [**Medium with the same general method**](https://towardsdatascience.com/topic-modeling-with-bert-779f7db187e6)

### **NAMED ENTITY RECOGNITION (NER)**

* [**State of the art LSTM architectures using NN**](https://blog.paralleldots.com/data-science/named-entity-recognition-milestone-models-papers-and-technologies/)
* **Medium:** [**Ner free datasets**](https://towardsdatascience.com/deep-learning-for-ner-1-public-datasets-and-annotation-methods-8b1ad5e98caf) **and** [**bilstm implementation**](https://towardsdatascience.com/deep-learning-for-named-entity-recognition-2-implementing-the-state-of-the-art-bidirectional-lstm-4603491087f1) **using glove embeddings**

**Easy to implement in keras! They are based on the following** [**paper**](https://arxiv.org/abs/1511.08308)

* [**Medium**](https://medium.com/district-data-labs/named-entity-recognition-and-classification-for-entity-extraction-6f23342aa7c5)**: NLTK entities, polyglot entities, sner entities, finally an ensemble method wins all!**

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
* [**SNER demo - capital letters matter, a minimum of one.**](http://nlp.stanford.edu:8080/ner/process) ****&#x20;
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

### **SEARCH**

1. **Bert** [**search engine**](https://towardsdatascience.com/covid-19-bert-literature-search-engine-4d06cdac08bd)**, cosine between paragraphs and question.**
2. **Semantic search, autom completion, filtering, augmentation, scoring. Problems: Token matching, contextualization, query misunderstanding, image search, metric. Solutions: synonym generation, query autocompletion, alternate query generation, word and doc embedding, contextualization, ranking, ensemble, multilingual search**

### **SEQ2SEQ SEQUENCE TO SEQUENCE**

1. [**Keras blog**](https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html) **- char-level, token-using embedding layer, teacher forcing**
2. [**Teacher forcing explained**](https://towardsdatascience.com/what-is-teacher-forcing-3da6217fed1c)
3. [**Same as keras but with token-level**](https://towardsdatascience.com/machine-translation-with-the-seq2seq-model-different-approaches-f078081aaa37)
4. [**Medium on char, word, byte-level**](https://medium.com/@petepeeradejtanruangporn/experimenting-with-neural-machine-translation-for-thai-1681fd2b375a)
5. [**Mastery on enc-dec using the keras method**](https://machinelearningmastery.com/develop-encoder-decoder-model-sequence-sequence-prediction-keras/)**, and on** [**neural translation**](https://machinelearningmastery.com/define-encoder-decoder-sequence-sequence-model-neural-machine-translation-keras/)
6. [**Machine translation git from eng to jap**](https://github.com/samurainote/seq2seq\_translate\_slackbot/blob/master/seq2seq\_translate.py)**,** [**another**](https://github.com/samurainote/seq2seq\_translate\_slackbot)**, and its** [**medium**](https://towardsdatascience.com/how-to-implement-seq2seq-lstm-model-in-keras-shortcutnlp-6f355f3e5639)

![](https://lh6.googleusercontent.com/bcrIRzPLlcnQBl1zWR2s0\_tB-NNEQxd8ZNQK8oK2NJsc29Fv6RdfKynfjHeNsSvl5d0SqK55k8xN1NAIrvEcnFEtpfZCfOHZzCSFLKmxeBWXn903VOJKiKTMV4Ynm\_HL6Sgls2BN)

1. [**Incorporating Copying Mechanism in Sequence-to-Sequence Learning**](https://arxiv.org/abs/1603.06393) **- In this paper, we incorporate copying into neural network-based Seq2Seq learning and propose a new model called CopyNet with encoder-decoder structure. CopyNet can nicely integrate the regular way of word generation in the decoder with the new copying mechanism which can choose sub-sequences in the input sequence and put them at proper places in the output sequence.**

### **QUESTION ANSWERING**

1. [**Pythia, qna for images**](https://github.com/facebookresearch/pythia) **on colab**
2. [**Building a Q\&A system part 1**](https://towardsdatascience.com/building-a-question-answering-system-part-1-9388aadff507)
3. [**Building a Q\&A model**](https://towardsdatascience.com/nlp-building-a-question-answering-model-ed0529a68c54)
4. [**Vidhya on Q\&A**](https://medium.com/analytics-vidhya/how-i-build-a-question-answering-model-3548878d5db2)
5. [**Q\&A system using**](https://medium.com/voice-tech-podcast/building-an-intelligent-qa-system-with-nlp-and-milvus-75b496702490) **** [**milvus - An open source embedding vector similarity search engine powered by Faiss, NMSLIB and Annoy**](https://github.com/milvus-io/milvus)
6. [**Q\&A system**](https://medium.com/@akshaynavalakha/nlp-question-answering-system-f05825ef35c8)
7.

### **LANGUAGE MODELS**

1. [**Mastery on Word-based** ](https://machinelearningmastery.com/develop-word-based-neural-language-models-python-keras/)

### **NEURAL LANGUAGE GENERATION**

1. [**Using RNN**](https://www.aclweb.org/anthology/C16-1103)
2. [**Using language modeling**](https://medium.com/@shivambansal36/language-modelling-text-generation-using-lstms-deep-learning-for-nlp-ed36b224b275)
3. [**Word based vs char based**](https://datascience.stackexchange.com/questions/13138/what-is-the-difference-between-word-based-and-char-based-text-generation-rnns) **- Word-based LMs display higher accuracy and lower computational cost than char-based LMs. However, char-based RNN LMs better model languages with a rich morphology such as Finish, Turkish, Russian etc. Using word-based RNN LMs to model such languages is difficult if possible at all and is not advised. Char-based RNN LMs can mimic grammatically correct sequences for a wide range of languages, require bigger hidden layer and computationally more expensive while word-based RNN LMs train faster and generate more coherent texts and yet even these generated texts are far from making actual sense.**
4. [**mediu m on Char based with code, leads to better grammer**](https://towardsdatascience.com/besides-word-embedding-why-you-need-to-know-character-embedding-6096a34a3b10)
5. [**Git, keras language models, char level word level and sentence using VAE**](https://github.com/pbloem/language-models)

### **LANGUAGE DETECTION / IDENTIFICATION**&#x20;

1. [**A qualitative comparison of google, azure, amazon, ibm LD LI**](https://medium.com/activewizards-machine-learning-company/comparison-of-the-most-useful-text-processing-apis-e4b4c1e6626a)
2. [**CLD2**](https://github.com/CLD2Owners/cld2/tree/master/docs)**,** [**CLD3**](https://github.com/google/cld3)**,** [**PYCLD**](https://github.com/aboSamoor/pycld2)**2,** [**POLYGLOT wraps CLD**](https://polyglot.readthedocs.io/en/latest/Detection.html)**,** [**alex ott cld stats**](https://gist.github.com/alexott/dd43fa8d1db4b8202d55c6325b2c69c2)**,** [**cld comparison vs tika langid**](http://blog.mikemccandless.com/2011/10/accuracy-and-performance-of-googles.html)
3. [**Fast text LI**](https://fasttext.cc/blog/2017/10/02/blog-post.html?fbclid=IwAR3dtJFRmpoZYq24U9ePlGeC65PT1Gy2Rsz9fH834CZ74Vs70utk2suuFsc)**,** [**facebook post**](https://www.facebook.com/groups/1174547215919768/permalink/1702123316495486/?comment\_id=1704414996266318\&reply\_comment\_id=1705159672858517\&notif\_id=1507280476710677\&notif\_t=group\_comment)
4. **OPENNLP**
5. [**Google detect language**](https://cloud.google.com/translate/docs/detecting-language)**,** [**github code**](https://github.com/GoogleCloudPlatform/python-docs-samples/blob/master/translate/cloud-client/snippets.py)**,** [**v3beta**](https://cloud.google.com/translate/docs/detecting-language-v3)
6. [**Microsoft azure LD,**](https://docs.microsoft.com/en-us/azure/cognitive-services/text-analytics/how-tos/text-analytics-how-to-language-detection) **** [**2**](https://westcentralus.dev.cognitive.microsoft.com/docs/services/TextAnalytics-v2-1/operations/56f30ceeeda5650db055a3c7)
7. [**Ibm watson**](https://cloud.ibm.com/apidocs/language-translator)**,** [**2**](https://www.ibm.com/support/knowledgecenter/SS8NLW\_11.0.1/com.ibm.swg.im.infosphere.dataexpl.engine.doc/c\_vse\_language\_detection.html)
8. [**Amazon,**](https://docs.aws.amazon.com/comprehend/latest/dg/how-languages.html) **** [ **2**](https://aws.amazon.com/comprehend/)
9. [**Lingua - most accurate for java… doesn't seem like its accurate enough**](https://github.com/pemistahl/lingua)
10. [**LD with infinity gram 99.1 on a lot of data a benchmark for this 2012 method**](https://shuyo.wordpress.com/2012/02/21/language-detection-for-twitter-with-99-1-accuracy/)**,** [**LD with infinity gram**](https://github.com/shuyo/ldig)
11. [**WiLI dataset for LD, comparison of CLD vs others** ](https://arxiv.org/pdf/1801.07779.pdf)
12. [**Comparison of CLD vs FT vs OPEN NLP**](http://alexott.blogspot.com/2017/10/evaluating-fasttexts-models-for.html) **- beware based on 200 samples per language!!**

**Full results for every language that I tested are in table at the end of blog post & on** [**Github**](https://gist.github.com/alexott/dd43fa8d1db4b8202d55c6325b2c69c2)**. From them I can make following conclusions:**

* **all detectors are equally good on some languages, such as, Japanese, Chinese, Vietnamese, Greek, Arabic, Farsi, Georgian, etc. - for them the accuracy of detection is between 98 & 100%;**
* **CLD is much better in detection of "rare" languages, especially for languages, that are similar to more frequently used - Afrikaans vs Dutch, Azerbaijani vs. Turkish, Malay vs. Indonesian, Nepali vs. Hindi, Russian vs Bulgarian, etc. (it could be result of imbalance of training data - I need to check the source dataset);**
* **for "major" languages not mentioned above (English, French, German, Spanish, Portuguese, Dutch) the fastText results are much better than CLD's, and in many cases lingid.py's & OpenNLP's;**
* **for many languages results for "compressed" fastText model are slightly worse than results from "full" model (mostly only by 1-2%, but could be higher, like for Kazakh when difference is 33%), but there are languages where the situation is different - results for compressed are slight better than for full (for example, for German or Dutch);**

**OpenNLP has many misclassifications for Cyrillic languages - Russian/Ukrainian, ...**

**Rafael Oliveira** [**posted on FB**](https://www.facebook.com/groups/1174547215919768/permalink/1702123316495486/?comment\_id=1704414996266318\&reply\_comment\_id=1705159672858517\&notif\_id=1507280476710677\&notif\_t=group\_comment) **a simple diagram that shows what languages are detected better by CLD & what is better handled by fastText**

**Here are some additional notes about differences in behavior of detectors that I observe during analyzing results:**

* **fastText is more reliable than CLD on the short texts;**
* **fastText models & langid.py detect language as Hebrew instead of Jewish as in CLD. Similarly, CLD uses 'in' for Indonesian language instead of standard 'id' used by fastText & langid.py;**
* **fastText distinguish between Cyrillic- & Latin-based versions of Serbian;**
* **CLD tends to incorporate geographical & person's names into detection results - for example, blog post in German about travel to Iceland is detected as Icelandic, while fastText detects it as German;**
* **In extended detection mode CLD tends to select more rare language, like, Galician or Catalan over Spanish, Serbian instead of Russian, etc.**
* **OpenNLP isn't very good in detection for short texts.**

**The models released by fastText development team provides very good alternative to existing language detection tools, like, Google's CLD & langid.py - for most of "popular" languages, these models provides higher detection accuracy comparing to other tools, combined with high speed of detection (drawback of langid.py). Even using "compressed" model it's possible to reach good detection accuracy. Although for some less frequently used languages, CLD & langid.py may show better results.**

**Performance-wise, the langid.py is much slower than both CLD & fastText. On average, CLD requires 0.5-1 ms to perform language detection. For fastText & langid.py I don't have precise numbers yet, only approximates based on speed of execution of corresponding programs.**

![](https://lh6.googleusercontent.com/GPH9qBy-b9g1ReVRsuBhFdWR94wSFJ\_FLOOci3YFVHUHcc7PiCyEMZHkzgNwuN5x4vzvW5QR1AwqrnJrRgKSukh\_WSc83GLeyCG3BaTvVVh8uY5ODjmSBl\_h\_arIwBmlfvFdM1cT)

**GIT:**&#x20;

1. [**LD with infinity gram**](https://github.com/shuyo/ldig)

**Articles:**

1. [**Medium on training LI models**](https://towardsdatascience.com/how-i-trained-a-language-detection-ai-in-20-minutes-with-a-97-accuracy-fdeca0fb7724)
2.

**Papers:**&#x20;

1. [**A comparison of lang ident approaches**](https://link.springer.com/chapter/10.1007/978-3-642-12275-0\_59)
2. [**lI on code switch social media**](https://www.aclweb.org/anthology/W18-3206)
3. [**Comparing LI methods, has 6 big languages**](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.149.630\&rep=rep1\&type=pdf)
4. [**Comparing LI techniques**](https://dbs.cs.uni-duesseldorf.de/lehre/bmarbeit/barbeiten/ba\_panich.pdf)
5. [**Radim rehurek LI on the web extending the dictionary**](https://radimrehurek.com/cicling09.pdf)
6. [**Comparative study of LI methods**](https://pdfs.semanticscholar.org/c422/3cc3765a1ac2e085b420e771d8022e6c244f.pdf)**,** [**2**](https://www.semanticscholar.org/paper/A-Comparative-Study-on-Language-Identification-Grothe-Luca/3f47b38b434f614d0cbf9af94cb4d74aa2bfe759)

### **LANGUAGE TRANSLATION**

1. [**State of the art methods for neural machine translation**](https://www.topbots.com/ai-nlp-research-neural-machine-translation/) **- a review of papers**
2. **LASER:** [**Zero shot multi lang-translation by facebook**](https://code.fb.com/ai-research/laser-multilingual-sentence-embeddings/)**,** [**github**](https://github.com/facebookresearch/LASER)
3. [**How to use laser on medium**](https://medium.com/the-artificial-impostor/multilingual-similarity-search-using-pretrained-bidirectional-lstm-encoder-e34fac5958b0)
4. **Stanford coreNLP language POS/NER/DEP PARSE etc for** [**53 languages**](https://www.analyticsvidhya.com/blog/2019/02/stanfordnlp-nlp-library-python)
5. [**Using embedding spaces**](https://rare-technologies.com/translation-matrix-in-gensim-python/) **w2v by gensim**
6. [**The risk of using bleu**](https://towardsdatascience.com/evaluating-text-output-in-nlp-bleu-at-your-own-risk-e8609665a213)
7. **Really good:** [**BLUE - what is it, how it is calculated?**](https://slator.com/technology/how-bleu-measures-translation-and-why-it-matters/)

**“\[BLEU] looks at the presence or absence of particular words, as well as the ordering and the degree of distortion—how much they actually are separated in the output.”**

**BLEU’s evaluation system requires two inputs: (i) a numerical translation closeness metric, which is then assigned and measured against (ii) a corpus of human reference translations.**

**BLEU averages out various metrics using an** [**n-gram method**](https://en.wikipedia.org/wiki/N-gram)**, a probabilistic language model often used in computational linguistics.**

![BLEU sample](https://lh4.googleusercontent.com/lSpgGLtUzukIldm3nDRFBlAigfv\_vggMinKuKjeVtIpFSR5r8VnJ6u8sZ9KkrrTuzpzO42tPjfrRlcwQVj9IAbrP6ou6pzd2XzFzxAzqlYSrCmFODdI4WvhMg7CASMk7ybANtrFw)

**The result is typically measured on a 0 to 1 scale, with 1 as the hypothetical “perfect” translation. Since the human reference, against which MT is measured, is always made up of multiple translations, even a human translation would not score a 1, however. Sometimes the score is expressed as multiplied by 100 or, as in the case of Google mentioned above, by 10.**

**a BLEU score offers more of an intuitive rather than an absolute meaning and is best used for relative judgments: “If we get a BLEU score of 35 (out of 100), it seems okay, but it actually has no correlation to the quality of the output in any meaningful sense. If it’s less than 15, we can probably safely say it’s very bad. If it’s greater than 60, we probably have some mistake in our testing! So it will generally fall in there.”**

&#x20;**“Typically, if you have multiple \[human translation] references, the BLEU score tends to be higher. So if you hear a very large BLEU score—someone gives you a value that seems very high—you can ask them if there are multiple references being used; because, then, that is the reason that the score is actually higher.”**

1. [**General talk**](https://slator.com/technology/google-facebook-amazon-neural-machine-translation-just-had-its-busiest-month-ever/) **about FAMG (fb, ama, micro, goog) and research direction atm, including some info about BLUE scores and the comparison issues with reports of BLUE (boils down to diff unmentioned parameters)**
2. **One proposed solution is** [**sacreBLUE**](https://arxiv.org/pdf/1804.08771.pdf)**, pip install sacreblue**

**Named entity language transliteration**

1. [**Paper**](https://arxiv.org/pdf/1808.02563.pdf)**,** [**blog post**](https://developer.amazon.com/blogs/alexa/post/ec66406c-094c-4dbc-8e9f-01050b27d43d/automatic-transliteration-can-help-alexa-find-data-across-language-barriers)**:  English russian, hebrew, arabic, japanese, with data set and** [**github**](https://github.com/steveash/NETransliteration-COLING2018)

### **CHAT BOTS**

1. [**A list of chat bots, pros and cons, example code.**](https://nlpforhackers.io/chatbots-introduction/#more-8595)

### ****
