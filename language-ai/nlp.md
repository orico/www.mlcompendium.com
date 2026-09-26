# NLP Tools

The foundations only matter once a library runs them, so the practical question is which tool to pick and what data to feed it. The page starts with spaCy, then compares it with the other NLP libraries, and ends with dataset collections and embedding repositories.

#### SPACY

spaCy is the default starting point, so the first links are how to use it, how to learn it properly, and how to make it fast.

The same notes are in [Named Entity Recognition (NER)](named-entity-recognition-ner.md) and [Swiss army knife libraries](foundation-nlp.md#swiss-army-knife-libraries).

[Vidhaya on spacy vs ner](https://www.analyticsvidhya.com/blog/2017/04/natural-language-processing-made-easy-using-spacy-%E2%80%8Bin-python/) — tutorial + code on how to use spacy for pos, dep, ner, compared to nltk/corenlp (sner etc). The results reflect a global score not specific to LOC for example. For a structured path, the [spaCy course](https://course.spacy.io/) is "Advanced NLP with spaCy", a free online course. Once the pipeline works, SPACY OPTIMIZATION — [LP using CYTHON and SPACY.](https://medium.com/huggingface/100-times-faster-natural-language-processing-in-python-ee32033bdced) is Thomas Wolf's "100 Times Faster Natural Language Processing in Python".

### NLP Libraries

spaCy is not the only choice, and the comparisons below are about accuracy, speed, and fit for production. A page that has all the known libraries used to sit here; its address now points to an unrelated site and is kept at the end of the page.

For deep learning, "Deep Learning for NLP: SpaCy vs PyTorch vs AllenNLP" on Lucky's Notes is the [Comparison between spacy, pytorch, allenlp](https://luckytoilet.wordpress.com/2018/12/29/deep-learning-for-nlp-spacy-vs-pytorch-vs-allennlp/). spaCy's own Facts & Figures page, the hard numbers for spaCy and how it compares to other tools, is the [Comparison spacy,nltk](https://spacy.io/usage/facts-figures). For production, [Comparing Production grade nlp libs](https://www.oreilly.com/ideas/comparing-production-grade-nlp-libraries-accuracy-performance-and-scalability) is Saif Addin Ellafi's comparison of the accuracy and performance of Spark-NLP vs. spaCy, with use case recommendations. The older framing of the choice is [nltk vs spacy](https://web.archive.org/web/20201109032639/https://blog.thedataincubator.com/2016/04/nltk-vs-spacy-natural-language-processing-in-python/): the venerable NLTK has been the standard tool for natural language processing in Python, with a wide variety of tools, algorithms, and corpuses, and spaCy arrived as a competitor aiming at powerful, streamlined processing.

### NLP DATASETS

A library needs data, and a few collections gather most of it. [The bid bad](https://datasets.quantumstat.com/) is the big NLP dataset database, and the 600 [medium](https://medium.com/towards-artificial-intelligence/600-nlp-datasets-and-glory-4b0080bf5ab) post, "600 NLP Datasets and Glory", is the newsletter update on the current state of that Big Bad NLP Database. For many languages at once, [Amazon 51 Language datasets for NLU](https://www.amazon.science/blog/amazon-releases-51-language-dataset-for-language-understanding) is the MASSIVE dataset, released with the Massively Multilingual NLU (MMNLU-22) competition and workshop to help researchers scale natural-language-understanding technology to every language on Earth.

### NLP embedding repositories

Beyond raw datasets, pretrained word vectors save training from scratch.

The same notes are in [Embedding](../deep-learning/representations.md).

[Nlpl](http://vectors.nlpl.eu/repository/) is the Nordic Language Processing Laboratory word embeddings repository.

## Deprecated links

{% hint style="warning" %}
These links and images no longer work. The original wording is kept here. A same-resource copy, when one was checked, is used above.
{% endhint %}

- nltk vs spacy This address no longer opens: https://blog.thedataincubator.com/2016/04/nltk-vs-spacy-natural-language-processing-in-python/
- Has all the known libraries. This address now points to an unrelated site: https://nlpforhackers.io/libraries/
