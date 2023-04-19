# Embedding

## **Intro**

**(amazing)** [**embeddings from the ground up singlelunch**](https://www.singlelunch.com/2020/02/16/embeddings-from-the-ground-up/)

## **VECTOR SIMILARITY SEARCH**

1. [**Faiss**](https://github.com/facebookresearch/faiss) **- a library for efficient similarity search**
2. [**Benchmarking**](https://github.com/erikbern/ann-benchmarks) **- complete with almost everything imaginable**
3. [**Singlestore**](https://www.singlestore.com/solutions/predictive-ml-ai/)
4. **Elastic search -** [**dense vector**](https://www.elastic.co/guide/en/elasticsearch/reference/7.6/query-dsl-script-score-query.html#vector-functions)
5. **Google cloud vertex matching engine** [**NN search**](https://cloud.google.com/blog/products/ai-machine-learning/vertex-matching-engine-blazing-fast-and-massively-scalable-nearest-neighbor-search)
   1. **search**
      1. **Recommendation engines**
      2. **Search engines**
      3. **Ad targeting systems**
      4. **Image classification or image search**
      5. **Text classification**
      6. **Question answering**
      7. **Chat bots**
   2. **Features**
      1. **Low latency**
      2. **High recall**
      3. **managed**
      4. **Filtering**
      5. **scale**
6. **Pinecone - managed** [**vector similarity search**](https://www.pinecone.io/) **- Pinecone is a fully managed vector database that makes it easy to add vector search to production applications. No more hassles of benchmarking and tuning algorithms or building and maintaining infrastructure for vector search.**
7. [**Nmslib**](https://github.com/nmslib/nmslib) **(**[**benchmarked**](https://github.com/erikbern/ann-benchmarks) **- Benchmarks of approximate nearest neighbor libraries in Python) is a Non-Metric Space Library (NMSLIB): An efficient similarity search library and a toolkit for evaluation of k-NN methods for generic non-metric spaces.**
8. **scann,**
9. [**Vespa.ai**](https://vespa.ai/) **- Make AI-driven decisions using your data, in real time. At any scale, with unbeatable performance**
10. [**Weaviate**](https://www.semi.technology/developers/weaviate/current/) **- Weaviate is an** [**open source**](https://github.com/semi-technologies/weaviate) **vector search engine and vector database. Weaviate uses machine learning to vectorize and store data, and to find answers to natural language queries, or any other media type.**
11. [**Neural Search with BERT and Solr**](https://dmitry-kan.medium.com/list/vector-search-e9b564d14274) **- Indexing BERT vector data in Solr and searching with full traversal**
12. [**Fun With Apache Lucene and BERT Embeddings**](https://medium.com/swlh/fun-with-apache-lucene-and-bert-embeddings-c2c496baa559) **- This post goes much deeper -- to the similarity search algorithm on Apache Lucene level. It upgrades the code from 6.6 to 8.0**
13. [**Speeding up BERT Search in Elasticsearch**](https://towardsdatascience.com/speeding-up-bert-search-in-elasticsearch-750f1f34f455) **- Neural Search in Elasticsearch: from vanilla to KNN to hardware acceleration**
14. [**Ask Me Anything about Vector Search**](https://towardsdatascience.com/ask-me-anything-about-vector-search-4252a01f3889) **- In the Ask Me Anything: Vector Search! session Max Irwin and Dmitry Kan discussed major topics of vector search, ranging from its areas of applicability to comparing it to good ol’ sparse search (TF-IDF/BM25), to its readiness for prime time and what specific engineering elements need further tuning before offering this to users.**
15. [**Search with BERT vectors in Solr and Elasticsearch**](https://github.com/DmitryKey/bert-solr-search) **- GitHub repository used for experiments with Solr and Elasticsearch using DBPedia abstracts comparing Solr, vanilla Elasticsearch, elastiknn enhanced Elasticsearch, OpenSearch, and GSI APU**
16. [**Not All Vector Databases Are Made Equal**](https://towardsdatascience.com/milvus-pinecone-vespa-weaviate-vald-gsi-what-unites-these-buzz-words-and-what-makes-each-9c65a3bd0696) **- A detailed comparison of Milvus, Pinecone, Vespa, Weaviate, Vald, GSI and Qdrant**
17. [**Vector Podcast**](https://dmitry-kan.medium.com/vector-podcast-e27d83ecd0be) **- Podcast hosted by Dmitry Kan, interviewing the makers in the Vector / Neural Search industry. Available on YouTube, Spotify, Apple Podcasts and RSS**
18. [**Players in Vector Search: Video**](https://dmitry-kan.medium.com/players-in-vector-search-video-2fd390d00d6) **-Video recording and slides of the talk presented on London IR Meetup on the topic of players, algorithms, software and use cases in Vector Search**
19. **(paper)** [**Hybrid retrieval using  search and semantic search**](https://arxiv.org/abs/2210.11934)

## **TOOLS**

### **FLAIR**

1. **Name-Entity Recognition (NER): It can recognise whether a word represents a person, location or names in the text.**
2. **Parts-of-Speech Tagging (PoS): Tags all the words in the given text as to which “part of speech” they belong to.**
3. **Text Classification: Classifying text based on the criteria (labels)**
4. **Training Custom Models: Making our own custom models.**
5. **It comprises of popular and state-of-the-art word embeddings, such as GloVe, BERT, ELMo, Character Embeddings, etc. There are very easy to use thanks to the Flair API**
6. **Flair’s interface allows us to combine different word embeddings and use them to embed documents. This in turn leads to a significant uptick in results**
7. **‘Flair Embedding’ is the signature embedding provided within the Flair library. It is powered by contextual string embeddings. We’ll understand this concept in detail in the next section**
8. **Flair supports a number of languages – and is always looking to add new ones**

### **HUGGING FACE**

1. [**Git**](https://github.com/huggingface/transformers)
2.
   1. [**Hugging face pytorch transformers**](https://github.com/huggingface/pytorch-transformers)
3. [**Hugging face nlp pretrained**](https://huggingface.co/models?search=Helsinki-NLP%2Fopus-mt\&fbclid=IwAR0YN7qn9uTlCeBOZw4jzWgq9IXq\_9ju1ww\_rVL-f1fa9EjlSP50q05QcmU)
4. [**hugging face on emotions**](https://medium.com/huggingface/understanding-emotions-from-keras-to-pytorch-3ccb61d5a983)
   1. **how to make a custom pyTorch LSTM with custom activation functions,**
   2. **how the PackedSequence object works and is built,**
   3. **how to convert an attention layer from Keras to pyTorch,**
   4. **how to load your data in pyTorch: DataSets and smart Batching,**
   5. **how to reproduce Keras weights initialization in pyTorch.**
5. **A** [**thorough tutorial on bert**](http://mccormickml.com/2019/07/22/BERT-fine-tuning/)**, fine tuning using hugging face transformers package.** [**Code**](https://colab.research.google.com/drive/1Y4o3jh3ZH70tl6mCd76vz\_IxX23biCPP)

**Youtube** [**ep1**](https://www.youtube.com/watch?v=FKlPCK1uFrc)**,** [**2**](https://www.youtube.com/watch?v=zJW57aCBCTk)**,** [**3**](https://www.youtube.com/watch?v=x66kkDnbzi4)**,** [**3b**](https://www.youtube.com/watch?v=Hnvb9b7a\_Ps)**,**

## **LANGUAGE EMBEDDINGS**

![](https://lh6.googleusercontent.com/aibqScGzh66aJK9E5Rho61W\_pX8Kw82vJrrUkvRZrRN7vaRBOWnDOz0k29szquWdU3i4cwFFUj6b4-rPZvU2AUIlP5ouxwS7Kq2RwxDwFxtm9fpJZcnVXCMHY3SJ43FEsWj\_GTcT)

### **History**

1. [**Google’s intro to transformers and multi-head self attention**](https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html)
2. [**How self attention and relative positioning work**](https://medium.com/@\_init\_/how-self-attention-with-relative-position-representations-works-28173b8c245a) **(great!)**
   1. **Rnns are sequential, same word in diff position will have diff encoding due to the input from the previous word, which is inherently different.**
   2. **Attention without positional! Will have distinct (Same) encoding.**
   3. **Relative look at a window around each word and adds a distance vector in terms of how many words are before and after, which fixes the problem.**
   4. ![](https://lh3.googleusercontent.com/XmFsG2XDB2sLXNkRwmsc90iPfXPBWDgr4AzO-u8lejinMcwb5XzTppAZ5oekBjUjIsJ8u8IBA83Z31bP3rgMjdkvq0qZAteTE2VvxSOa79AUH4KqsRQb0w1Eworanxxm7zFuo494)
   5. ![](https://lh6.googleusercontent.com/JNAgD9NAJQzXCtfZ3ekWddZ1m8nzgMwXoqoQ3rjLsKfHl2NdqVrdrYexDnXCUzik2ZYalllJhm7Hp5Zl1\_L5EHumNGN0NAfFHHH0RM6gqBZc4bPkg7Bd4D5ea5gmV1\_hXtMXW\_9K)
   6. **The authors hypothesized that precise relative position information is not useful beyond a certain distance.**
   7. **Clipping the maximum distance enables the model to generalize to sequence lengths not seen during training.**
3. [**From bert to albert**](https://medium.com/@hamdan.hussam/from-bert-to-albert-pre-trained-langaug-models-5865aa5c3762)
4. [**All the latest buzz algos**](https://www.topbots.com/most-important-ai-nlp-research/#ai-nlp-paper-2018-12)
5. **A** [**Summary of them**](https://www.topbots.com/ai-nlp-research-pretrained-language-models/?utm\_source=facebook\&utm\_medium=group\_post\&utm\_campaign=pretrained\&fbclid=IwAR0smqf8qanfMayo4fRH2hFuc5LYA8-Bn5oEp-xedKcRR43QsqXIelIAzEE)
6. [**8 pretrained language embeddings**](https://www.analyticsvidhya.com/blog/2019/03/pretrained-models-get-started-nlp/)
7. [**Hugging face pytorch transformers**](https://github.com/huggingface/pytorch-transformers)
8. [**Hugging face nlp pretrained**](https://huggingface.co/models?search=Helsinki-NLP%2Fopus-mt\&fbclid=IwAR0YN7qn9uTlCeBOZw4jzWgq9IXq\_9ju1ww\_rVL-f1fa9EjlSP50q05QcmU)

### **Embedding Foundation Knowledge**

1. [**Medium on Introduction into word embeddings, sentence embeddings, trends in the field.**](https://towardsdatascience.com/deep-transfer-learning-for-natural-language-processing-text-classification-with-universal-1a2c69e5baa9) **The Indian guy,** [**git**](https://nbviewer.jupyter.org/github/dipanjanS/data\_science\_for\_all/blob/master/tds\_deep\_transfer\_learning\_nlp\_classification/Deep%20Transfer%20Learning%20for%20NLP%20-%20Text%20Classification%20with%20Universal%20Embeddings.ipynb) **notebook,** [**his git**](https://github.com/dipanjanS)**,**
   1. **Baseline Averaged Sentence Embeddings**
   2. **Doc2Vec**
   3. **Neural-Net Language Models (Hands-on Demo!)**
   4. **Skip-Thought Vectors**
   5. **Quick-Thought Vectors**
   6. **InferSent**
   7. **Universal Sentence Encoder**
2. [**Shay palachy on word embedding covering everything from bow to word/doc/sent/phrase.**](https://medium.com/@shay.palachy/document-embedding-techniques-fed3e7a6a25d)
3. [**Another intro, not as good as the one above**](https://medium.com/huggingface/universal-word-sentence-embeddings-ce48ddc8fc3a)
4. [**Using sklearn vectorizer to create custom ones, i.e. a vectorizer that does preprocessing and tfidf and other things.**](https://towardsdatascience.com/hacking-scikit-learns-vectorizers-9ef26a7170af)
5. [**TFIDF - n-gram based top weighted tfidf words**](https://stackoverflow.com/questions/25217510/how-to-see-top-n-entries-of-term-document-matrix-after-tfidf-in-scikit-learn)
6. [**Gensim bi-gram phraser/phrases analyser/converter**](https://radimrehurek.com/gensim/models/phrases.html)
7. [**Countvectorizer, stemmer, lemmatization code tutorial**](https://medium.com/@rnbrown/more-nlp-with-sklearns-countvectorizer-add577a0b8c8)
8. [**Current 2018 best universal word and sentence embeddings -> elmo**](https://medium.com/huggingface/universal-word-sentence-embeddings-ce48ddc8fc3a)
9. [**5-part series on word embeddings**](http://ruder.io/word-embeddings-1/)**,** [**part 2**](http://ruder.io/word-embeddings-softmax/index.html)**,** [**3**](http://ruder.io/secret-word2vec/index.html)**,** [**4 - cross lingual review**](http://ruder.io/cross-lingual-embeddings/index.html)**,** [**5-future trends**](http://ruder.io/word-embeddings-2017/index.html)
10. [**Word embedding posts**](https://datawarrior.wordpress.com/2016/05/15/word-embedding-algorithms/)
11. [**Facebook github for embedings called starspace**](https://github.com/facebookresearch/StarSpace)
12. [**Medium on Fast text / elmo etc**](https://medium.com/huggingface/universal-word-sentence-embeddings-ce48ddc8fc3a)

### **Language modeling**

1. [**Ruder on language modelling as the next imagenet**](http://ruder.io/nlp-imagenet/) **- Language modelling, the last approach mentioned, has been shown to capture many facets of language relevant for downstream tasks, such as** [**long-term dependencies**](https://arxiv.org/abs/1611.01368) **,** [**hierarchical relations**](https://arxiv.org/abs/1803.11138) **, and** [**sentiment**](https://arxiv.org/abs/1704.01444) **. Compared to related unsupervised tasks such as skip-thoughts and autoencoding,** [**language modelling performs better on syntactic tasks even with less training data**](https://openreview.net/forum?id=BJeYYeaVJ7)**.**
2. **A** [**tutorial**](https://blog.myyellowroad.com/unsupervised-sentence-representation-with-deep-learning-104b90079a93) **about w2v skipthought - with code!, specifically language modelling here is important - Our second method is training a language model to represent our sentences. A language model describes the probability of a text existing in a language. For example, the sentence “I like eating bananas” would be more probable than “I like eating convolutions.” We train a language model by slicing windows of n words and predicting what the next word will be in the text**
3. [**Unread - universal language model fine tuning for text-classification**](https://arxiv.org/abs/1801.06146)
4. **ELMO -** [**medium**](https://towardsdatascience.com/beyond-word-embeddings-part-2-word-vectors-nlp-modeling-from-bow-to-bert-4ebd4711d0ec)
5. [**Bert**](https://arxiv.org/abs/1810.04805v1) **\*\*\[python git]\(**[https://github.com/CyberZHG/keras-bert](https://github.com/CyberZHG/keras-bert)**)**- We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT representations can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications. BERT is conceptually simple and empirically powerful. It obtains new state-of-the-art results on eleven natural language processing tasks.\*\* ![](https://lh4.googleusercontent.com/anFY63RxhdYt82bb\_XUGDLRUmj2vuR1I0iJye66cOqgC2gQegXVf2ibkC64LRPIfgUj8Brl7VYUFfxw3gG0KBnwTuqJ2NCohd6mi9YzCkZmHGuDz1QxXl7JUtMv5BpiBJXGnC-Zc)
6. [**Open.ai on language modelling**](https://blog.openai.com/language-unsupervised/) **- We’ve obtained state-of-the-art results on a suite of diverse language tasks with a scalable, task-agnostic system, which we’re also releasing. Our approach is a combination of two existing ideas:** [**transformers**](https://arxiv.org/abs/1706.03762) **and** [**unsupervised pre-training**](https://arxiv.org/abs/1511.01432)**.** [**READ PAPER**](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language\_understanding\_paper.pdf)**,** [**VIEW CODE**](https://github.com/openai/finetune-transformer-lm)**.**
7. **Scikit-learn inspired model finetuning for natural language processing.**

[**finetune**](https://finetune.indico.io/#module-finetune) **ships with a pre-trained language model from** [**“Improving Language Understanding by Generative Pre-Training”**](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language\_understanding\_paper.pdf) **and builds off the** [**OpenAI/finetune-language-model repository**](https://github.com/openai/finetune-transformer-lm)**.**

1. **Did not read -** [**The annotated Transformer**](http://nlp.seas.harvard.edu/2018/04/03/attention.html?fbclid=IwAR2\_ZOfUfXcto70apLdT\_StObPwatYHNRPP4OlktcmGfj9uPLhgsZPsAXzE) **- jupyter on transformer with annotation**
2. **Medium on** [**Dissecting Bert**](https://medium.com/dissecting-bert/dissecting-bert-part-1-d3c3d495cdb3)**,** [**appendix**](https://medium.com/dissecting-bert/dissecting-bert-appendix-the-decoder-3b86f66b0e5f)
3. [**Medium on distilling 6 patterns from bert**](https://towardsdatascience.com/deconstructing-bert-distilling-6-patterns-from-100-million-parameters-b49113672f77)

### **Embedding spaces**

1. [**A good overview of sentence embedding methods**](http://mlexplained.com/2017/12/28/an-overview-of-sentence-embedding-methods/) **- w2v ft s2v skip, d2v**
2. [**A very good overview of word embeddings**](http://sanjaymeena.io/tech/word-embeddings/)
3. [**Intro to word embeddings - lots of images**](https://www.springboard.com/blog/introduction-word-embeddings/)
4. [**A very long and extensive thesis about embeddings**](http://ad-publications.informatik.uni-freiburg.de/theses/Bachelor\_Jon\_Ezeiza\_2017.pdf)
5. [**Sent2vec by gensim**](https://rare-technologies.com/sent2vec-an-unsupervised-approach-towards-learning-sentence-embeddings/) **- sentence embedding is defined as the average of the source word embeddings of its constituent words. This model is furthermore augmented by also learning source embeddings for not only unigrams but also n-grams of words present in each sentence, and averaging the n-gram embeddings along with the words**
6. [**Sent2vec vs fasttext - with info about s2v parameters**](https://github.com/epfml/sent2vec/issues/19)
7. [**Wordrank vs fasttext vs w2v comparison**](https://en.wikipedia.org/wiki/Automatic\_summarization#TextRank\_and\_LexRank) **- the better word similarity algorithm**
8. [**W2v vs glove vs sppmi vs svd by gensim**](https://rare-technologies.com/making-sense-of-word2vec/)
9. [**Medium on a gentle intro to d2v**](https://medium.com/scaleabout/a-gentle-introduction-to-doc2vec-db3e8c0cce5e)
10. [**Doc2vec tutorial by gensim**](https://rare-technologies.com/doc2vec-tutorial/) **- Doc2vec (aka paragraph2vec, aka sentence embeddings) modifies the word2vec algorithm to unsupervised learning of continuous representations for larger blocks of text, such as sentences, paragraphs or entire documents. - Most importantly this tutorial has crucial information about the implementation parameters that should be read before using it.**
11. [**Lbl2Vec**](https://github.com/sebischair/Lbl2Vec)**,** [medium](https://towardsdatascience.com/unsupervised-text-classification-with-lbl2vec-6c5e040354de), is an algorithm for unsupervised document classification and unsupervised document retrieval. It automatically generates jointly embedded label, document and word vectors and returns documents of categories modeled by manually predefined keywords.
12. [**Git for word embeddings - taken from mastery’s nlp course**](https://github.com/IshayTelavivi/nlp\_crash\_course)
13. [**Skip-thought -**](http://mlexplained.com/2017/12/28/an-overview-of-sentence-embedding-methods/) **\*\*\[git]\(**[https://github.com/ryankiros/skip-thoughts](https://github.com/ryankiros/skip-thoughts)**)**- Where word2vec attempts to predict surrounding words from certain words in a sentence, skip-thought vector extends this idea to sentences: it predicts surrounding sentences from a given sentence. NOTE: Unlike the other methods, skip-thought vectors require the sentences to be ordered in a semantically meaningful way. This makes this method difficult to use for domains such as social media text, where each snippet of text exists in isolation.\*\*
14. [**Fastsent**](http://mlexplained.com/2017/12/28/an-overview-of-sentence-embedding-methods/) **- Skip-thought vectors are slow to train. FastSent attempts to remedy this inefficiency while expanding on the core idea of skip-thought: that predicting surrounding sentences is a powerful way to obtain distributed representations. Formally, FastSent represents sentences as the simple sum of its word embeddings, making training efficient. The word embeddings are learned so that the inner product between the sentence embedding and the word embeddings of surrounding sentences is maximized. NOTE: FastSent sacrifices word order for the sake of efficiency, which can be a large disadvantage depending on the use-case.**
15. **Weighted sum of words - In this method, each word vector is weighted by the factor** ![\frac{a}{a + p(w)}](https://lh3.googleusercontent.com/p6He6GoHCb-yA8QgNrn4eIrWTa5i\_7lolQyY6EplDa1l7bmf1IF0y-eNuGOPfMfLKMkyw5qOpkwzoejmNB44Fg9fIwt4bIPkYOSWT7r50wdgdhT7qUiDwyNh1toe21CQFolKp5py) **where** ![a](https://lh5.googleusercontent.com/qeqpAm9JfrNP8TnZzbUsMBKcsv2v-ZpZbmbM01Uf22HVUBcZMwa5nseCQMW\_XGYNZQQJ1HvYqOMwGfaL\_5NDbrOa\_aJTAsA3JdoHEUaB9XMq-sDUKtR348dq6TJuHEr05hetP0-7) **is a hyperparameter and** ![p(w)](https://lh6.googleusercontent.com/cPiXavxPJ8voQb9UE8cmzaNsV\_dMWFvG1E5SYJGGm6QrMiA9X\_uNUWjb45L96WWhAKLxvLIF4oOXI2q0m5NQRNNzKgBrogEubQDN5bDXPw66sSOyfdx3dzGxjSvwdGYgpAy60B33) **is the (estimated) word frequency. This is similar to tf-idf weighting, where more frequent terms are weighted downNOTE: Word order and surrounding sentences are ignored as well, limiting the information that is encoded.**
16. [**Infersent by facebook**](https://github.com/facebookresearch/InferSent) **-** [**paper**](https://arxiv.org/abs/1705.02364) **InferSent is a sentence embeddings method that provides semantic representations for English sentences. It is trained on natural language inference data and generalizes well to many different tasks. ABSTRACT: we show how universal sentence representations trained using the supervised data of the Stanford Natural Language Inference datasets can consistently outperform unsupervised methods like SkipThought vectors on a wide range of transfer tasks. Much like how computer vision uses ImageNet to obtain features, which can then be transferred to other tasks, our work tends to indicate the suitability of natural language inference for transfer learning to other NLP tasks.**
17. [**Universal sentence encoder - google**](https://tfhub.dev/google/universal-sentence-encoder/1) **-** [**notebook**](https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/semantic\_similarity\_with\_tf\_hub\_universal\_encoder.ipynb#scrollTo=8OKy8WhnKRe\_)**,** [**git**](https://github.com/tensorflow/hub/blob/master/examples/colab/semantic\_similarity\_with\_tf\_hub\_universal\_encoder.ipynb) **The Universal Sentence Encoder encodes text into high dimensional vectors that can be used for text classification, semantic similarity, clustering and other natural language tasks. The model is trained and optimized for greater-than-word length text, such as sentences, phrases or short paragraphs. It is trained on a variety of data sources and a variety of tasks with the aim of dynamically accommodating a wide variety of natural language understanding tasks. The input is variable length English text and the output is a 512 dimensional vector. We apply this model to the** [**STS benchmark**](http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark) **for semantic similarity, and the results can be seen in the** [**example notebook**](https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/semantic\_similarity\_with\_tf\_hub\_universal\_encoder.ipynb) **made available. The universal-sentence-encoder model is trained with a deep averaging network (DAN) encoder.**
18. [**Multi language universal sentence encoder**](https://ai.googleblog.com/2019/07/multilingual-universal-sentence-encoder.html?fbclid=IwAR2fubNOwrxWWxYous7IyQCJ3\_bY0UAdAYO\_yuWONMv-aV3o8hDckSS3FCE) **- no hebrew**
19. **Pair2vec -** [**paper**](https://arxiv.org/abs/1810.08854) **- paper proposes new methods for learning and using embeddings of word pairs that implicitly represent background knowledge about such relationships. I.e., using p2v information with existing models to increase performance. Experiments show that our pair embeddings can complement individual word embeddings, and that they are perhaps capturing information that eludes the traditional interpretation of the Distributional Hypothesis**
20. [**Fast text python tutorial**](http://ai.intelligentonlinetools.com/ml/fasttext-word-embeddings-text-classification-python-mlp/)

## Embedding Models

### **Cat2vec**

1. **Part1:** [**Label encoder/ ordinal, One hot, one hot with a rare bucket, hash**](https://blog.myyellowroad.com/using-categorical-data-in-machine-learning-with-python-from-dummy-variables-to-deep-category-66041f734512)
2. [**Part2: cat2vec using w2v**](https://blog.myyellowroad.com/using-categorical-data-in-machine-learning-with-python-from-dummy-variables-to-deep-category-42fd0a43b009)**, and entity embeddings for categorical data**

![](https://lh6.googleusercontent.com/BJjrzp0YPmsy2\_OKecufELzNU\_AO2I2kSAx9ekSbGmGYNJ27AGkbdhwPv45iMVub\_6q0AHF91N6BYdxA4l-eAUspOIat-QMU8xHQrSYYpWmu7TEO8NmRPIcrPItwq1TgkJN-LTd3)

### **ENTITY EMBEDDINGS**

1. **Star -** [**General purpose embedding paper with code somewhere**](https://arxiv.org/pdf/1709.03856.pdf)
2. [**Using embeddings on tabular data, specifically categorical - introduction**](http://www.fast.ai/2018/04/29/categorical-embeddings/)**, using fastai without limiting ourselves to pytorch - the material from this post is covered in much more detail starting around 1:59:45 in** [**the Lesson 3 video**](http://course.fast.ai/lessons/lesson3.html) **and continuing in** [**Lesson 4**](http://course.fast.ai/lessons/lesson4.html) **of our free, online** [**Practical Deep Learning for Coders**](http://course.fast.ai/) **course. To see example code of how this approach can be used in practice, check out our** [**Lesson 3 jupyter notebook**](https://github.com/fastai/fastai/blob/master/courses/dl1/lesson3-rossman.ipynb)**. Perhaps Saturday and Sunday have similar behavior, and maybe Friday behaves like an average of a weekend and a weekday. Similarly, for zip codes, there may be patterns for zip codes that are geographically near each other, and for zip codes that are of similar socio-economic status. The jupyter notebook doesn't seem to have the embedding example they are talking about.**
3. [**Rossman on kaggle**](http://blog.kaggle.com/2016/01/22/rossmann-store-sales-winners-interview-3rd-place-cheng-gui/)**, used entity-embeddings,** [**here**](https://www.kaggle.com/c/rossmann-store-sales/discussion/17974)**,** [**github**](https://github.com/entron/entity-embedding-rossmann)**,** [**paper**](https://arxiv.org/abs/1604.06737)
4. [**Medium on rossman - good**](https://towardsdatascience.com/deep-learning-structured-data-8d6a278f3088)
5. [**Embedder**](https://github.com/dkn22/embedder) **- git code for a simplified entity embedding above.**
6. **Finally what they do is label encode each feature using labelEncoder into an int-based feature, then push each feature into its own embedding layer of size 1 with an embedding size defined by a rule of thumb (so it seems), merge all layers, train a synthetic regression/classification and grab the weights of the corresponding embedding layer.**
7. [**Entity2vec**](https://github.com/ot/entity2vec)
8. [**Categorical using keras**](https://medium.com/@satnalikamayank12/on-learning-embeddings-for-categorical-data-using-keras-165ff2773fc9)

### **ALL2VEC EMBEDDINGS**

1. [**ALL ???-2-VEC ideas**](https://github.com/MaxwellRebo/awesome-2vec)
2. **Fast.ai** [**post**](http://www.fast.ai/2018/04/29/categorical-embeddings/) **regarding embedding for tabular data, i.e., cont and categorical data**
3. [**Entity embedding for**](https://github.com/fastai/fastai/blob/master/courses/dl1/lesson3-rossman.ipynb) **categorical data +** [**notebook**](https://github.com/fastai/fastai/blob/master/courses/dl1/lesson3-rossman.ipynb)
4. [**Kaggle taxi competition + code**](http://blog.kaggle.com/2015/07/27/taxi-trajectory-winners-interview-1st-place-team-%F0%9F%9A%95/)
5. [**Ross man competition - entity embeddings, code missing**](http://blog.kaggle.com/2016/01/22/rossmann-store-sales-winners-interview-3rd-place-cheng-gui/) **+**[**alternative code**](https://github.com/entron/entity-embedding-rossmann)
6. [**CODE TO CREATE EMBEDDINGS straight away, based onthe ideas by cheng guo in keras**](https://github.com/dkn22/embedder)
7. [**PIN2VEC - pinterest embeddings using the same idea**](https://medium.com/the-graph/applying-deep-learning-to-related-pins-a6fee3c92f5e)
8. [**Tweet2Vec**](https://github.com/soroushv/Tweet2Vec) **- code in theano,** [**paper**](https://dl.acm.org/citation.cfm?doid=2911451.2914762)**.**
9. [**Clustering**](https://github.com/svakulenk0/tweet2vec\_clustering) **of tweet2vec,** [**paper**](https://arxiv.org/abs/1703.05123)
10. **Paper:** [**Character neural embeddings for tweet clustering**](https://arxiv.org/pdf/1703.05123.pdf)
11. **Diff2vec - might be useful on social network graphs,** [**paper**](http://homepages.inf.ed.ac.uk/s1668259/papers/sequence.pdf)**,** [**code**](https://github.com/benedekrozemberczki/diff2vec)
12. **emoji 2vec (below)**
13. [**Char2vec**](https://hackernoon.com/chars2vec-character-based-language-model-for-handling-real-world-texts-with-spelling-errors-and-a3e4053a147d) **\*\*\[Git]\(**[https://github.com/IntuitionEngineeringTeam/chars2vec](https://github.com/IntuitionEngineeringTeam/chars2vec)**)**, similarity measure for words with types. **\[ \*\***]\([https://arxiv.org/abs/1708.00524](https://arxiv.org/abs/1708.00524))

**EMOJIS**

1. **1.** [**Deepmoji**](http://datadrivenjournalism.net/featured\_projects/deepmoji\_using\_emojis\_to\_teach\_ai\_about\_emotions)**,**
2. [**hugging face on emotions**](https://medium.com/huggingface/understanding-emotions-from-keras-to-pytorch-3ccb61d5a983)
   1. **how to make a custom pyTorch LSTM with custom activation functions,**
   2. **how the PackedSequence object works and is built,**
   3. **how to convert an attention layer from Keras to pyTorch,**
   4. **how to load your data in pyTorch: DataSets and smart Batching,**
   5. **how to reproduce Keras weights initialization in pyTorch.**
3. [**Another great emoji paper, how to get vector representations from**](https://aclweb.org/anthology/S18-1039)
4. [**3. What can we learn from emojis (deep moji)**](https://www.media.mit.edu/posts/what-can-we-learn-from-emojis/)
5. [**Learning millions of**](https://arxiv.org/pdf/1708.00524.pdf) **for emoji, sentiment, sarcasm,** [**medium**](https://medium.com/@bjarkefelbo/what-can-we-learn-from-emojis-6beb165a5ea0)
6. [**EMOJI2VEC**](https://tech.instacart.com/deep-learning-with-emojis-not-math-660ba1ad6cdc) **- medium article with keras code, a**[**nother paper on classifying tweets using emojis**](https://arxiv.org/abs/1708.00524)
7. [**Group2vec**](https://github.com/cerlymarco/MEDIUM\_NoteBook/tree/master/Group2Vec) **git and** [**medium**](https://towardsdatascience.com/group2vec-for-advance-categorical-encoding-54dfc7a08349)**, which is a multi input embedding network using a-f below. plus two other methods that involve groupby and applying entropy and join/countvec per class. Really interesting**
   1. **Initialize embedding layers for each categorical input;**
   2. **For each category, compute dot-products among other embedding representations. These are our ‘groups’ at the categorical level;**
   3. **Summarize each ‘group’ adopting an average pooling;**
   4. **Concatenate ‘group’ averages;**
   5. **Apply regularization techniques such as BatchNormalization or Dropout;**
   6. **Output probabilities.**

### **WORD2VEC**

1. **Monitor** [**train loss**](https://stackoverflow.com/questions/52038651/loss-does-not-decrease-during-training-word2vec-gensim) **using callbacks for word2vec**
2. **Cleaning datasets using weighted w2v sentence encoding, then pca and isolation forest to remove outlier sentences.**
3. [**Removing ‘gender bias using pair mean pca**](https://stackoverflow.com/questions/48019843/pca-on-word2vec-embeddings)
4. [**KPCA w2v approach on a very small dataset**](https://medium.com/@vishwanigupta/kpca-skip-gram-model-improving-word-embedding-a6a0cb7aad49)**,** [**similar git**](https://github.com/niitsuma/wordca) **for correspondence analysis,** [**paper**](https://arxiv.org/abs/1605.05087)
5. [**The best w2v/tfidf/bow/ embeddings post ever**](https://www.analyticsvidhya.com/blog/2017/06/word-embeddings-count-word2veec/)
6. [**Chris mccormick ml on w2v,**](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/) **\*\*\[post #2]\(**[http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/](http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/)**)** - negative sampling “Negative sampling addresses this by having each training sample only modify a small percentage of the weights, rather than all of them. With negative sampling, we are instead going to randomly select just a small number of “negative” words (let’s say 5) to update the weights for. (In this context, a “negative” word is one for which we want the network to output a 0 for). We will also still update the weights for our “positive” word (which is the word “quick” in our current example). The “negative samples” (that is, the 5 output words that we’ll train to output 0) are chosen using a “unigram distribution”. Essentially, the probability for selecting a word as a negative sample is related to its frequency, with more frequent words being more likely to be selected as negative samples.\*\*
7. [**Chris mccormick on negative sampling and hierarchical soft max**](https://www.youtube.com/watch?v=pzyIWCelt\_E) **training, i.e., huffman binary tree for the vocabulary, learning internal tree nodes ie.,, the path as the probability vector instead of having len(vocabulary) neurons.**
8. [**Great W2V tutorial**](https://towardsdatascience.com/word2vec-skip-gram-model-part-1-intuition-78614e4d6e0b)
9. **Another** [**gensim-based w2v tutorial**](http://kavita-ganesan.com/gensim-word2vec-tutorial-starter-code/)**, with starter code and some usage examples of similarity**
10. [**Clustering using gensim word2vec**](http://ai.intelligentonlinetools.com/ml/k-means-clustering-example-word2vec/)
11. [**Yet another w2v medium explanation**](https://towardsdatascience.com/word-embeddings-exploration-explanation-and-exploitation-with-code-in-python-5dac99d5d795)
12. **Mean w2v**
13. **Sequential w2v embeddings.**
14. [**Negative sampling, why does it work in w2v - didnt read**](https://www.quora.com/How-does-negative-sampling-work-in-Word2vec-models)
15. [**Semantic contract using w2v/ft - he chose a good food category and selected words that worked best in order to find similar words to good bad etc. lior magen**](https://groups.google.com/forum/#!topic/gensim/wh7B00cc80w)
16. [**Semantic contract, syn-antonym DS, using w2v, a paper that i havent read**](http://anthology.aclweb.org/P16-2074) **yet but looks promising**
17. [**Amazing w2v most similar tutorial, examples for vectors, misspellings, semantic contrast and relations that may or may not be captured in the network.**](https://quomodocumque.wordpress.com/2016/01/15/messing-around-with-word2vec/)
18. [**Followup tutorial about genderfying words using ‘he’ ‘she’ similarity**](https://quomodocumque.wordpress.com/2016/01/15/gendercycle-a-dynamical-system-on-words/)
19. [**W2v Analogies using predefined anthologies of the**](https://gist.github.com/kylemcdonald/9bedafead69145875b8c) **form x:y:**:a:**b, plus code, plus insights of why it works and doesn't. presence : absence :: happy : unhappy absence : presence :: happy : proud abundant : scarce :: happy : glad refuse : accept :: happy : satisfied accurate : inaccurate :: happy : disappointed admit : deny :: happy : delighted never : always :: happy : Said\_Hirschbeck modern : ancient :: happy : ecstatic**
20. [**Nlpforhackers on bow, w2v embeddings with code on how to use**](https://nlpforhackers.io/word-embeddings/)
21. [**Hebrew word embeddings with w2v, ron shemesh, on wiki/twitter**](https://drive.google.com/drive/folders/1qBgdcXtGjse9Kq7k1wwMzD84HH\_Z8aJt?fbclid=IwAR03PeUTGCgluILOQ6EaMR7AgkcRux5rs6Z8HEgWMRvFAwLGqb7-7bznbxM)

**GLOVE**

1. [**W2v vs glove vs fasttext, in terms of overfitting and what is the idea behind**](https://www.kaggle.com/sbongo/do-pretrained-embeddings-give-you-the-extra-edge)
2. [**W2v against glove performance**](http://dsnotes.com/post/glove-enwiki/) **comparison - glove wins in % and time.**
3. [**How glove and w2v work, but the following has a very good description**](https://geekyisawesome.blogspot.com/2017/03/word-embeddings-how-word2vec-and-glove.html) **- “GloVe takes a different approach. Instead of extracting the embeddings from a neural network that is designed to perform a surrogate task (predicting neighbouring words), the embeddings are optimized directly so that the dot product of two word vectors equals the log of the number of times the two words will occur near each other (within 5 words for example). For example if "dog" and "cat" occur near each other 10 times in a corpus, then vec(dog) dot vec(cat) = log(10). This forces the vectors to somehow encode the frequency distribution of which words occur near them.”**
4. [**Glove vs w2v, concise explanation**](https://www.quora.com/What-is-the-difference-between-fastText-and-GloVe/answer/Ajit-Rajasekharan)

### **FastText**

1. [**Fasttext - using fast text and upsampling/oversapmling on twitter data**](https://medium.com/@media\_73863/fasttext-sentiment-analysis-for-tweets-a-straightforward-guide-9a8c070449a2)
2. [**A great youtube lecture 9m about ft, rarity, loss, class tree speedup**](https://www.youtube.com/watch?v=4l\_At3oalzk) _\*\*_
3. [**A thorough tutorial about what is FT and how to use it, performance, pros and cons.**](https://www.analyticsvidhya.com/blog/2017/07/word-representations-text-classification-using-fasttext-nlp-facebook/)
4. [**Docs**](https://fasttext.cc/blog/2016/08/18/blog-post.html)
5. [**Medium: word embeddings with w2v and fast text in gensim**](https://towardsdatascience.com/word-embedding-with-word2vec-and-fasttext-a209c1d3e12c) **, data cleaning and word similarity**
6. **Gensim -** [**fasttext docs**](https://radimrehurek.com/gensim/models/fasttext.html)**, similarity, analogies**
7. [**Alternative to gensim**](https://github.com/plasticityai/magnitude#benchmarks-and-features) **- promises speed and out of the box support for many embeddings.**
8. [**Comparison of usage w2v fasttext**](http://ai.intelligentonlinetools.com/ml/fasttext-word-embeddings-text-classification-python-mlp/)
9. [**Using gensim fast text - recommendation against using the fb version**](https://blog.manash.me/how-to-use-pre-trained-word-vectors-from-facebooks-fasttext-a71e6d55f27)
10. [**A comparison of w2v vs ft using gensim**](https://rare-technologies.com/fasttext-and-gensim-word-embeddings/) **- “Word2Vec embeddings seem to be slightly better than fastText embeddings at the semantic tasks, while the fastText embeddings do significantly better on the syntactic analogies. Makes sense, since fastText embeddings are trained for understanding morphological nuances, and most of the syntactic analogies are morphology based.**
    1. [**Syntactic**](https://stackoverflow.com/questions/48356421/what-is-the-difference-between-syntactic-analogy-and-semantic-analogy) **means syntax, as in tasks that have to do with the structure of the sentence, these include tree parsing, POS tagging, usually they need less context and a shallower understanding of world knowledge**
    2. [**Semantic**](https://stackoverflow.com/questions/48356421/what-is-the-difference-between-syntactic-analogy-and-semantic-analogy) **tasks mean meaning related, a higher level of the language tree, these also typically involve a higher level understanding of the text and might involve tasks s.a. question answering, sentiment analysis, etc...**
    3. **As for analogies, he is referring to the mathematical operator like properties exhibited by word embedding, in this context a syntactic analogy would be related to plurals, tense or gender, those sort of things, and semantic analogy would be word meaning relationships s.a. man + queen = king, etc... See for instance** [**this article**](http://www.aclweb.org/anthology/W14-1618) **(and many others)**
11. [**Skip gram vs CBOW**](https://www.quora.com/What-are-the-continuous-bag-of-words-and-skip-gram-architectures)

![](https://lh5.googleusercontent.com/lnuntHia-uXCNiGbmw0bWYski3uPkeryHj3Rf8si9E9GUCyUi1aXsMv3sKgY\_YLjqWbRRWjGLzCZymjWwRlMquDTsQdcd05PcSJ74ZEOmd1QW59SaZlC3XCzTGpyPdPjVDUljOvG)

1. [**Paper**](http://workshop.colips.org/dstc6/papers/track2\_paper18\_zhuang.pdf) **on fasttext vs glove vs w2v on a single DS, performance comparison. Ft wins by a small margin**
2. [**Medium on w2v/fast text ‘most similar’ words with code**](https://towardsdatascience.com/word-embedding-with-word2vec-and-fasttext-a209c1d3e12c)
3. [**keras/tf code for a fast text implementation**](http://debajyotidatta.github.io/nlp/deep/learning/word-embeddings/2016/09/28/fast-text-and-skip-gram/)
4. [**Medium on fast text and imbalance data**](https://medium.com/@yeyrama/fasttext-and-imbalanced-classification-1f9543f9e0ce)
5. **Medium on universal** [**Sentence encoder, w2v, Fast text for sentiment**](https://medium.com/@jatinmandav3/opinion-mining-sometimes-known-as-sentiment-analysis-or-emotion-ai-refers-to-the-use-of-natural-874f369194c0) **with code.**

### **SENTENCE EMBEDDING**

#### **Sense2vec**

1. [**Blog**](https://explosion.ai/blog/sense2vec-with-spacy)**,** [**github**](https://github.com/explosion/sense2vec)**: Using spacy or not, with w2v using POS/ENTITY TAGS to find similarities.based on reddit. “We follow Trask et al in adding part-of-speech tags and named entity labels to the tokens. Additionally, we merge named entities and base noun phrases into single tokens, so that they receive a single vector.”**
2. **>>> model.similarity('fair\_game|NOUN', 'game|NOUN') 0.034977455677555599 >>> model.similarity('multiplayer\_game|NOUN', 'game|NOUN') 0.54464530644393849**

#### **SENT2VEC aka “skip-thoughts”**

1. [**Gensim implementation of sent2vec**](https://rare-technologies.com/sent2vec-an-unsupervised-approach-towards-learning-sentence-embeddings/) **- usage examples, parallel training, a detailed comparison against gensim doc2vec**
2. [**Git implementation**](https://github.com/ryankiros/skip-thoughts)
3. [**Another git - worked**](https://github.com/epfml/sent2vec)

#### **USE - Universal sentence encoder**

1. [**Git notebook, usage and sentence similarity benchmark / visualization**](https://github.com/tensorflow/hub/blob/master/examples/colab/semantic\_similarity\_with\_tf\_hub\_universal\_encoder.ipynb)

#### **BERT+W2V**

1. [**Sentence similarity**](https://towardsdatascience.com/how-to-compute-sentence-similarity-using-bert-and-word2vec-ab0663a5d64)

### **PARAGRAPH2Vec**

1. [**Paragraph2VEC by stanford**](https://cs.stanford.edu/\~quocle/paragraph\_vector.pdf)

### **Doc2Vec**

1. [**Shuffle before training each**](https://groups.google.com/forum/#!topic/gensim/IVQBUF5n6aI) **epoch in d2v in order to fight overfitting**
