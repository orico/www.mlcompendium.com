# Embedding

This page lists embedding methods, vector search tools, language models, and categorical or domain-specific *2vec techniques.

The same notes are in [Augmentation](../language-ai/augmentation.md), [Image embeddings and others](../predictive-ml/audio-algorithms.md#image-embeddings-and-others), [KERAS EMBEDDING LAYER](deep-neural-frameworks.md#keras-embedding-layer), [LDA2VEC](../language-ai/topics-modeling.md#lda2vec), [NLP embedding repositories](../language-ai/nlp.md#nlp-embedding-repositories), and [Recommender Systems](../ai-product/recommender-systems.md).


This page lists embedding methods, vector search tools, language models, and categorical or domain-specific *2vec techniques.


## Intro

This section starts with a from-scratch embedding tutorial.


This section starts with a from-scratch embedding tutorial.


(amazing) embeddings from the ground up Single Lunch

## VECTOR SIMILARITY SEARCH

This section lists libraries and managed services for approximate nearest neighbors.

The same notes are in [A metric learning reality check](../evals/evaluation-metrics.md#a-metric-learning-reality-check), [Question Answering](../language-ai/question-answering.md), [Search](../language-ai/search.md), and [Vector databases](../data/engineering/lakes-and-warehouses.md#vector-databases).


This section lists libraries and managed services for approximate nearest neighbors.


1. [Faiss](https://github.com/facebookresearch/faiss) - a library for efficient similarity search
2. [Benchmarking](https://github.com/erikbern/ann-benchmarks) - complete with almost everything imaginable
3. [Singlestore](https://www.singlestore.com/solutions/predictive-ml-ai/)
4. Elastic search - [dense vector](https://www.elastic.co/guide/en/elasticsearch/reference/7.6/query-dsl-script-score-query.html#vector-functions)
5. Google cloud vertex matching engine [NN search](https://cloud.google.com/blog/products/ai-machine-learning/vertex-matching-engine-blazing-fast-and-massively-scalable-nearest-neighbor-search)
   1. search
      1. Recommendation engines
      2. Search engines
      3. Ad targeting systems
      4. Image classification or image search
      5. Text classification
      6. Question answering
      7. Chat bots
   2. Features
      1. Low latency
      2. High recall
      3. managed
      4. Filtering
      5. scale
6. Pinecone - managed [vector similarity search](https://www.pinecone.io/) - Pinecone is a fully managed vector database that makes it easy to add vector search to production applications. No more hassles of benchmarking and tuning algorithms or building and maintaining infrastructure for vector search.
7. [Nmslib](https://github.com/nmslib/nmslib) ([benchmarked](https://github.com/erikbern/ann-benchmarks) - Benchmarks of approximate nearest neighbor libraries in Python) is a Non-Metric Space Library (NMSLIB): An efficient similarity search library and a toolkit for evaluation of k-NN methods for generic non-metric spaces.
8. scann,
9. [Vespa.ai](https://vespa.ai/) - Make AI-driven decisions using your data, in real time. At any scale, with unbeatable performance
10. [Weaviate](https://weaviate.io/developers/weaviate) - Weaviate is an [open source](https://github.com/semi-technologies/weaviate) vector search engine and vector database. Weaviate uses machine learning to vectorize and store data, and to find answers to natural language queries, or any other media type.
11. [Neural Search with BERT and Solr](https://dmitry-kan.medium.com/list/vector-search-e9b564d14274) - Indexing BERT vector data in Solr and searching with full traversal
12. [Fun With Apache Lucene and BERT Embeddings](https://medium.com/swlh/fun-with-apache-lucene-and-bert-embeddings-c2c496baa559) - This post goes much deeper -- to the similarity search algorithm on Apache Lucene level. It upgrades the code from 6.6 to 8.0
13. Speeding up BERT Search in Elasticsearch - Neural Search in Elasticsearch: from vanilla to KNN to hardware acceleration
14. Ask Me Anything about Vector Search - In the Ask Me Anything: Vector Search! session Max Irwin and Dmitry Kan discussed major topics of vector search, ranging from its areas of applicability to comparing it to good ol’ sparse search (TF-IDF/BM25), to its readiness for prime time and what specific engineering elements need further tuning before offering this to users.
15. [Search with BERT vectors in Solr and Elasticsearch](https://github.com/DmitryKey/bert-solr-search) - GitHub repository used for experiments with Solr and Elasticsearch using DBPedia abstracts comparing Solr, vanilla Elasticsearch, elastiknn enhanced Elasticsearch, OpenSearch, and GSI APU
16. Not All Vector Databases Are Made Equal - A detailed comparison of Milvus, Pinecone, Vespa, Weaviate, Vald, GSI and Qdrant
17. [Vector Podcast](https://dmitry-kan.medium.com/vector-podcast-e27d83ecd0be) - Podcast hosted by Dmitry Kan, interviewing the makers in the Vector / Neural Search industry. Available on YouTube, Spotify, Apple Podcasts and RSS
18. [Players in Vector Search: Video](https://dmitry-kan.medium.com/players-in-vector-search-video-2fd390d00d6) -Video recording and slides of the talk presented on London IR Meetup on the topic of players, algorithms, software and use cases in Vector Search
19. (paper) [Hybrid retrieval using search and semantic search](https://arxiv.org/abs/2210.11934)

## TOOLS

This section covers Flair and Hugging Face tooling for embeddings and NLP.


This section covers Flair and Hugging Face tooling for embeddings and NLP.


### FLAIR

Flair provides NER, PoS tagging, text classification, and combined embeddings.

The same notes are in [Named Entity Recognition (NER)](../language-ai/named-entity-recognition-ner.md).


Flair provides NER, PoS tagging, text classification, and combined embeddings.


1. Name-Entity Recognition (NER): It can recognise whether a word represents a person, location or names in the text.
2. Parts-of-Speech Tagging (PoS): Tags all the words in the given text as to which “part of speech” they belong to.
3. Text Classification: Classifying text based on the criteria (labels)
4. Training Custom Models: Making our own custom models.
5. It comprises of popular and state-of-the-art word embeddings, such as GloVe, BERT, ELMo, Character Embeddings, etc. There are very easy to use thanks to the Flair API
6. Flair’s interface allows us to combine different word embeddings and use them to embed documents. This in turn leads to a significant uptick in results
7. ‘Flair Embedding’ is the signature embedding provided within the Flair library. It is powered by contextual string embeddings. We’ll understand this concept in detail in the next section
8. Flair supports a number of languages – and is always looking to add new ones

### HUGGING FACE

This section links Hugging Face model hubs, tutorials, and emotion-classification walkthroughs.


This section links Hugging Face model hubs, tutorials, and emotion-classification walkthroughs.


1. [Git](https://github.com/huggingface/transformers)
2. [Hugging face pytorch transformers](https://github.com/huggingface/pytorch-transformers)
3. [Hugging face nlp pretrained](https://huggingface.co/models?search=Helsinki-NLP%2Fopus-mt)
4. [hugging face on emotions](https://medium.com/huggingface/understanding-emotions-from-keras-to-pytorch-3ccb61d5a983)
   1. how to make a custom pyTorch LSTM with custom activation functions,
   2. how the PackedSequence object works and is built,
   3. how to convert an attention layer from Keras to pyTorch,
   4. how to load your data in pyTorch: DataSets and smart Batching,
   5. how to reproduce Keras weights initialization in pyTorch.
5. A [thorough tutorial on bert](http://mccormickml.com/2019/07/22/BERT-fine-tuning/), fine tuning using hugging face transformers package. [Code](https://colab.research.google.com/drive/1Y4o3jh3ZH70tl6mCd76vz_IxX23biCPP)

Youtube [ep1](https://www.youtube.com/watch?v=FKlPCK1uFrc), [2](https://www.youtube.com/watch?v=zJW57aCBCTk), [3](https://www.youtube.com/watch?v=x66kkDnbzi4), [3b](https://www.youtube.com/watch?v=Hnvb9b7a_Ps),

## Embedding Models

This section groups specialized embedding approaches beyond plain word vectors.


This section groups specialized embedding approaches beyond plain word vectors.


### Cat2vec

This section covers categorical encoding and cat2vec-style entity embeddings.


This section covers categorical encoding and cat2vec-style entity embeddings.


1. Part1: Label encoder/ ordinal, One hot, one hot with a rare bucket, hash
2. Part2: cat2vec using w2v, and entity embeddings for categorical data

<figure><img src="../.gitbook/assets/gimg-aee192aa5736.png" alt=""><figcaption><p>Cat2vec</p><p>Credit: <a href="https://lh6.googleusercontent.com/BJjrzp0YPmsy2_OKecufELzNU_AO2I2kSAx9ekSbGmGYNJ27AGkbdhwPv45iMVub_6q0AHF91N6BYdxA4l-eAUspOIat-QMU8xHQrSYYpWmu7TEO8NmRPIcrPItwq1TgkJN-LTd3">copied from the original hosted image</a>.</p></figcaption></figure>


### ENTITY EMBEDDINGS

This section links entity embeddings for tabular and Kaggle-style data.

The same notes are in [Deep Neural Tabular](deep-neural-tabular.md).


This section links entity embeddings for tabular and Kaggle-style data.


1. Star - [General purpose embedding paper with code somewhere](https://arxiv.org/pdf/1709.03856.pdf)
2. [Using embeddings on tabular data, specifically categorical - introduction](http://www.fast.ai/2018/04/29/categorical-embeddings/), using fastai without limiting ourselves to pytorch - the material from this post is covered in much more detail starting around 1:59:45 in the Lesson 3 video and continuing in Lesson 4 of our free, online [Practical Deep Learning for Coders](http://course.fast.ai/) course. To see example code of how this approach can be used in practice, check out our Lesson 3 jupyter notebook. Perhaps Saturday and Sunday have similar behavior, and maybe Friday behaves like an average of a weekend and a weekday. Similarly, for zip codes, there may be patterns for zip codes that are geographically near each other, and for zip codes that are of similar socio-economic status. The jupyter notebook doesn't seem to have the embedding example they are talking about.
3. [Rossman on kaggle](http://blog.kaggle.com/2016/01/22/rossmann-store-sales-winners-interview-3rd-place-cheng-gui/), used entity-embeddings, [here](https://www.kaggle.com/c/rossmann-store-sales/discussion/17974), [github](https://github.com/entron/entity-embedding-rossmann), [paper](https://arxiv.org/abs/1604.06737)
4. Medium on rossman - good
5. [Embedder](https://github.com/dkn22/embedder) - git code for a simplified entity embedding above.
6. Finally what they do is label encode each feature using labelEncoder into an int-based feature, then push each feature into its own embedding layer of size 1 with an embedding size defined by a rule of thumb (so it seems), merge all layers, train a synthetic regression/classification and grab the weights of the corresponding embedding layer.
7. [Entity2vec](https://github.com/ot/entity2vec)
8. [Categorical using keras](https://medium.com/@satnalikamayank12/on-learning-embeddings-for-categorical-data-using-keras-165ff2773fc9)

### ALL2VEC EMBEDDINGS

This section collects miscellaneous *2vec repositories and competitions.


This section collects miscellaneous *2vec repositories and competitions.


1. [ALL ???-2-VEC ideas](https://github.com/MaxwellRebo/awesome-2vec)
2. Fast.ai [post](http://www.fast.ai/2018/04/29/categorical-embeddings/) regarding embedding for tabular data, i.e., cont and categorical data
3. Entity embedding for categorical data + notebook
4. [Kaggle taxi competition + code](http://blog.kaggle.com/2015/07/27/taxi-trajectory-winners-interview-1st-place-team-%F0%9F%9A%95/)
5. [Ross man competition - entity embeddings, code missing](http://blog.kaggle.com/2016/01/22/rossmann-store-sales-winners-interview-3rd-place-cheng-gui/) +[alternative code](https://github.com/entron/entity-embedding-rossmann)
6. [CODE TO CREATE EMBEDDINGS straight away, based onthe ideas by cheng guo in keras](https://github.com/dkn22/embedder)
7. [PIN2VEC - pinterest embeddings using the same idea](https://medium.com/the-graph/applying-deep-learning-to-related-pins-a6fee3c92f5e)
8. [Tweet2Vec](https://github.com/soroushv/Tweet2Vec) - code in theano, [paper](https://dl.acm.org/citation.cfm?doid=2911451.2914762).
9. [Clustering](https://github.com/svakulenk0/tweet2vec_clustering) of tweet2vec, [paper](https://arxiv.org/abs/1703.05123)
10. Paper: [Character neural embeddings for tweet clustering](https://arxiv.org/pdf/1703.05123.pdf)
11. Diff2vec - might be useful on social network graphs, paper, [code](https://github.com/benedekrozemberczki/diff2vec)
12. emoji 2vec (below)
13. [Char2vec](https://hackernoon.com/chars2vec-character-based-language-model-for-handling-real-world-texts-with-spelling-errors-and-a3e4053a147d) [Git](https://github.com/IntuitionEngineeringTeam/chars2vec), similarity measure for words with types. [https://arxiv.org/abs/1708.00524](https://arxiv.org/abs/1708.00524)

EMOJIS

1. 1. [Deepmoji](http://datadrivenjournalism.net/featured_projects/deepmoji_using_emojis_to_teach_ai_about_emotions),
2. [hugging face on emotions](https://medium.com/huggingface/understanding-emotions-from-keras-to-pytorch-3ccb61d5a983)
   1. how to make a custom pyTorch LSTM with custom activation functions,
   2. how the PackedSequence object works and is built,
   3. how to convert an attention layer from Keras to pyTorch,
   4. how to load your data in pyTorch: DataSets and smart Batching,
   5. how to reproduce Keras weights initialization in pyTorch.
3. [Another great emoji paper, how to get vector representations from](https://aclweb.org/anthology/S18-1039)
4. [3. What can we learn from emojis (deep moji)](https://www.media.mit.edu/posts/what-can-we-learn-from-emojis/)
5. [Learning millions of](https://arxiv.org/pdf/1708.00524.pdf) for emoji, sentiment, sarcasm, [medium](https://medium.com/@bjarkefelbo/what-can-we-learn-from-emojis-6beb165a5ea0)
6. [EMOJI2VEC](https://tech.instacart.com/deep-learning-with-emojis-not-math-660ba1ad6cdc) - medium article with keras code, a[nother paper on classifying tweets using emojis](https://arxiv.org/abs/1708.00524)
7. [Group2vec](https://github.com/cerlymarco/MEDIUM_NoteBook/tree/master/Group2Vec) git and medium, which is a multi input embedding network using a-f below. plus two other methods that involve groupby and applying entropy and join/countvec per class. Really interesting
   1. Initialize embedding layers for each categorical input;
   2. For each category, compute dot-products among other embedding representations. These are our ‘groups’ at the categorical level;
   3. Summarize each ‘group’ adopting an average pooling;
   4. Concatenate ‘group’ averages;
   5. Apply regularization techniques such as BatchNormalization or Dropout;
   6. Output probabilities.

