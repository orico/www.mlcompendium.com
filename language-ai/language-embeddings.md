## LANGUAGE EMBEDDINGS

This section gathers transformer history and embedding survey material.

The same notes are in [Multi Language](multi-language.md).


This section gathers transformer history and embedding survey material.


<figure><img src="../.gitbook/assets/gimg-cb444fcbcf82.png" alt=""><figcaption><p>BERT</p><p>Credit: <a href="https://lh6.googleusercontent.com/aibqScGzh66aJK9E5Rho61W_pX8Kw82vJrrUkvRZrRN7vaRBOWnDOz0k29szquWdU3i4cwFFUj6b4-rPZvU2AUIlP5ouxwS7Kq2RwxDwFxtm9fpJZcnVXCMHY3SJ43FEsWj_GTcT">copied from the original hosted image</a>.</p></figcaption></figure>


### History

This section traces transformers, relative attention, and pretrained language models.


1. [Google’s intro to transformers and multi-head self attention](https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html)
2. [How self attention and relative positioning work](https://medium.com/@_init_/how-self-attention-with-relative-position-representations-works-28173b8c245a) (great!)
   1. Rnns are sequential, same word in diff position will have diff encoding due to the input from the previous word, which is inherently different.
   2. Attention without positional! Will have distinct (Same) encoding.
   3. Relative look at a window around each word and adds a distance vector in terms of how many words are before and after, which fixes the problem.
   4. <figure><img src="../.gitbook/assets/gimg-dd8433f09172.png" alt=""><figcaption><p>BERT</p><p>Credit: <a href="https://lh3.googleusercontent.com/XmFsG2XDB2sLXNkRwmsc90iPfXPBWDgr4AzO-u8lejinMcwb5XzTppAZ5oekBjUjIsJ8u8IBA83Z31bP3rgMjdkvq0qZAteTE2VvxSOa79AUH4KqsRQb0w1Eworanxxm7zFuo494">copied from the original hosted image</a>.</p></figcaption></figure>
   5. <figure><img src="../.gitbook/assets/gimg-ac55cd7c9ffa.png" alt=""><figcaption><p>BERT</p><p>Credit: <a href="https://lh6.googleusercontent.com/JNAgD9NAJQzXCtfZ3ekWddZ1m8nzgMwXoqoQ3rjLsKfHl2NdqVrdrYexDnXCUzik2ZYalllJhm7Hp5Zl1_L5EHumNGN0NAfFHHH0RM6gqBZc4bPkg7Bd4D5ea5gmV1_hXtMXW_9K">copied from the original hosted image</a>.</p></figcaption></figure>
   6. The authors hypothesized that precise relative position information is not useful beyond a certain distance.
   7. Clipping the maximum distance enables the model to generalize to sequence lengths not seen during training.
3. [From bert to albert](https://medium.com/@hamdan.hussam/from-bert-to-albert-pre-trained-langaug-models-5865aa5c3762)
4. [All the latest buzz algos](https://www.topbots.com/most-important-ai-nlp-research/#ai-nlp-paper-2018-12)
5. A [Summary of them](https://www.topbots.com/ai-nlp-research-pretrained-language-models/?utm_source=facebook&utm_medium=group_post&utm_campaign=pretrained)
6. [8 pretrained language embeddings](https://www.analyticsvidhya.com/blog/2019/03/pretrained-models-get-started-nlp/)
7. [Hugging face pytorch transformers](https://github.com/huggingface/pytorch-transformers)
8. [Hugging face nlp pretrained](https://huggingface.co/models?search=Helsinki-NLP%2Fopus-mt)

### Embedding Foundation Knowledge

This section compares sentence and word embedding families with notebooks.


This section compares sentence and word embedding families with notebooks.


1. Medium on Introduction into word embeddings, sentence embeddings, trends in the field. The Indian guy, [git](https://nbviewer.jupyter.org/github/dipanjanS/data_science_for_all/blob/master/tds_deep_transfer_learning_nlp_classification/Deep%20Transfer%20Learning%20for%20NLP%20-%20Text%20Classification%20with%20Universal%20Embeddings.ipynb) notebook, [his git](https://github.com/dipanjanS),
   1. Baseline Averaged Sentence Embeddings
   2. Doc2Vec
   3. Neural-Net Language Models (Hands-on Demo!)
   4. Skip-Thought Vectors
   5. Quick-Thought Vectors
   6. InferSent
   7. Universal Sentence Encoder
2. [Shay palachy on word embedding covering everything from bow to word/doc/sent/phrase.](https://medium.com/@shay.palachy/document-embedding-techniques-fed3e7a6a25d)
3. [Another intro, not as good as the one above](https://medium.com/huggingface/universal-word-sentence-embeddings-ce48ddc8fc3a)
4. Using sklearn vectorizer to create custom ones, i.e. a vectorizer that does preprocessing and tfidf and other things.
5. [TFIDF - n-gram based top weighted tfidf words](https://stackoverflow.com/questions/25217510/how-to-see-top-n-entries-of-term-document-matrix-after-tfidf-in-scikit-learn)

The same notes are in [TF-IDF](tf-idf.md).

6. [Gensim bi-gram phraser/phrases analyser/converter](https://radimrehurek.com/gensim/models/phrases.html)
7. [Countvectorizer, stemmer, lemmatization code tutorial](https://medium.com/@rnbrown/more-nlp-with-sklearns-countvectorizer-add577a0b8c8)
8. [Current 2018 best universal word and sentence embeddings -> elmo](https://medium.com/huggingface/universal-word-sentence-embeddings-ce48ddc8fc3a)
9. 5-part series on word embeddings, part 2, 3, 4 - cross lingual review, 5-future trends
10. [Word embedding posts](https://datawarrior.wordpress.com/2016/05/15/word-embedding-algorithms/)
11. [Facebook github for embedings called starspace](https://github.com/facebookresearch/StarSpace)
12. [Medium on Fast text / elmo etc](https://medium.com/huggingface/universal-word-sentence-embeddings-ce48ddc8fc3a)

### Language modeling

This section links language-model pretraining, BERT, and GPT-style resources.

The same notes are in [BERT](pretrained-language-models.md#bert), [ELMO](pretrained-language-models.md#elmo), [GPT2](pretrained-language-models.md#gpt2), [GPT3](pretrained-language-models.md#gpt3), and [ULMFIT](pretrained-language-models.md#ulmfit).


This section links language-model pretraining, BERT, and GPT-style resources.


1. Ruder on language modelling as the next imagenet - Language modelling, the last approach mentioned, has been shown to capture many facets of language relevant for downstream tasks, such as [long-term dependencies](https://arxiv.org/abs/1611.01368) , [hierarchical relations](https://arxiv.org/abs/1803.11138) , and [sentiment](https://arxiv.org/abs/1704.01444) . Compared to related unsupervised tasks such as skip-thoughts and autoencoding, [language modelling performs better on syntactic tasks even with less training data](https://openreview.net/forum?id=BJeYYeaVJ7).
2. A tutorial about w2v skipthought - with code!, specifically language modelling here is important - Our second method is training a language model to represent our sentences. A language model describes the probability of a text existing in a language. For example, the sentence “I like eating bananas” would be more probable than “I like eating convolutions.” We train a language model by slicing windows of n words and predicting what the next word will be in the text
3. [Universal language model fine tuning for text-classification](https://arxiv.org/abs/1801.06146)

The ULMFiT paper (Howard and Ruder) proposes pretraining a language model on a large corpus, then fine-tuning it on a target text-classification task with discriminative learning rates and gradual unfreezing, reporting strong results with limited labeled data.

4. ELMO - medium
5. [Bert](https://arxiv.org/abs/1810.04805v1) [python git](https://github.com/CyberZHG/keras-bert) - We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT representations can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications. BERT is conceptually simple and empirically powerful. It obtains new state-of-the-art results on eleven natural language processing tasks.

<figure><img src="../.gitbook/assets/gimg-df19465b818d.png" alt=""><figcaption><p>BERT</p><p>Credit: <a href="https://lh4.googleusercontent.com/anFY63RxhdYt82bb_XUGDLRUmj2vuR1I0iJye66cOqgC2gQegXVf2ibkC64LRPIfgUj8Brl7VYUFfxw3gG0KBnwTuqJ2NCohd6mi9YzCkZmHGuDz1QxXl7JUtMv5BpiBJXGnC-Zc">copied from the original hosted image</a>.</p></figcaption></figure>
6. [Open.ai on language modelling](https://blog.openai.com/language-unsupervised/) - We’ve obtained state-of-the-art results on a suite of diverse language tasks with a scalable, task-agnostic system, which we’re also releasing. Our approach is a combination of two existing ideas: [transformers](https://arxiv.org/abs/1706.03762) and [unsupervised pre-training](https://arxiv.org/abs/1511.01432). [READ PAPER](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf), [VIEW CODE](https://github.com/openai/finetune-transformer-lm).
7. Scikit-learn inspired model finetuning for natural language processing.

[finetune](https://finetune.indico.io/#module-finetune) ships with a pre-trained language model from [“Improving Language Understanding by Generative Pre-Training”](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) and builds off the [OpenAI/finetune-language-model repository](https://github.com/openai/finetune-transformer-lm).

1. Did not fully read - [The annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) - jupyter on transformer with annotation
2. Medium on [Dissecting Bert](https://medium.com/dissecting-bert/dissecting-bert-part-1-d3c3d495cdb3), [appendix](https://medium.com/dissecting-bert/dissecting-bert-appendix-the-decoder-3b86f66b0e5f)
3. Medium on distilling 6 patterns from bert

### Embedding spaces

This section compares word and sentence embedding algorithms and benchmarks.


This section compares word and sentence embedding algorithms and benchmarks.


1. [A good overview of sentence embedding methods](http://mlexplained.com/2017/12/28/an-overview-of-sentence-embedding-methods/) - w2v ft s2v skip, d2v
2. [A very good overview of word embeddings](http://sanjaymeena.io/tech/word-embeddings/)
3. Intro to word embeddings - lots of images
4. [A very long and extensive thesis about embeddings](http://ad-publications.informatik.uni-freiburg.de/theses/Bachelor_Jon_Ezeiza_2017.pdf)
5. [Sent2vec by gensim](https://rare-technologies.com/sent2vec-an-unsupervised-approach-towards-learning-sentence-embeddings/) - sentence embedding is defined as the average of the source word embeddings of its constituent words. This model is furthermore augmented by also learning source embeddings for not only unigrams but also n-grams of words present in each sentence, and averaging the n-gram embeddings along with the words
6. [Sent2vec vs fasttext - with info about s2v parameters](https://github.com/epfml/sent2vec/issues/19)
7. [Wordrank vs fasttext vs w2v comparison](https://en.wikipedia.org/wiki/Automatic_summarization#TextRank_and_LexRank) - the better word similarity algorithm
8. [W2v vs glove vs sppmi vs svd by gensim](https://rare-technologies.com/making-sense-of-word2vec/)
9. [Medium on a gentle intro to d2v](https://medium.com/scaleabout/a-gentle-introduction-to-doc2vec-db3e8c0cce5e)
10. [Doc2vec tutorial by gensim](https://rare-technologies.com/doc2vec-tutorial/) - Doc2vec (aka paragraph2vec, aka sentence embeddings) modifies the word2vec algorithm to unsupervised learning of continuous representations for larger blocks of text, such as sentences, paragraphs or entire documents. - Most importantly this tutorial has crucial information about the implementation parameters that should be read before using it.
11. [Lbl2Vec](https://github.com/sebischair/Lbl2Vec), medium, is an algorithm for unsupervised document classification and unsupervised document retrieval. It automatically generates jointly embedded label, document and word vectors and returns documents of categories modeled by manually predefined keywords.
12. [Git for word embeddings - taken from mastery’s nlp course](https://github.com/IshayTelavivi/nlp_crash_course)
13. [Skip-thought -](http://mlexplained.com/2017/12/28/an-overview-of-sentence-embedding-methods/) [git](https://github.com/ryankiros/skip-thoughts) - Where word2vec attempts to predict surrounding words from certain words in a sentence, skip-thought vector extends this idea to sentences: it predicts surrounding sentences from a given sentence. NOTE: Unlike the other methods, skip-thought vectors require the sentences to be ordered in a semantically meaningful way. This makes this method difficult to use for domains such as social media text, where each snippet of text exists in isolation.
14. [Fastsent](http://mlexplained.com/2017/12/28/an-overview-of-sentence-embedding-methods/) - Skip-thought vectors are slow to train. FastSent attempts to remedy this inefficiency while expanding on the core idea of skip-thought: that predicting surrounding sentences is a powerful way to obtain distributed representations. Formally, FastSent represents sentences as the simple sum of its word embeddings, making training efficient. The word embeddings are learned so that the inner product between the sentence embedding and the word embeddings of surrounding sentences is maximized. NOTE: FastSent sacrifices word order for the sake of efficiency, which can be a large disadvantage depending on the use-case.
15. Weighted sum of words - In this method, each word vector is weighted by the factor $$\frac{a}{a + p(w)}$$ where $$a$$ is a hyperparameter and $$p(w)$$ is the (estimated) word frequency. This is similar to tf-idf weighting, where more frequent terms are weighted downNOTE: Word order and surrounding sentences are ignored as well, limiting the information that is encoded.
16. [Infersent by facebook](https://github.com/facebookresearch/InferSent) - [paper](https://arxiv.org/abs/1705.02364) InferSent is a sentence embeddings method that provides semantic representations for English sentences. It is trained on natural language inference data and generalizes well to many different tasks. ABSTRACT: we show how universal sentence representations trained using the supervised data of the Stanford Natural Language Inference datasets can consistently outperform unsupervised methods like SkipThought vectors on a wide range of transfer tasks. Much like how computer vision uses ImageNet to obtain features, which can then be transferred to other tasks, our work tends to indicate the suitability of natural language inference for transfer learning to other NLP tasks.
17. [Universal sentence encoder - google](https://tfhub.dev/google/universal-sentence-encoder/1) - [notebook](https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/semantic_similarity_with_tf_hub_universal_encoder.ipynb#scrollTo=8OKy8WhnKRe_), git The Universal Sentence Encoder encodes text into high dimensional vectors that can be used for text classification, semantic similarity, clustering and other natural language tasks. The model is trained and optimized for greater-than-word length text, such as sentences, phrases or short paragraphs. It is trained on a variety of data sources and a variety of tasks with the aim of dynamically accommodating a wide variety of natural language understanding tasks. The input is variable length English text and the output is a 512 dimensional vector. We apply this model to the [STS benchmark](http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark) for semantic similarity, and the results can be seen in the [example notebook](https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/semantic_similarity_with_tf_hub_universal_encoder.ipynb) made available. The universal-sentence-encoder model is trained with a deep averaging network (DAN) encoder.
18. [Multi language universal sentence encoder](https://ai.googleblog.com/2019/07/multilingual-universal-sentence-encoder.html) - no hebrew
19. Pair2vec - [paper](https://arxiv.org/abs/1810.08854) - paper proposes new methods for learning and using embeddings of word pairs that implicitly represent background knowledge about such relationships. I.e., using p2v information with existing models to increase performance. Experiments show that our pair embeddings can complement individual word embeddings, and that they are perhaps capturing information that eludes the traditional interpretation of the Distributional Hypothesis
20. [Fast text python tutorial](http://ai.intelligentonlinetools.com/ml/fasttext-word-embeddings-text-classification-python-mlp/)

### WORD2VEC

This section is a long word2vec reading list with gensim and analogy examples.

The same notes are in [ZERO SHOT LEARNING](../problem-framing/n-shot-learning.md#zero-shot-learning).


This section is a long word2vec reading list with gensim and analogy examples.


1. Monitor [train loss](https://stackoverflow.com/questions/52038651/loss-does-not-decrease-during-training-word2vec-gensim) using callbacks for word2vec
2. Cleaning datasets using weighted w2v sentence encoding, then pca and isolation forest to remove outlier sentences.
3. [Removing ‘gender bias using pair mean pca](https://stackoverflow.com/questions/48019843/pca-on-word2vec-embeddings)
4. [KPCA w2v approach on a very small dataset](https://medium.com/@vishwanigupta/kpca-skip-gram-model-improving-word-embedding-a6a0cb7aad49), [similar git](https://github.com/niitsuma/wordca) for correspondence analysis, [paper](https://arxiv.org/abs/1605.05087)
5. [The best w2v/tfidf/bow/ embeddings post ever](https://www.analyticsvidhya.com/blog/2017/06/word-embeddings-count-word2veec/)
6. [Chris mccormick ml on w2v,](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/) [post #2](http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/) - negative sampling “Negative sampling addresses this by having each training sample only modify a small percentage of the weights, rather than all of them. With negative sampling, we are instead going to randomly select just a small number of “negative” words (let’s say 5) to update the weights for. (In this context, a “negative” word is one for which we want the network to output a 0 for). We will also still update the weights for our “positive” word (which is the word “quick” in our current example). The “negative samples” (that is, the 5 output words that we’ll train to output 0) are chosen using a “unigram distribution”. Essentially, the probability for selecting a word as a negative sample is related to its frequency, with more frequent words being more likely to be selected as negative samples.
7. [Chris mccormick on negative sampling and hierarchical soft max](https://www.youtube.com/watch?v=pzyIWCelt_E) training, i.e., huffman binary tree for the vocabulary, learning internal tree nodes ie.,, the path as the probability vector instead of having len(vocabulary) neurons.
8. Great W2V tutorial
9. Another [gensim-based w2v tutorial](http://kavita-ganesan.com/gensim-word2vec-tutorial-starter-code/), with starter code and some usage examples of similarity
10. [Clustering using gensim word2vec](http://ai.intelligentonlinetools.com/ml/k-means-clustering-example-word2vec/)
11. Yet another w2v medium explanation
12. Mean w2v
13. Sequential w2v embeddings.
14. [Negative sampling, why does it work in w2v - didnt read](https://www.quora.com/How-does-negative-sampling-work-in-Word2vec-models)
15. [Semantic contract using w2v/ft - he chose a good food category and selected words that worked best in order to find similar words to good bad etc. lior magen](https://groups.google.com/forum/#!topic/gensim/wh7B00cc80w)
16. [Semantic contract, syn-antonym DS, using w2v, a paper that i havent read](http://anthology.aclweb.org/P16-2074) yet but looks promising
17. [Amazing w2v most similar tutorial, examples for vectors, misspellings, semantic contrast and relations that may or may not be captured in the network.](https://quomodocumque.wordpress.com/2016/01/15/messing-around-with-word2vec/)
18. [Followup tutorial about genderfying words using ‘he’ ‘she’ similarity](https://quomodocumque.wordpress.com/2016/01/15/gendercycle-a-dynamical-system-on-words/)
19. [W2v Analogies using predefined anthologies of the](https://gist.github.com/kylemcdonald/9bedafead69145875b8c) form x:y::a:b, plus code, plus insights of why it works and doesn't. presence : absence :: happy : unhappy absence : presence :: happy : proud abundant : scarce :: happy : glad refuse : accept :: happy : satisfied accurate : inaccurate :: happy : disappointed admit : deny :: happy : delighted never : always :: happy : Said_Hirschbeck modern : ancient :: happy : ecstatic
20. [Nlpforhackers on bow, w2v embeddings with code on how to use](https://nlpforhackers.io/word-embeddings/)
21. [Hebrew word embeddings with w2v, ron shemesh, on wiki/twitter](https://drive.google.com/drive/folders/1qBgdcXtGjse9Kq7k1wwMzD84HH_Z8aJt)

GLOVE

1. [W2v vs glove vs fasttext, in terms of overfitting and what is the idea behind](https://www.kaggle.com/sbongo/do-pretrained-embeddings-give-you-the-extra-edge)
2. [W2v against glove performance](http://dsnotes.com/post/glove-enwiki/) comparison - glove wins in % and time.
3. [How glove and w2v work, but the following has a very good description](https://geekyisawesome.blogspot.com/2017/03/word-embeddings-how-word2vec-and-glove.html) - “GloVe takes a different approach. Instead of extracting the embeddings from a neural network that is designed to perform a surrogate task (predicting neighbouring words), the embeddings are optimized directly so that the dot product of two word vectors equals the log of the number of times the two words will occur near each other (within 5 words for example). For example if "dog" and "cat" occur near each other 10 times in a corpus, then vec(dog) dot vec(cat) = log(10). This forces the vectors to somehow encode the frequency distribution of which words occur near them.”
4. [Glove vs w2v, concise explanation](https://www.quora.com/What-is-the-difference-between-fastText-and-GloVe/answer/Ajit-Rajasekharan)

### FastText

This section covers fastText training, gensim usage, and syntactic versus semantic analogies.


This section covers fastText training, gensim usage, and syntactic versus semantic analogies.


1. [Fasttext - using fast text and upsampling/oversapmling on twitter data](https://medium.com/@media_73863/fasttext-sentiment-analysis-for-tweets-a-straightforward-guide-9a8c070449a2)
{% embed url="https://www.youtube.com/watch?v=4l_At3oalzk" %}

3. [A thorough tutorial about what is FT and how to use it, performance, pros and cons.](https://www.analyticsvidhya.com/blog/2017/07/word-representations-text-classification-using-fasttext-nlp-facebook/)
4. [Docs](https://fasttext.cc/blog/2016/08/18/blog-post.html)
5. Medium: word embeddings with w2v and fast text in gensim , data cleaning and word similarity
6. Gensim - [fasttext docs](https://radimrehurek.com/gensim/models/fasttext.html), similarity, analogies
7. [Alternative to gensim](https://github.com/plasticityai/magnitude#benchmarks-and-features) - promises speed and out of the box support for many embeddings.
8. [Comparison of usage w2v fasttext](http://ai.intelligentonlinetools.com/ml/fasttext-word-embeddings-text-classification-python-mlp/)
9. Using gensim fast text - recommendation against using the fb version
10. [A comparison of w2v vs ft using gensim](https://rare-technologies.com/fasttext-and-gensim-word-embeddings/) - “Word2Vec embeddings seem to be slightly better than fastText embeddings at the semantic tasks, while the fastText embeddings do significantly better on the syntactic analogies. Makes sense, since fastText embeddings are trained for understanding morphological nuances, and most of the syntactic analogies are morphology based.
    1. [Syntactic](https://stackoverflow.com/questions/48356421/what-is-the-difference-between-syntactic-analogy-and-semantic-analogy) means syntax, as in tasks that have to do with the structure of the sentence, these include tree parsing, POS tagging, usually they need less context and a shallower understanding of world knowledge
    2. [Semantic](https://stackoverflow.com/questions/48356421/what-is-the-difference-between-syntactic-analogy-and-semantic-analogy) tasks mean meaning related, a higher level of the language tree, these also typically involve a higher level understanding of the text and might involve tasks s.a. question answering, sentiment analysis, etc...
    3. As for analogies, he is referring to the mathematical operator like properties exhibited by word embedding, in this context a syntactic analogy would be related to plurals, tense or gender, those sort of things, and semantic analogy would be word meaning relationships s.a. man + queen = king, etc... See for instance [this article](http://www.aclweb.org/anthology/W14-1618) (and many others)
11. [Skip gram vs CBOW](https://www.quora.com/What-are-the-continuous-bag-of-words-and-skip-gram-architectures)

<figure><img src="../.gitbook/assets/gimg-08bd2e68b40c.png" alt=""><figcaption><p>BERT</p><p>Credit: <a href="https://lh5.googleusercontent.com/lnuntHia-uXCNiGbmw0bWYski3uPkeryHj3Rf8si9E9GUCyUi1aXsMv3sKgY_YLjqWbRRWjGLzCZymjWwRlMquDTsQdcd05PcSJ74ZEOmd1QW59SaZlC3XCzTGpyPdPjVDUljOvG">copied from the original hosted image</a>.</p></figcaption></figure>


1. Paper on fasttext vs glove vs w2v on a single DS, performance comparison. Ft wins by a small margin
2. Medium on w2v/fast text ‘most similar’ words with code
3. [keras/tf code for a fast text implementation](http://debajyotidatta.github.io/nlp/deep/learning/word-embeddings/2016/09/28/fast-text-and-skip-gram/)
4. [Medium on fast text and imbalance data](https://medium.com/@yeyrama/fasttext-and-imbalanced-classification-1f9543f9e0ce)
5. Medium on universal [Sentence encoder, w2v, Fast text for sentiment](https://medium.com/@jatinmandav3/opinion-mining-sometimes-known-as-sentiment-analysis-or-emotion-ai-refers-to-the-use-of-natural-874f369194c0) with code.

### SENTENCE EMBEDDING

This section lists sentence-level embedding tools and benchmarks.


This section lists sentence-level embedding tools and benchmarks.


#### Sense2vec

Sense2vec augments tokens with POS or entity tags for finer similarity.


Sense2vec augments tokens with POS or entity tags for finer similarity.


1. [Blog](https://explosion.ai/blog/sense2vec-with-spacy), [github](https://github.com/explosion/sense2vec): Using spacy or not, with w2v using POS/ENTITY TAGS to find similarities.based on reddit. “We follow Trask et al in adding part-of-speech tags and named entity labels to the tokens. Additionally, we merge named entities and base noun phrases into single tokens, so that they receive a single vector.”
2. >>> model.similarity('fair_game|NOUN', 'game|NOUN') 0.034977455677555599 >>> model.similarity('multiplayer_game|NOUN', 'game|NOUN') 0.54464530644393849

#### SENT2VEC aka “skip-thoughts”

This section links sent2vec and skip-thought implementations.


This section links sent2vec and skip-thought implementations.


1. [Gensim implementation of sent2vec](https://rare-technologies.com/sent2vec-an-unsupervised-approach-towards-learning-sentence-embeddings/) - usage examples, parallel training, a detailed comparison against gensim doc2vec
2. [Git implementation](https://github.com/ryankiros/skip-thoughts)
3. [Another git - worked](https://github.com/epfml/sent2vec)

#### USE - Universal sentence encoder

This section points at Universal Sentence Encoder notebooks.


This section points at Universal Sentence Encoder notebooks.


1. Git notebook, usage and sentence similarity benchmark / visualization

#### BERT+W2V

This section mixes BERT and word2vec for sentence similarity.


This section mixes BERT and word2vec for sentence similarity.


1. Sentence similarity

### PARAGRAPH2Vec

This section links Stanford paragraph-vector material.


This section links Stanford paragraph-vector material.


1. [Paragraph2VEC by stanford](https://cs.stanford.edu/~quocle/paragraph_vector.pdf)

### Doc2Vec

This section notes doc2vec training tips such as shuffling each epoch.


This section notes doc2vec training tips such as shuffling each epoch.


1. [Shuffle before training each](https://groups.google.com/forum/#!topic/gensim/IVQBUF5n6aI) epoch in d2v in order to fight overfitting

