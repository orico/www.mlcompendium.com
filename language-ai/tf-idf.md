# TF-IDF

This page defines TF-IDF and notes sparse-text tricks that weight word vectors with IDF.
Formulas for term frequency and inverse document frequency sit below, with sparse-text weighting tricks afterward.

The same notes are in [Embedding Foundation Knowledge](language-embeddings.md#embedding-foundation-knowledge), [LSA (TFIDF + SVD)](topics-modeling.md#lsa-tfidf--svd), [Recommender Systems](../ai-product/recommender-systems.md), and [TFIDF](#tfidf).

[TF-IDF](http://www.tfidf.com/) — how important is a word to a document in a corpus

$$TF(t) = \frac{\text{Number of times term } t \text{ appears in a document}}{\text{Total number of terms in the document}}.$$

Frequency of word in doc / all words in document (normalized bcz docs have diff sizes)

$$IDF(t) = \log_e\left(\frac{\text{Total number of documents}}{\text{Number of documents with term } t \text{ in it}}\right).$$

measures how important a term is

TF-IDF is TF*IDF

- Tutorial: Finding Important Words in Text Using TF-IDF | stevenloria.com, by Steven Loria. [A much clearer explanation plus python code](https://stevenloria.com/tf-idf/)
- Read the first part of this tutorial: Text feature extraction (tf-idf) - Part I, by Christian S. Perone. [part 2](http://blog.christianperone.com/2011/10/machine-learning-text-feature-extraction-tf-idf-part-ii/)
2. [Get top tfidf keywords](https://stackoverflow.com/questions/34232190/scikit-learn-tfidfvectorizer-how-to-get-top-n-terms-with-highest-tf-idf-score)
- Do TF-IDF with scikit-learn and print top features - tfidf_features.py. [Print top features](https://gist.github.com/StevenMaude/ea46edc315b0f94d03b9)

Data sets:

- Library for fast text representation and classification. [Fast text multilingual](https://github.com/facebookresearch/fastText/blob/main/docs/crawl-vectors.md)
- Nordic Language Processing Laboratory word embeddings repository. [NLP embeddings](http://vectors.nlpl.eu/repository/)

### Sparse textual content

This section is about weighting embeddings with IDF when text is sparse.

mean(IDF(i) * w2v word vectors (i)) with or without reducing PC1 from the whole w2 average (amir pupko)

```python
def mean_weighted_embedding(model, words, idf=1.0):
 if words:
 return np.mean(idf * model[words], axis=0)
 else:
 print('we have an empty list')
 return []

idf_mapping = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))

logs_sequences_df['idf_vectors'] = logs_sequences_df.message.apply(lambda x: [idf_mapping[token] for token in splitter(x)])

logs_sequences_df['mean_weighted_idf_w2v'] = [mean_weighted_embedding(ft, splitter(logs_sequences_df['message'].iloc[i]), 1 / np.array(logs_sequences_df['idf_vectors'].iloc[i]).reshape(-1,1)) for i in range(logs_sequences_df.shape[0])]
```

- [Multiply by TFIDF](https://medium.com/data-science/supercharging-word-vectors-be80ee5513d)
2. Enriching using lstm-next word (char or word-wise)
3. Using external wiktionary/pedia data for certain words, phrases
4. Finding clusters of relevant data and figuring out if you can enrich based on the content of the clusters
- [Applying deep nlp methods without big data, i.e., sparseness](https://medium.com/data-science/lessons-learned-from-applying-deep-learning-for-nlp-without-big-data-d470db4f27bf)

### **TFIDF**

This section notes TF-IDF vectorizer options and retrieval links.

The same notes are in [LSA (TFIDF + SVD)](topics-modeling.md#lsa-tfidf--svd) and [TF-IDF](tf-idf.md).

1. [Max\_features in tf idf](https://stackoverflow.com/questions/46118910/scikit-learn-vectorizer-max-features) -Sometimes it is not effective to transform the whole vocabulary, as the data may have some exceptionally rare words, which, if passed to TfidfVectorizer().fit(), will add unwanted dimensions to inputs in the future. One of the appropriate techniques in this case, for instance, would be to print out word frequences accross documents and then set a certain threshold for them. Imagine you have set a threshold of 50, and your data corpus consists of 100 words. After looking at the word frequences 20 words occur less than 50 times. Thus, you set max\_features=80 and you are good to go. If max\_features is set to None, then the whole corpus is considered during the TF-IDFtransformation. Otherwise, if you pass, say, 5 to max\_features, that would mean creating a feature matrix out of the most 5 frequent words accross text documents.
2. Understanding Term based retrieval - TFIDF Bm25
- This post will show you precisely how BM25 builds upon TF-IDF, what its parameters do, and why it is so effective, by Rudi Seitz. [understanding TFIDF and BM25](https://kmwllc.com/index.php/2020/03/20/understanding-tf-idf-and-bm-25/)
- Understanding Term based retrieval - TFIDF Bm25. This address no longer opens: https://towardsdatascience.com/understanding-term-based-retrieval-methods-in-information-retrieval-2be5eb3dde9f
## Deprecated links

{% hint style="warning" %}
These links and images no longer work. The original wording is kept here. A same-resource copy, when one was checked, is used above.
{% endhint %}

- Fast text multilingual This address no longer opens: https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md
- Multiply by TFIDF This address no longer opens: https://towardsdatascience.com/supercharging-word-vectors-be80ee5513d
- Applying deep nlp methods without big data, i.e., sparseness This address no longer opens: https://towardsdatascience.com/lessons-learned-from-applying-deep-learning-for-nlp-without-big-data-d470db4f27bf?_branch_match_id=584170448791192656
