# TF-IDF

This page defines TF-IDF and notes sparse-text tricks that weight word vectors with IDF.

The same notes are in [Embedding Foundation Knowledge](../deep-learning/embedding.md#embedding-foundation-knowledge), [LSA (TFIDF + SVD)](topics-modeling.md#lsa-tfidf--svd), [Recommender Systems](../machine-learning/recommender-systems.md), and [TFIDF](../validation-and-evaluation/features.md#tfidf).

[TF-IDF](http://www.tfidf.com/) — how important is a word to a document in a corpus

$$TF(t) = \frac{\text{Number of times term } t \text{ appears in a document}}{\text{Total number of terms in the document}}.$$

Frequency of word in doc / all words in document (normalized bcz docs have diff sizes)

$$IDF(t) = \log_e\left(\frac{\text{Total number of documents}}{\text{Number of documents with term } t \text{ in it}}\right).$$

measures how important a term is

TF-IDF is TF*IDF

1. [A much clearer explanation plus python code](https://stevenloria.com/tf-idf/), [part 2](http://blog.christianperone.com/2011/10/machine-learning-text-feature-extraction-tf-idf-part-ii/)
2. [Get top tfidf keywords](https://stackoverflow.com/questions/34232190/scikit-learn-tfidfvectorizer-how-to-get-top-n-terms-with-highest-tf-idf-score)
3. [Print top features](https://gist.github.com/StevenMaude/ea46edc315b0f94d03b9)

Data sets:

1. [Fast text multilingual](https://github.com/facebookresearch/fastText/blob/main/docs/crawl-vectors.md)
2. [NLP embeddings](http://vectors.nlpl.eu/repository/)

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

1. [Multiply by TFIDF](https://medium.com/data-science/supercharging-word-vectors-be80ee5513d)
2. Enriching using lstm-next word (char or word-wise)
3. Using external wiktionary/pedia data for certain words, phrases
4. Finding clusters of relevant data and figuring out if you can enrich based on the content of the clusters
5. [Applying deep nlp methods without big data, i.e., sparseness](https://medium.com/data-science/lessons-learned-from-applying-deep-learning-for-nlp-without-big-data-d470db4f27bf)

## Deprecated links

{% hint style="warning" %}
These links and images no longer work. The original wording is kept here. A same-resource copy, when one was checked, is used above.
{% endhint %}

- Fast text multilingual This address no longer opens: https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md
- Multiply by TFIDF This address no longer opens: https://towardsdatascience.com/supercharging-word-vectors-be80ee5513d
- Applying deep nlp methods without big data, i.e., sparseness This address no longer opens: https://towardsdatascience.com/lessons-learned-from-applying-deep-learning-for-nlp-without-big-data-d470db4f27bf?_branch_match_id=584170448791192656
