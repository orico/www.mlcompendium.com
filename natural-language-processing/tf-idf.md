# TF-IDF

[**TF-IDF**](http://www.tfidf.com/) **- how important is a word to a document in a corpus**

**TF(t) = (Number of times term t appears in a document) / (Total number of terms in the document).**

**Frequency of word in doc / all words in document (normalized bcz docs have diff sizes)**

**IDF(t) = log\_e(Total number of documents / Number of documents with term t in it).**

**measures how important a term is**

**TF-IDF is TF\*IDF**\


1. [**A much clearer explanation plus python code**](https://stevenloria.com/tf-idf/)**,** [**part 2**](http://blog.christianperone.com/2011/10/machine-learning-text-feature-extraction-tf-idf-part-ii/)
2. [**Get top tfidf keywords**](https://stackoverflow.com/questions/34232190/scikit-learn-tfidfvectorizer-how-to-get-top-n-terms-with-highest-tf-idf-score)
3. [**Print top features**](https://gist.github.com/StevenMaude/ea46edc315b0f94d03b9)

**Data sets:**

1. [**Fast text multilingual**](https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md)
2. [**NLP embeddings**](http://vectors.nlpl.eu/repository/)

### **Sparse textual content**

1. **mean(IDF(i) \* w2v word vectors (i)) with or without reducing PC1 from the whole w2 average (amir pupko)**\
   \


**def mean\_weighted\_embedding(model, words, idf=1.0):**

&#x20;   **if words:**

&#x20;       **return np.mean(idf \* model\[words], axis=0)a**

&#x20;   **else:**

&#x20;       **print('we have an empty list')**

&#x20;       **return \[]**\


**idf\_mapping = dict(zip(vectorizer.get\_feature\_names(), vectorizer.idf\_))**&#x20;

**logs\_sequences\_df\['idf\_vectors'] = logs\_sequences\_df.message.apply(lambda x: \[idf\_mapping\[token] for token in splitter(x)])**

**logs\_sequences\_df\['mean\_weighted\_idf\_w2v'] = \[mean\_weighted\_embedding(ft, splitter(logs\_sequences\_df\['message'].iloc\[i]), 1 / np.array(logs\_sequences\_df\['idf\_vectors'].iloc\[i]).reshape(-1,1)) for i in range(logs\_sequences\_df.shape\[0])]**\
\


1. [**Multiply by TFIDF**](https://towardsdatascience.com/supercharging-word-vectors-be80ee5513d)
2. **Enriching using lstm-next word (char or word-wise)**
3. **Using external wiktionary/pedia data for certain words, phrases**
4. **Finding clusters of relevant data and figuring out if you can enrich based on the content of the clusters**
5. [**Applying deep nlp methods without big data, i.e., sparseness**](https://towardsdatascience.com/lessons-learned-from-applying-deep-learning-for-nlp-without-big-data-d470db4f27bf?\_branch\_match\_id=584170448791192656)
