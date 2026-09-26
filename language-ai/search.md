# Search

Search over text breaks when the words in the query do not match the words in the document. The page starts with a BERT-based search engine that matches meaning instead of tokens, then lists the recurring semantic-search problems and the fixes for each.

The same notes are in [RAG](../generative-ai/rag.md), [Vector databases](../data/engineering/lakes-and-warehouses.md#vector-databases), and [VECTOR SIMILARITY SEARCH](../deep-learning/representations.md#vector-similarity-search).

The concrete example is a Bert [search engine](https://medium.com/data-science/covid-19-bert-literature-search-engine-4d06cdac08bd) built to organize the fast-growing pile of COVID-19 research papers. Bert cosine between paragraphs and question is the whole ranking step: embed both, and return the paragraphs closest to the question.

A single embedding model is only one piece of a search system. The rest of the system is semantic search, auto completion, filtering, augmentation, and scoring. The problems it has to handle are token matching, contextualization, query misunderstanding, image search, and the metric. The solutions are synonym generation, query autocompletion, alternate query generation, word and doc embedding, contextualization, ranking, ensemble, and multilingual search.

## Deprecated links

{% hint style="warning" %}
These links and images no longer work. The original wording is kept here. A same-resource copy, when one was checked, is used above.
{% endhint %}

- **Bert** search engine**, cosine between paragraphs and question.** This address no longer opens: https://towardsdatascience.com/covid-19-bert-literature-search-engine-4d06cdac08bd
