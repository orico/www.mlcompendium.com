# Knowledge Graphs

A knowledge graph stores entities as nodes and their relations as edges, and the question is how to build one from text and from structured sources. The page starts with automatic creation from text with spaCy, then reconciling your own data with outside knowledge, and ends with a Medium series on creating graphs, building them from structured sources, and the semantic models behind them.

The same notes are in [GenAI Applications](../generative-ai/genai-applications.md), [Graph RAG](../generative-ai/rag.md#graph-rag), and [Root Cause Effects (RCE/RCA)](../decision-intelligence/root-cause-effects-rce-rca.md).

The fastest start is text you already have. [Automatic creation of KG using spacy](https://medium.com/data-science/auto-generated-knowledge-graphs-92ca99a81121) is Chris Thornton's auto-generated knowledge graphs, built from entity pairs, people, organizations, places, and events, that can be traversed to uncover connections in unstructured data.

 Knowledge graphs can be constructed automatically from text using part-of-speech and dependency parsing. The extraction of entity pairs from grammatical patterns is fast and scalable to large amounts of text using NLP library SpaCy.

A graph built from your text still has to agree with what the rest of the world knows. [Medium on Reconciling your data and the world of knowledge graphs](https://medium.com/data-science/reconciling-your-data-and-the-world-with-knowledge-graphs-bce66b377b14) is Akash Tandon on that step, starting from knowledge as the core of any successful initiative.

For the full method, Giuseppe Futia's Medium Series goes in order. [Creating kg](https://medium.com/data-science/knowledge-graphs-at-a-glance-c9119130a9f0) is Knowledge Graphs at a Glance, on graphs as a core abstraction for putting human knowledge into intelligent systems, with nodes for real-world entities and edges for their relations. [Building from structured sources](https://medium.com/data-science/building-knowledge-graphs-from-structured-sources-346c56c9d40e) is the second part. [Semantic models](https://medium.com/data-science/semantic-models-for-constructing-knowledge-graphs-38c0a1df316a) covers the semantic models for constructing knowledge graphs, including the mapping from data sources to ontologies, and assumes the earlier introductory articles.

## Deprecated links

{% hint style="warning" %}
These links and images no longer work. The original wording is kept here. A same-resource copy, when one was checked, is used above.
{% endhint %}

- Automatic creation of KG using spacy **and networx** This address no longer opens: https://towardsdatascience.com/auto-generated-knowledge-graphs-92ca99a81121
- Medium on Reconciling your data and the world of knowledge graphs This address no longer opens: https://towardsdatascience.com/reconciling-your-data-and-the-world-with-knowledge-graphs-bce66b377b14
- Creating kg This address no longer opens: https://towardsdatascience.com/knowledge-graphs-at-a-glance-c9119130a9f0
- Building from structured sources This address no longer opens: https://towardsdatascience.com/building-knowledge-graphs-from-structured-sources-346c56c9d40e
- Semantic models This address no longer opens: https://towardsdatascience.com/semantic-models-for-constructing-knowledge-graphs-38c0a1df316a
