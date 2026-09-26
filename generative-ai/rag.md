# RAG

A language model only knows what it was trained on, so answering from your own documents means retrieving the right text first and generating from it. This page starts with practical retrieval-augmented generation (RAG) tutorials and ways to improve the pipeline, then moves to Graph RAG, where the retrieved context comes from a knowledge graph.

The same notes are in [GenAI Applications](genai-applications.md), [Search](../language-ai/search.md), [Tools](large-language-models-llms.md#tools), and [Vector databases](../data/engineering/lakes-and-warehouses.md#vector-databases).

## Tutorials

The first step is to build one pipeline end to end on real, messy documents. [RAG on Complex PDF using LlamaParse, Langchain and Groq](https://medium.com/the-ai-forum/rag-on-complex-pdf-using-llamaparse-langchain-and-groq-5b132bd1f9f3) is Plaban Nayak's walkthrough; it describes RAG as an approach that uses LLMs to automate knowledge search, synthesis, extraction, and planning from unstructured data, adding contextual information to LLM applications, and it lays out the components of the RAG data stack. Once a pipeline works, [Optimization methods for RAG](https://pub.towardsai.net/so-you-want-to-improve-your-rag-pipeline-28b0cfadbfd7) is "So, You Want To Improve Your RAG Pipeline", which points to an implementation of improving the pipeline with different indexing. Both tutorials end in a working bot, and [Q&A Bot Using Gen-AI](https://pub.towardsai.net/q-a-bot-using-gen-ai-2ab934b180af) (October 2024) is a retrieval bot built the way this page describes.

## Graph RAG

Plain retrieval returns chunks of text; Graph RAG builds a knowledge graph first so the model can reason over connections in the data.

The same notes are in [Knowledge Graphs](../language-ai/knowledge-graphs.md).

[Microsoft on GraphRAG: Unlocking LLM discovery on narrative private data](https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/) is the Microsoft Research post, by Brenda Potts, on using LLM-generated knowledge graphs to improve question answering over complex information, where it consistently outperforms baseline RAG. An easier explainer of how GraphRAG works used to sit here; that address no longer opens and is kept at the end of the page.

## Deprecated links

{% hint style="warning" %}
These links and images no longer work. The original wording is kept here. A same-resource copy, when one was checked, is used above.
{% endhint %}

- Understanding Graph RAG. This address no longer opens: https://towardsdatascience.com/an-easy-way-to-comprehend-how-graphrag-works-6d53f8b540d0
