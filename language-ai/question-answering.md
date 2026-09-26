# Question Answering

A question answering system takes a question and returns an answer from text, or from an image, and most of the work is in how the answer is found. The notes start with question answering over images, then walk through building a text Q&A model from scratch, and end with retrieval-backed systems that look answers up in a vector search engine.

Questions do not have to be about text. [Pythia, qna for images](https://github.com/facebookresearch/pythia) is Facebook AI Research's repository, now the modular framework for vision and language multimodal research.

For text, the usual starting point is a model trained on SQuAD, the Stanford Question Answering Dataset. [Building a Q&A system part 1](https://medium.com/data-science/building-a-question-answering-system-part-1-9388aadff507) is Alvira Swalin's end-of-master's project at USF, a question-answering model built from scratch on SQuAD. [Building a Q&A model](https://medium.com/data-science/nlp-building-a-question-answering-model-ed0529a68c54) is Priya Dwivedi's final project from Stanford's CS224N course, NLP through Deep Learning, and covers the main building blocks of a question answering model on SQuAD. [Vidhya on Q&A](https://medium.com/analytics-vidhya/how-i-build-a-question-answering-model-3548878d5db2) is Martin Decombarieu's account of building one, starting from the plain definition: a computer program that answers the questions you ask.

A model that reads one passage at a time still needs a way to find the passage. [Q&A system using](https://medium.com/voice-tech-podcast/building-an-intelligent-qa-system-with-nlp-and-milvus-75b496702490) NLP and Milvus is the Milvus team's intelligent QA system, and [milvus - An open source embedding vector similarity search engine powered by Faiss, NMSLIB and Annoy](https://github.com/milvus-io/milvus) is the engine itself, now described as a high-performance, cloud-native vector database built for scalable vector ANN search.

The same notes are in [VECTOR SIMILARITY SEARCH](../deep-learning/representations.md#vector-similarity-search).

The last note comes back to the model side of the system. [Q&A system](https://medium.com/@akshaynavalakha/nlp-question-answering-system-f05825ef35c8) covers the basic building blocks of a QA system with deep learning, built as a modified bi-directional attention flow model for the same Stanford CS224N course.

## Deprecated links

{% hint style="warning" %}
These links and images no longer work. The original wording is kept here. A same-resource copy, when one was checked, is used above.
{% endhint %}

- Building a Q\&A system part 1 This address no longer opens: https://towardsdatascience.com/building-a-question-answering-system-part-1-9388aadff507
- Building a Q\&A model This address no longer opens: https://towardsdatascience.com/nlp-building-a-question-answering-model-ed0529a68c54
