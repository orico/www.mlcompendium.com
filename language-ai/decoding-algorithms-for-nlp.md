# Decoding Algorithms For NLP

A generative model gives a probability for every next token, and something still has to pick one. The page goes from a visual overview of greedy decoding, beam search, and sampling, to Hugging Face's practical guide to generating text, and then to shorter explanations of greedy and beam search in seq2seq models.

The same notes are in [Basics](../predictive-ml/audio-basics.md), [GPT](../generative-ai/large-language-models-llms.md), and [SEQ2SEQ SEQUENCE TO SEQUENCE](neural-nlp.md#seq2seq-sequence-to-sequence).

The overview to start with is (great): [greedy, beam search, pure sampling, top k sampling](https://medium.com/voice-tech-podcast/visualising-beam-search-and-other-decoding-algorithms-for-natural-language-generation-fbba7cba2c5b) visualises beam search and the other decoding algorithms for natural language generation, and it is great, by Katnoria.

Once the options are clear, the next step is running them. (great) [How to generate text by HuggingFace](https://huggingface.co/blog/how-to-generate) shows how to use the different decoding methods for language generation with Transformers.

Beam search is the method most worth understanding in detail. For what is [beam](https://angelina-yang.medium.com/what-is-beam-search-decoding-in-a-neural-machine-translation-model-adaab30c6579) what search, angelina yang explains beam search decoding in a neural machine translation model through example interview questions, with Dharti Dhami also credited in the note. [understanding greedy and beam](https://medium.com/@jessica_lopez/understanding-greedy-search-and-beam-search-98c1e3cd821d) is by Jessica lopez, on greedy search and beam search as the well-known algorithms behind generation tasks such as neural machine translation and automatic summarization. The same interview-style post is the pointer for [beam in seq2seq](https://angelina-yang.medium.com/what-is-beam-search-decoding-in-a-neural-machine-translation-model-adaab30c6579).
