# Tokenization

Before a language model can read a sentence, the text has to be cut into tokens, and the way it is cut decides what the model sees. The notes below start with byte pair encoding (BPE) and tiktoken, the tokenizer behind the OpenAI API, then move to the Hugging Face summary of tokenization algorithms, and end with a guide to the BERT tokenizer.

The same notes are in [BERT](pretrained-language-models.md#bert) and [GPT](../generative-ai/large-language-models-llms.md).

The OpenAI side comes first. [Tokenization In OpenAI API : Let’s Explore Tiktoken Library](https://medium.com/@basics.machinelearning/tokenization-in-openai-api-lets-explore-tiktoken-library-d02d3ce94b0a) explores Tiktoken, the open-source tool OpenAI built for tokenizing text. The library itself is on [Github](https://github.com/openai/tiktoken). Under it sits pair encoding: BPE for tokenization, where BPE began as a data compression algorithm.

BPE is only one of the algorithms in use, so the next stop is the wider survey. [Hugging Face on tokenization](https://huggingface.co/docs/transformers/tokenizer_summary) is the Hugging Face summary of tokenization algorithms, and it is really good.

The last step narrows back to a single model. [An Explanatory Guide to BERT Tokenizer](https://www.analyticsvidhya.com/blog/2021/09/an-explanatory-guide-to-bert-tokenizer/) is Pritesh's walk through the BERT tokenizer, written from the point that the input matters for transformers and the tokenizer libraries are crucial to getting it right.
