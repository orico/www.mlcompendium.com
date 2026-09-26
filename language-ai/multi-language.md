# Multi Language

A model trained on one language does not help users of another, unless every language shares one embedding space. The page starts with LASER's multilingual sentence embeddings, then cross-lingual language-model pretraining with XLM and XLM-R, and ends with Google's universal embedding space.

The same notes are in [LANGUAGE EMBEDDINGS](language-embeddings.md#language-embeddings) and [LANGUAGE TRANSLATION](language-detection-identification-generation-nld-nli-nlg.md#language-translation).

The sentence-level route comes first. [Fb’s laser](https://engineering.fb.com/ai-research/laser-multilingual-sentence-embeddings/) is the LASER toolkit, which performs zero-shot cross-lingual transfer with more than 90 languages and is now open source.

The pretraining route follows. [Xlm](https://github.com/facebookresearch/XLM) is the PyTorch original implementation of Cross-lingual Language Model Pretraining, and [xlm-r](https://ai.facebook.com/blog/-xlm-r-state-of-the-art-cross-lingual-understanding-through-self-supervision/) is Facebook AI open-sourcing XLM-R, a multilingual model that uses self-supervised training to reach state-of-the-art performance on four cross-lingual understanding benchmarks. Google universal embedding space is the last pointer, without its own link on this page.
