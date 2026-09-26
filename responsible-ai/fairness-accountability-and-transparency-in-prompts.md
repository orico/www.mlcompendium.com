# Fairness, Accountability, and Transparency In Prompts

With large models, the prompt becomes both a source of bias and a tool for removing it, and the model can also state things that are simply untrue. This page first collects papers that debias models through prompts, then turns to LLM hallucinations.

The same notes are in [Fairness, Accountability, and Transparency](fairness-accountability-and-transparency.md).

## Debiasing using prompts

The debiasing methods on the previous page change the representation; the papers here change the prompt instead, across text, vision-language, and 3D generation.

In text, [MsPrompt: Multi-step Prompt Learning for Debiasing Few-shot Event Detection](https://arxiv.org/abs/2305.09335) targets event detection, identifying key trigger words and predicting event types, where data-hungry models suffer from the trigger bias that comes from ED datasets. [Auto-Debias: Debiasing Masked Language Models with Automated Biased Prompts](https://aclanthology.org/2022.acl-long.72.pdf) is the ACL 2022 long paper that does the same for masked language models.

Vision-language models inherit biases from uncurated datasets scraped from the internet, and those biases can be amplified and propagated to zero-shot classifiers and text-to-image generative models. [Debiasing Vision-Language Models via Biased Prompts](https://arxiv.org/abs/2302.00070) proposes a general approach to that problem, and [A Prompt Array Keeps the Bias Away: Debiasing Vision-Language Models with Adversarial Learning](https://arxiv.org/pdf/2203.11933.pdf) brings adversarial learning to it. The same idea reaches 3D in [Debiasing Scores and Prompts of 2D Diffusion for Robust Text-to-3D Generation](https://arxiv.org/pdf/2303.15413.pdf), listed on arXiv as view-consistent text-to-3D generation.

Before debiasing, the bias has to be measured. (good) [Understanding Stereotypes in Language Models: Towards Robust Measurement and Zero-Shot Debiasing](https://arxiv.org/pdf/2212.10678.pdf) does both; its arXiv entry now reads "Causally Testing Gender Bias in LLMs: A Case Study on Occupational Bias". Finally, the model can be asked to find its own bias: [Self-Diagnosis and Self-Debiasing: A Proposal for Reducing Corpus-Based Bias in NLP](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00434/108865).

## Hallucinations

A debiased answer can still be false, which is the last prompt-level failure on this page. (very good) [understanding LLM hallucinations](https://www.rungalileo.io/blog/deep-dive-into-llm-hallucinations-across-generative-tasks?utm_medium=email&_hsmi=304176203&utm_content=303486713&utm_source=hs_email) describes two types of LLM hallucinations and how they appear across different Natural Language Generation (NLG) tasks.
