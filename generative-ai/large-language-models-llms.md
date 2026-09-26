# Large Language Models (LLMs)

This page collects papers, models, tools, methods, and use cases for large language models.
The sections below group papers, models, datasets, tools, practices, guardrails, metrics, use cases, and GPT notes.

The same notes are in [GPT](large-language-models-llms.md), [GPT3 is ZERO, ONE, FEW](../problem-framing/n-shot-learning.md#gpt3-is-zero-one-few), [NEURAL LANGUAGE GENERATION](../language-ai/language-detection-identification-generation-nld-nli-nlg.md#neural-language-generation), [Prompt](prompt.md), and [Unlearning](../responsible-ai/unlearning.md).

## Papers

This section lists early LLM papers on generative pre-training and few-shot learning.

1. [Language understanding by generative pre-training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) - Alec et al. openAI

 <figure><img src="../.gitbook/assets/image (8).png" alt=""><figcaption><p>Language understanding by generative pre-training</p></figcaption></figure>

2. [LLM are few shot learners](https://proceedings.neurips.cc/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf) - scaling LLMs with data is enough to make them few shot.

## Models

This section lists open and commercial LLM releases.

1. Databricks dolly
 - Introducing Dolly, the first open-source, commercially viable instruction-tuned LLM, enabling accessible and cost-effective AI solutions. [Version 1.0](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm)
 - Introducing 'Hello Dolly,' a project to democratize AI by integrating ChatGPT and open models, making advanced AI accessible to everyone. [Version 2.0](https://www.databricks.com/blog/2023/03/24/hello-dolly-democratizing-magic-chatgpt-open-models.html)
 - We’re on a journey to advance and democratize artificial intelligence through open source and open science. [Huggingface](https://huggingface.co/databricks/dolly-v2-12b)
- Today, we’re releasing our LLaMA (Large Language Model Meta AI) foundational model with a gated release. [LLaMA](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/)
- ‎Google Gemini. [Bard](https://bard.google.com/)
- GitHub - Stability-AI/StableLM: StableLM: Stability AI Language Models. [StabilityLM](https://github.com/Stability-AI/StableLM)
 - Vicuna
 - LLaMA

## Instructor

This section covers instruction following and the Instructor embedding model.

1. [Training language models to follow instructions with human feedback](https://arxiv.org/pdf/2203.02155) (using RLHF)
- Instructor Text Embedding. Instructor Text Embedding. [Instructor model](https://instructor-embedding.github.io/)

 > We introduce Instructor👨‍🏫, an instruction-finetuned text embedding model that can generate text embeddings tailored to any task (e.g., classification, retrieval, clustering, text evaluation, etc.) and domains (e.g., science, finance, etc.) by simply providing the task instruction, without any finetuning. Instructor achieves sota on 70 diverse embedding tasks!

 <figure><img src="../.gitbook/assets/image (51).png" alt=""><figcaption><p>Instructor model</p></figcaption></figure>

## Datasets

This section points at a Dolly training dataset.

- Databricks’ Dolly, a large language model trained on the Databricks Machine Learning Platform - dolly/data at master · databrickslabs/dolly. [Databricks' 15K QA for Dolly 2.0](https://github.com/databrickslabs/dolly/tree/master/data)

## Tools

This section lists libraries and UIs for building with LLMs.

The same notes are in [Agents](agents.md) and [RAG](rag.md).

- Seamlessly integrate LLMs into scikit-learn. [Scikit-LLM](https://github.com/iryna-kondr/scikit-llm)
- LangChain provides create_agent: a minimal, highly configurable agent harness. [LangChain](https://python.langchain.com/en/latest/index.html)
 - In this LangChain Crash Course you will learn how to build applications powered by large language models. [An amazing tutorial](http://web.archive.org/web/20260506162130/https://www.python-engineer.com/posts/langchain-crash-course/)
 - LangChain Crash Course - Build apps with language models, by Patrick Loeber. in [Youtube](https://www.youtube.com/watch?v=LbT1yp6quS8)
 - LLMs
 - Prompt Templates
 - Chains
 - Agents and Tools
 - Memory
 - Document Loaders
 - Indexes
 - LangChain Explained in 13 Minutes | QuickStart Tutorial for Beginners, by Rabbitmetrics. [Langchain in 13 minutes](https://www.youtube.com/watch?v=aywZrzNaKjs)
- ReAct (Reason+Act) prompting in LLM – tsmatz. [ReAct & LangChain](https://tsmatz.wordpress.com/2023/03/07/react-with-openai-gpt-and-langchain/)
4. [LangFlow](https://github.com/logspace-ai/langflow), [Medium](https://medium.com/logspace/language-models-on-steroids-441cfcc66b24), [HuggingFace](https://medium.com/logspace/language-models-on-steroids-441cfcc66b24) - is a UI for LangChain, designed with react-flow to provide an effortless way to experiment and prototype flows.
5. [PandasAI](https://github.com/gventuri/pandas-ai) - PandasAI, asking data Qs using LLMs on Panda's DFs with two code lines. 𝚙𝚊𝚗𝚍𝚊𝚜_𝚊𝚒 = 𝙿𝚊𝚗𝚍𝚊𝚜𝙰𝙸(𝚕𝚕𝚖) & 𝚙𝚊𝚗𝚍𝚊𝚜_𝚊𝚒.𝚛𝚞𝚗(𝚍𝚏, 𝚙𝚛𝚘𝚖𝚙𝚝='𝚆𝚑𝚒𝚌𝚑 𝚊𝚛𝚎 𝚝𝚑𝚎 𝟻 𝚑𝚊𝚙𝚙𝚒𝚎𝚜𝚝 𝚌𝚘𝚞𝚗𝚝𝚛𝚒𝚎𝚜?')
6. [LLaMa Index](https://github.com/jerryjliu/llama_index) - LlamaIndex (GPT Index) is a project that provides a central interface to connect your LLM's with external data.
7. [LLM-foundry](https://github.com/mosaicml/llm-foundry) - LLM training code for Databricks foundation models using MoasicML
- Curated list of awesome tools, demos, docs for ChatGPT and GPT-3 - humanloop/awesome-chatgpt. [Awesome ChatGPT - Curated list of awesome tools, demos, docs for ChatGPT and GPT-3](https://github.com/humanloop/awesome-chatgpt)
- GitHub - nomic-ai/gpt4all: GPT4All: Run Local LLMs on Any Device. Open-source and available for commercial use. [GPT4 All Privacy-oriented software for chatting with large language models that run on your own computer.](https://github.com/nomic-ai/gpt4all)
10. [MinGPT](https://github.com/karpathy/minGPT) - A minimal PyTorch re-implementation of the OpenAI GPT (Generative Pretrained Transformer) training
11. [NanoGPT](https://github.com/karpathy/nanoGPT) - The simplest, fastest repository for training/finetuning medium-sized GPTs.
- Open-sourced codes for MiniGPT-4 and MiniGPT-v2 (https://minigpt-4.github.io, https://minigpt-v2.github.io/) - Vision-CAIR/MiniGPT-4. [Open-sourced codes for MiniGPT-4 and MiniGPT-v2](https://github.com/Vision-CAIR/MiniGPT-4)
- Minigpt-4. Minigpt-4. ( [https://minigpt-4.github.io](https://minigpt-4.github.io)
- MiniGPT-v2. MiniGPT-v2. [https://minigpt-v2.github.io/](https://minigpt-v2.github.io/)

## Articles

This section collects explainers on prompting, production LLMs, decoding, and related topics.

The same notes are in [Prompt](prompt.md).

1. [GPT4 can improve itself](https://www.youtube.com/watch?v=5SgJKZLBrmg)
- Prompt Engineering, by Lilian Weng. Prompt Engineering, by Lilian Weng. [Lil Weng - Prompt Engineering](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/)
- [Hacker News discussion, LinkedIn discussion, Twitter thread]. [Chip Huyen - Building LLM applicaations for production](https://huyenchip.com/2023/04/11/llm-engineering.html)
- We’re on a journey to advance and democratize artificial intelligence through open source and open science. [How to generate text using different decoding methods for language generation with transformers](https://huggingface.co/blog/how-to-generate)
- (great) [a gentle intro to LLMs and Langchain](https://medium.com/data-science/a-gentle-intro-to-chaining-llms-agents-and-utils-via-langchain-16cd385fca81)
6. [LLMs can explain NN of other LLMs](https://openai.com/research/language-models-can-explain-neurons-in-language-models)
- [Fine tuning LLMs](https://medium.com/@miloszivic99/finetuning-large-language-models-customize-llama-3-8b-for-your-needs-bfe0f43cd239)
8. [Model size vs Computer overhead](https://www.harmdevries.com/post/model-size-vs-compute-overhead/) - The trade-off between model size and compute overhead and reveal there is significant room to reduce the compute-optimal model size with minimal compute overhead.

## Best Practices

This section points at Databricks notes on LLM and RAG evaluation.

- This blog post discusses best practices for evaluating retrieval-augmented generation (RAG) applications using large language models (LLMs). [Best Practices for LLM Evaluation of RAG Applications](https://www.databricks.com/blog/LLM-auto-eval-best-practices-RAG)
- MLflow 2.8 introduces automated evaluation with LLM judges, saving time and costs. [Announcing MLflow 2.8 LLM-as-a-judge metrics and Best Practices for LLM Evaluation of RAG Applications, Part 2](https://www.databricks.com/blog/announcing-mlflow-28-llm-judge-metrics-and-best-practices-llm-evaluation-rag-applications-part)

 <figure><img src="../.gitbook/assets/image (48).png" alt=""><figcaption><p>Announcing MLflow 2.8 LLM-as-a-judge metrics and Best Practices for LLM Evaluation of RAG Applications, Part 2</p></figcaption></figure>

## Guardrails

This section points at tools and write-ups for LLM guardrails.

- Guardrails Hub | Guardrails AI. Guardrails Hub | Guardrails AI. [GuardrailsAI](https://hub.guardrailsai.com/)

 <figure><img src="../.gitbook/assets/image (50).png" alt=""><figcaption><p>GuardrailsAI</p></figcaption></figure>

- [Safeguarding LLMs with Guardrails](https://medium.com/data-science/safeguarding-llms-with-guardrails-4f5d9f57cff2)

 <figure><img src="../.gitbook/assets/image (47).png" alt=""><figcaption><p>Safeguarding LLMs with Guardrails</p></figcaption></figure>

3. [Databricks GR](https://www.databricks.com/blog/implementing-llm-guardrails-safe-and-responsible-generative-ai-deployment-databricks) - Implementing LLM Guardrails for Safe and Responsible Generative AI Deployment on Databricks

## Reinforcement Learning for LLM

This section collects notes on RLHF.

The same notes are in [RLHF](../decision-intelligence/reinforcement-learning.md#rlhf).

- RLHF: Reinforcement Learning from Human Feedback, by Chip Huyen. [RLHF: Reinforcement Learning from Human Feedback](https://huyenchip.com/2023/05/02/rlhf.html)
- GitHub Gist: instantly share code, notes, and snippets. [Yoav on RL](https://gist.github.com/yoavg/6bff0fecd65950898eba1bb321cfbd81)
3. [John Schulman](https://www.youtube.com/watch?v=hhiLw5Q_UFg) - Reinforcement Learning from Human Feedback: Progress and Challenges

## Metrics

This section explains ROUGE for summarization evaluation.

The same notes are in [LANGUAGE TRANSLATION](../language-ai/language-detection-identification-generation-nld-nli-nlg.md#language-translation), [Perplexity](../evals/evaluation-metrics.md#perplexity), and [Summarization](../language-ai/summarization.md).

1. [Understanding ROUGE](https://dataman-ai.medium.com/understand-rouge-9ade61b0e0bc) - a family of metrics that evaluate the performance of a LLM in text summarization, i.e., ROUGE-1, ROUGE-2, ROUGE-L, for unigrams, bi grams, LCS, respectively.

## Use Cases

This section points at an external-memory ChatGPT setup.

- [Enhancing ChatGPT With Infinite External Memory Using Vector Database and ChatGPT Retrieval Plugin](https://betterprogramming.pub/enhancing-chatgpt-with-infinite-external-memory-using-vector-database-and-chatgpt-retrieval-plugin-b6f4ea16ab8)

- Exploring NeMo Guardrails' practical use cases, by Wenqi Glantz. Safeguarding LLMs with Guardrails. [https://towardsdatascience.com/safeguarding-llms-with-guardrails-4f5d9f57cff2](https://towardsdatascience.com/safeguarding-llms-with-guardrails-4f5d9f57cff2)
# GPT

This page collects precursors, articles, tools, competitions, and assistants around GPT.
The sections below cover PPO and instructGPT precursors, embedding tools, alignment articles, hackathons, and virtual-assistant catalogs.

The same notes are in [Chat Bots](chat-bots.md), [Decoding Algorithms For NLP](../language-ai/decoding-algorithms-for-nlp.md), [GPT2](../language-ai/pretrained-language-models.md#gpt2), [GPT3](../language-ai/pretrained-language-models.md#gpt3), [Large Language Models (LLMs)](large-language-models-llms.md), [Prompt](prompt.md), and [Tokenization](../language-ai/tokenization.md).

## Precursor

This section lists methods that led into instruction-tuned GPT models.

1. [Proximal Policy Optimization](https://openai.com/research/openai-baselines-ppo) (PPO) - an RL algorithm, PPO is better than state-of-the-art approaches while being much simpler to implement and tune and is the default reinforcement learning algorithm at OpenAI.
2. [Learning from human preference](https://openai.com/research/learning-from-human-preferences) (human in the loop) - a method used to infer what humans want by being told which of two proposed behaviors is better.
3. [instructGPT](https://openai.com/research/instruction-following) - arguably better at following user intentions than GPT-3 while also making them more truthful and less toxic, using human in the loop.

## Tools

This section points at sentence-embedding tools related to GPT.

1. Sentence Embeddings
 - SGPT: GPT Sentence Embeddings for Semantic Search. [sentence embedding for semantic search](https://github.com/Muennighoff/sgpt)
 - [GPT 3 Dense sentence embeddings](https://medium.com/@nils_reimers/openai-gpt-3-text-embeddings-really-a-new-state-of-the-art-in-dense-text-embeddings-6571fe3ec9d9)

## Articles

This section collects explanations and studies of how GPT-style models work and align.

- Stephen Wolfram explores the broader picture of what. [what is chatGPT doing and why does it work?](https://writings.stephenwolfram.com/2023/02/what-is-chatgpt-doing-and-why-does-it-work/)
- Let's build GPT: from scratch, in code, spelled out, by Andrej Karpathy. [Karpathy on building GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=191s)
3. [Is DPO Superior to PPO for LLM Alignment](https://arxiv.org/pdf/2404.10719)? A Comprehensive Study -

 > PPO is able to surpass other alignment methods in all cases and achieve state-of-the-art results in challenging code competitions.

## Competitions

This section points at hackathon result sheets.

- GPT 4 [Hackathon code results](https://docs.google.com/spreadsheets/d/1tmfn8jKb7T1x7PpyO7rD023tH2zc_WDg_OHh0aVXIrw/edit#gid=174517450)
- Langchain Hackathon. Langchain Hackathon. [LangChain Gen Hackathon](https://docs.google.com/spreadsheets/d/1GqwPo1FpAbe_awmNZW5ZMH69yc5QtEr7ZYw-ckaz_mQ/edit#gid=795016726)

## Virtual assistants

This section points at a catalog of bots and prompts.

- Discover prompts and AI apps built by a global community, or publish your own. [flowGPT](https://flowgpt.com/)

## Deprecated links

{% hint style="warning" %}
These links and images no longer work. The original wording is kept here. A same-resource copy, when one was checked, is used above.
{% endhint %}

- An amazing tutorial in Youtube by Patrick Loeber. This address no longer opens: https://www.python-engineer.com/posts/langchain-crash-course/
- (great) a gentle intro to LLMs and Langchain. This address no longer opens: https://towardsdatascience.com/a-gentle-intro-to-chaining-llms-agents-and-utils-via-langchain-16cd385fca81
