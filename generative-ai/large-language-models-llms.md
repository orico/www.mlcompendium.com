# Large Language Models (LLMs)

This page collects papers, models, tools, methods, and use cases for large language models.

## Papers

This section lists early LLM papers on generative pre-training and few-shot learning.

1. [Language understanding by generative pre-training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) - Alec et al. openAI

   <figure><img src="../.gitbook/assets/image (8).png" alt=""><figcaption><p>Language understanding by generative pre-training</p></figcaption></figure>

2. [LLM are few shot learners](https://proceedings.neurips.cc/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf) - scaling LLMs with data is enough to make them few shot.

## Models

This section lists open and commercial LLM releases.

1. Databricks dolly
   1. [Version 1.0](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm)
   2. [Version 2.0](https://www.databricks.com/blog/2023/03/24/hello-dolly-democratizing-magic-chatgpt-open-models.html), [Huggingface](https://huggingface.co/databricks/dolly-v2-12b)
2. [LLaMA](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/)
3. [Bard](https://bard.google.com/)
4. [StabilityLM](https://github.com/Stability-AI/StableLM)
   1. Vicuna
   2. LLaMA

## Instructor

This section covers instruction following and the Instructor embedding model.

1. [Training language models to follow instructions with human feedback](https://arxiv.org/pdf/2203.02155) (using RLHF)
2. [Instructor model](https://instructor-embedding.github.io/)

   > We introduce Instructor👨‍🏫, an instruction-finetuned text embedding model that can generate text embeddings tailored to any task (e.g., classification, retrieval, clustering, text evaluation, etc.) and domains (e.g., science, finance, etc.) by simply providing the task instruction, without any finetuning. Instructor achieves sota on 70 diverse embedding tasks!

   <figure><img src="../.gitbook/assets/image (51).png" alt=""><figcaption><p>Instructor model</p></figcaption></figure>

## Datasets

This section points at a Dolly training dataset.

1. [Databricks' 15K QA for Dolly 2.0](https://github.com/databrickslabs/dolly/tree/master/data)

## Tools

This section lists libraries and UIs for building with LLMs.

1. [Scikit-LLM](https://github.com/iryna-kondr/scikit-llm)
2. [LangChain](https://python.langchain.com/en/latest/index.html)
   1. [An amazing tutorial](http://web.archive.org/web/20260506162130/https://www.python-engineer.com/posts/langchain-crash-course/) in [Youtube](https://www.youtube.com/watch?v=LbT1yp6quS8) by Patrick Loeber about
      * LLMs
        * Prompt Templates
        * Chains
        * Agents and Tools
        * Memory
        * Document Loaders
        * Indexes
   2. [Langchain in 13 minutes](https://www.youtube.com/watch?v=aywZrzNaKjs)
3. [ReAct & LangChain](https://tsmatz.wordpress.com/2023/03/07/react-with-openai-gpt-and-langchain/)
4. [LangFlow](https://github.com/logspace-ai/langflow), [Medium](https://medium.com/logspace/language-models-on-steroids-441cfcc66b24), [HuggingFace](https://medium.com/logspace/language-models-on-steroids-441cfcc66b24) - is a UI for LangChain, designed with react-flow to provide an effortless way to experiment and prototype flows.
5. [PandasAI](https://github.com/gventuri/pandas-ai) - PandasAI, asking data Qs using LLMs on Panda's DFs with two code lines. 𝚙𝚊𝚗𝚍𝚊𝚜_𝚊𝚒 = 𝙿𝚊𝚗𝚍𝚊𝚜𝙰𝙸(𝚕𝚕𝚖) & 𝚙𝚊𝚗𝚍𝚊𝚜_𝚊𝚒.𝚛𝚞𝚗(𝚍𝚏, 𝚙𝚛𝚘𝚖𝚙𝚝='𝚆𝚑𝚒𝚌𝚑 𝚊𝚛𝚎 𝚝𝚑𝚎 𝟻 𝚑𝚊𝚙𝚙𝚒𝚎𝚜𝚝 𝚌𝚘𝚞𝚗𝚝𝚛𝚒𝚎𝚜?')
6. [LLaMa Index](https://github.com/jerryjliu/llama_index) - LlamaIndex (GPT Index) is a project that provides a central interface to connect your LLM's with external data.
7. [LLM-foundry](https://github.com/mosaicml/llm-foundry) - LLM training code for Databricks foundation models using MoasicML
8. [Awesome ChatGPT - Curated list of awesome tools, demos, docs for ChatGPT and GPT-3](https://github.com/humanloop/awesome-chatgpt)
9. [GPT4 All Privacy-oriented software for chatting with large language models that run on your own computer.](https://github.com/nomic-ai/gpt4all)
10. [MinGPT](https://github.com/karpathy/minGPT) - A minimal PyTorch re-implementation of the OpenAI GPT (Generative Pretrained Transformer) training
11. [NanoGPT](https://github.com/karpathy/nanoGPT) - The simplest, fastest repository for training/finetuning medium-sized GPTs.
12. [Open-sourced codes for MiniGPT-4 and MiniGPT-v2](https://github.com/Vision-CAIR/MiniGPT-4) ([https://minigpt-4.github.io](https://minigpt-4.github.io), [https://minigpt-v2.github.io/](https://minigpt-v2.github.io/))

## Articles

This section collects explainers on prompting, production LLMs, decoding, and related topics.

1. [GPT4 can improve itself](https://www.youtube.com/watch?v=5SgJKZLBrmg)
2. [Lil Weng - Prompt Engineering](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/)
3. [Chip Huyen - Building LLM applicaations for production](https://huyenchip.com/2023/04/11/llm-engineering.html)
4. [How to generate text using different decoding methods for language generation with transformers](https://huggingface.co/blog/how-to-generate)
5. (great) [a gentle intro to LLMs and Langchain](https://medium.com/data-science/a-gentle-intro-to-chaining-llms-agents-and-utils-via-langchain-16cd385fca81)
6. [LLMs can explain NN of other LLMs](https://openai.com/research/language-models-can-explain-neurons-in-language-models) by OpenAI
7. [Fine tuning LLMs](https://medium.com/@miloszivic99/finetuning-large-language-models-customize-llama-3-8b-for-your-needs-bfe0f43cd239)
8. [Model size vs Computer overhead](https://www.harmdevries.com/post/model-size-vs-compute-overhead/) - The trade-off between model size and compute overhead and reveal there is significant room to reduce the compute-optimal model size with minimal compute overhead.

## Guardrails

This section points at tools and write-ups for LLM guardrails.

1. [GuardrailsAI](https://hub.guardrailsai.com/)

   <figure><img src="../.gitbook/assets/image (50).png" alt=""><figcaption><p>GuardrailsAI</p></figcaption></figure>

2. [Safeguarding LLMs with Guardrails](https://medium.com/data-science/safeguarding-llms-with-guardrails-4f5d9f57cff2)

   <figure><img src="../.gitbook/assets/image (47).png" alt=""><figcaption><p>Safeguarding LLMs with Guardrails</p></figcaption></figure>

3. [Databricks GR](https://www.databricks.com/blog/implementing-llm-guardrails-safe-and-responsible-generative-ai-deployment-databricks) - Implementing LLM Guardrails for Safe and Responsible Generative AI Deployment on Databricks

## Best Practices

This section points at Databricks notes on LLM and RAG evaluation.

1. [Best Practices for LLM Evaluation of RAG Applications](https://www.databricks.com/blog/LLM-auto-eval-best-practices-RAG)
2. [Announcing MLflow 2.8 LLM-as-a-judge metrics and Best Practices for LLM Evaluation of RAG Applications, Part 2](https://www.databricks.com/blog/announcing-mlflow-28-llm-judge-metrics-and-best-practices-llm-evaluation-rag-applications-part)

   <figure><img src="../.gitbook/assets/image (48).png" alt=""><figcaption><p>Announcing MLflow 2.8 LLM-as-a-judge metrics and Best Practices for LLM Evaluation of RAG Applications, Part 2</p></figcaption></figure>

## Reinforcement Learning for LLM

This section collects notes on RLHF.

1. [RLHF: Reinforcement Learning from Human Feedback](https://huyenchip.com/2023/05/02/rlhf.html) by Chip Huyen
2. [Yoav on RL](https://gist.github.com/yoavg/6bff0fecd65950898eba1bb321cfbd81)
3. [John Schulman](https://www.youtube.com/watch?v=hhiLw5Q_UFg) - Reinforcement Learning from Human Feedback: Progress and Challenges

## Metrics

This section explains ROUGE for summarization evaluation.

1. [Understanding ROUGE](https://dataman-ai.medium.com/understand-rouge-9ade61b0e0bc) - a family of metrics that evaluate the performance of a LLM in text summarization, i.e., ROUGE-1, ROUGE-2, ROUGE-L, for unigrams, bi grams, LCS, respectively.

## Use Cases

This section points at an external-memory ChatGPT setup.

1. [Enhancing ChatGPT With Infinite External Memory Using Vector Database and ChatGPT Retrieval Plugin](https://betterprogramming.pub/enhancing-chatgpt-with-infinite-external-memory-using-vector-database-and-chatgpt-retrieval-plugin-b6f4ea16ab8)

## Deprecated links

{% hint style="warning" %}
These links and images no longer work. The original wording is kept here. A same-resource copy, when one was checked, is used above.
{% endhint %}

- An amazing tutorial in Youtube by Patrick Loeber. This address no longer opens: https://www.python-engineer.com/posts/langchain-crash-course/
- (great) a gentle intro to LLMs and Langchain. This address no longer opens: https://towardsdatascience.com/a-gentle-intro-to-chaining-llms-agents-and-utils-via-langchain-16cd385fca81
- Safeguarding LLMs with Guardrails. This address no longer opens: https://towardsdatascience.com/safeguarding-llms-with-guardrails-4f5d9f57cff2
