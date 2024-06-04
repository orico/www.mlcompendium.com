# Large Language Models (LLMs)

## Articles

1. [GPT4 can improve itself](https://www.youtube.com/watch?v=5SgJKZLBrmg)
2. [Lil Weng - Prompt Engineering](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/)
3. [Chip Huyen - Building LLM applicaations for production](https://huyenchip.com/2023/04/11/llm-engineering.html)
4. [How to generate text using different decoding methods for language generation with transformers](https://huggingface.co/blog/how-to-generate)
5. (great) [a gentle intro to LLMs and Langchain](https://towardsdatascience.com/a-gentle-intro-to-chaining-llms-agents-and-utils-via-langchain-16cd385fca81)
6. [LLMs can explain NN of other LLMs](https://openai.com/research/language-models-can-explain-neurons-in-language-models) by OpenAI
7. [Fine tuning LLMs](https://medium.com/@miloszivic99/finetuning-large-language-models-customize-llama-3-8b-for-your-needs-bfe0f43cd239)

## Papers

1.  [Language understanding by generative pre-training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language\_understanding\_paper.pdf) - Alec et al. openAI\


    <figure><img src="../.gitbook/assets/image (8).png" alt=""><figcaption></figcaption></figure>
2. [LLM are few shot learners](https://proceedings.neurips.cc/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf) - scaling LLMs with data is enough to make them few shot.

## Models

1. Databricks dolly
   1. [Version 1.0](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm)
   2. [Version 2.0](https://www.databricks.com/blog/2023/03/24/hello-dolly-democratizing-magic-chatgpt-open-models.html), [Huggingface](https://huggingface.co/databricks/dolly-v2-12b)
2. [LLaMA](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/)
3. [Bard](https://bard.google.com/)
4. [StabilityLM](https://github.com/Stability-AI/StableLM)
   1. Vicuna
   2. LLaMA

## Instructor

1.  [Instructor model](https://instructor-embedding.github.io/) - "We introduce Instructor👨‍🏫, an instruction-finetuned text embedding model that can generate text embeddings tailored to any task (e.g., classification, retrieval, clustering, text evaluation, etc.) and domains (e.g., science, finance, etc.) by simply providing the task instruction, without any finetuning. Instructor achieves sota on 70 diverse embedding tasks!"

    <figure><img src="../.gitbook/assets/image (51).png" alt=""><figcaption></figcaption></figure>

## Datasets

1. [Databricks' 15K QA for Dolly 2.0](https://github.com/databrickslabs/dolly/tree/master/data)

## Tools

1. [Scikit-LLM](https://github.com/iryna-kondr/scikit-llm)
2. [LangChain](https://python.langchain.com/en/latest/index.html) - [An amazing tutorial](https://www.python-engineer.com/posts/langchain-crash-course/) in [Youtube](https://www.youtube.com/watch?v=LbT1yp6quS8) by Patrick Loeber about
   * LLMs
   * Prompt Templates
   * Chains
   * Agents and Tools
   * Memory
   * Document Loaders
   * Indexes
3. [ReAct & LangChain](https://tsmatz.wordpress.com/2023/03/07/react-with-openai-gpt-and-langchain/)
4. [LangFlow](https://github.com/logspace-ai/langflow), [Medium](https://medium.com/logspace/language-models-on-steroids-441cfcc66b24), [HuggingFace](https://medium.com/logspace/language-models-on-steroids-441cfcc66b24) - is a UI for LangChain, designed with react-flow to provide an effortless way to experiment and prototype flows.
5. [PandasAI](https://github.com/gventuri/pandas-ai) - PandasAI, asking data Qs using LLMs on Panda's DFs with two code lines. 𝚙𝚊𝚗𝚍𝚊𝚜\_𝚊𝚒 = 𝙿𝚊𝚗𝚍𝚊𝚜𝙰𝙸(𝚕𝚕𝚖) & 𝚙𝚊𝚗𝚍𝚊𝚜\_𝚊𝚒.𝚛𝚞𝚗(𝚍𝚏, 𝚙𝚛𝚘𝚖𝚙𝚝='𝚆𝚑𝚒𝚌𝚑 𝚊𝚛𝚎 𝚝𝚑𝚎 𝟻 𝚑𝚊𝚙𝚙𝚒𝚎𝚜𝚝 𝚌𝚘𝚞𝚗𝚝𝚛𝚒𝚎𝚜?')
6. [LLaMa Index](https://github.com/jerryjliu/llama\_index) - LlamaIndex (GPT Index) is a project that provides a central interface to connect your LLM's with external data.

## Guardrails

1.  [GuardrailsAI](https://hub.guardrailsai.com/)\


    <figure><img src="../.gitbook/assets/image (50).png" alt=""><figcaption></figcaption></figure>
2.  [Safeguarding LLMs with Guardrails](https://towardsdatascience.com/safeguarding-llms-with-guardrails-4f5d9f57cff2)\


    <figure><img src="../.gitbook/assets/image (47).png" alt=""><figcaption></figcaption></figure>
3. [Databricks GR](https://www.databricks.com/blog/implementing-llm-guardrails-safe-and-responsible-generative-ai-deployment-databricks) - Implementing LLM Guardrails for Safe and Responsible Generative AI Deployment on Databricks

## Best Practices

1. [Best Practices for LLM Evaluation of RAG Applications](https://www.databricks.com/blog/LLM-auto-eval-best-practices-RAG)
2.  [Announcing MLflow 2.8 LLM-as-a-judge metrics and Best Practices for LLM Evaluation of RAG Applications, Part 2\
    ](https://www.databricks.com/blog/announcing-mlflow-28-llm-judge-metrics-and-best-practices-llm-evaluation-rag-applications-part)

    <figure><img src="../.gitbook/assets/image (48).png" alt=""><figcaption></figcaption></figure>

## Metrics

1. [Understanding ROUGE](https://dataman-ai.medium.com/understand-rouge-9ade61b0e0bc) - a family of metrics that evaluate the performance of a LLM in text summarization, i.e., ROUGE-1, ROUGE-2, ROUGE-L, for unigrams, bi grams, LCS, respectively.

## Use Cases

1. [Enhancing ChatGPT With Infinite External Memory Using Vector Database and ChatGPT Retrieval Plugin](https://betterprogramming.pub/enhancing-chatgpt-with-infinite-external-memory-using-vector-database-and-chatgpt-retrieval-plugin-b6f4ea16ab8)

