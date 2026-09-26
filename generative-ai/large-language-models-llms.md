# Large Language Models (LLMs)

A large language model is a generator you can talk to, and building with one means knowing where it came from, which ones you can run, and how to check and constrain what it says. The page goes in that order: the founding papers, the models and the instruction step, a dataset, the tools and articles for building, then evaluation practices, guardrails, reinforcement learning, metrics, and use cases. A second part on GPT follows the same path for that one family.

The same notes are in [GPT](large-language-models-llms.md), [GPT3 is ZERO, ONE, FEW](../problem-framing/n-shot-learning.md#gpt3-is-zero-one-few), [NEURAL LANGUAGE GENERATION](../language-ai/language-detection-identification-generation-nld-nli-nlg.md#neural-language-generation), [Prompt](prompt.md), and [Unlearning](../responsible-ai/unlearning.md).

## Papers

The story starts with two papers: one showed that generative pre-training teaches language understanding, and the next showed that scaling that recipe makes a model few-shot. [Language understanding by generative pre-training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) - Alec et al. openAI, is the first.

 <figure><img src="../.gitbook/assets/image (8).png" alt=""><figcaption><p>Language understanding by generative pre-training</p></figcaption></figure>

The figure above is from that paper. The second step is [LLM are few shot learners](https://proceedings.neurips.cc/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf) - scaling LLMs with data is enough to make them few shot.

## Models

Once scale was shown to work, open and commercial models followed, and the practical question became which one you can use. Databricks Dolly is the open, commercially usable line: [Version 1.0](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm) introduces Dolly as the first open-source, commercially viable instruction-tuned LLM, [Version 2.0](https://www.databricks.com/blog/2023/03/24/hello-dolly-democratizing-magic-chatgpt-open-models.html) is the "Hello Dolly" post on democratizing the magic of ChatGPT with open models, and the weights are on [Huggingface](https://huggingface.co/databricks/dolly-v2-12b). Meta's [LLaMA](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/) is a foundational 65-billion-parameter model released under a gated release, described as more efficient than, and competitive with, published models of similar size. Google's chat model was [Bard](https://bard.google.com/), which now opens as Google Gemini. [StabilityLM](https://github.com/Stability-AI/StableLM) is the repository of Stability AI's StableLM language models. Vicuna and LLaMA are also named on this list.

## Instructor

A raw model predicts text; following an instruction is a separate training step. [Training language models to follow instructions with human feedback](https://arxiv.org/pdf/2203.02155) (using RLHF) is the paper for that step. The same idea applied to embeddings is the [Instructor model](https://instructor-embedding.github.io/), billed as one embedder for all tasks:

 > We introduce Instructor👨‍🏫, an instruction-finetuned text embedding model that can generate text embeddings tailored to any task (e.g., classification, retrieval, clustering, text evaluation, etc.) and domains (e.g., science, finance, etc.) by simply providing the task instruction, without any finetuning. Instructor achieves sota on 70 diverse embedding tasks!

 <figure><img src="../.gitbook/assets/image (51).png" alt=""><figcaption><p>Instructor model</p></figcaption></figure>

## Datasets

Instruction tuning needs instruction data, and the Dolly models above came with theirs. [Databricks' 15K QA for Dolly 2.0](https://github.com/databrickslabs/dolly/tree/master/data) is the data folder of the Dolly repository, the set used for the model trained on the Databricks Machine Learning Platform.

## Tools

With a model and data in hand, the next layer is the libraries that turn a model into an application.

The same notes are in [Agents](agents.md) and [RAG](rag.md).

[Scikit-LLM](https://github.com/iryna-kondr/scikit-llm) integrates LLMs into scikit-learn. [LangChain](https://python.langchain.com/en/latest/index.html) is the framework most of the other tools here orbit; its documentation now leads with create_agent, a minimal, configurable agent harness composed from a model, tools, a prompt, and middleware. To learn it, [An amazing tutorial](http://web.archive.org/web/20260506162130/https://www.python-engineer.com/posts/langchain-crash-course/) is the archived crash course on building applications powered by large language models, and Patrick Loeber's matching video, "LangChain Crash Course - Build apps with language models", is in [Youtube](https://www.youtube.com/watch?v=LbT1yp6quS8). The course walks through LangChain's parts in this order:

 - LLMs
 - Prompt Templates
 - Chains
 - Agents and Tools
 - Memory
 - Document Loaders
 - Indexes

For a shorter start, [Langchain in 13 minutes](https://www.youtube.com/watch?v=aywZrzNaKjs) is Rabbitmetrics' quickstart for beginners. [ReAct & LangChain](https://tsmatz.wordpress.com/2023/03/07/react-with-openai-gpt-and-langchain/) explains ReAct (Reasoning + Acting), the LLM chain framework behind much advanced reasoning, and how LangChain composes it.

LangChain also grew a visual layer. [LangFlow](https://github.com/logspace-ai/langflow), [Medium](https://medium.com/logspace/language-models-on-steroids-441cfcc66b24), [HuggingFace](https://medium.com/logspace/language-models-on-steroids-441cfcc66b24) - is a UI for LangChain, designed with react-flow to provide an effortless way to experiment and prototype flows. The Medium post, Rodrigo Nader's "Language Models On Steroids", frames it by the success of ChatGPT, with 100 million users in its first two months.

Other tools connect a model to your data. [PandasAI](https://github.com/gventuri/pandas-ai) - PandasAI, asking data Qs using LLMs on Panda's DFs with two code lines. 𝚙𝚊𝚗𝚍𝚊𝚜_𝚊𝚒 = 𝙿𝚊𝚗𝚍𝚊𝚜𝙰𝙸(𝚕𝚕𝚖) & 𝚙𝚊𝚗𝚍𝚊𝚜_𝚊𝚒.𝚛𝚞𝚗(𝚍𝚏, 𝚙𝚛𝚘𝚖𝚙𝚝='𝚆𝚑𝚒𝚌𝚑 𝚊𝚛𝚎 𝚝𝚑𝚎 𝟻 𝚑𝚊𝚙𝚙𝚒𝚎𝚜𝚝 𝚌𝚘𝚞𝚗𝚝𝚛𝚒𝚎𝚜?') For data outside a data frame, [LLaMa Index](https://github.com/jerryjliu/llama_index) - LlamaIndex (GPT Index) is a project that provides a central interface to connect your LLM's with external data. [LLM-foundry](https://github.com/mosaicml/llm-foundry) - LLM training code for Databricks foundation models using MoasicML.

For finding more, [Awesome ChatGPT - Curated list of awesome tools, demos, docs for ChatGPT and GPT-3](https://github.com/humanloop/awesome-chatgpt) is humanloop's curated list. To run a model on your own machine, [GPT4 All Privacy-oriented software for chatting with large language models that run on your own computer.](https://github.com/nomic-ai/gpt4all) is nomic-ai's GPT4All, open source and available for commercial use.

To understand the model by building one, [MinGPT](https://github.com/karpathy/minGPT) - A minimal PyTorch re-implementation of the OpenAI GPT (Generative Pretrained Transformer) training. [NanoGPT](https://github.com/karpathy/nanoGPT) - The simplest, fastest repository for training/finetuning medium-sized GPTs.

The same kind of model can also see. [Open-sourced codes for MiniGPT-4 and MiniGPT-v2](https://github.com/Vision-CAIR/MiniGPT-4) is the Vision-CAIR repository with the code for both, and each has a project page (https://minigpt-4.github.io, https://minigpt-v2.github.io/): MiniGPT-4 at ( [https://minigpt-4.github.io](https://minigpt-4.github.io) and MiniGPT-v2 at [https://minigpt-v2.github.io/](https://minigpt-v2.github.io/).

## Articles

Tools answer how; the articles explain why the models behave the way they do and how to steer them.

The same notes are in [Prompt](prompt.md).

[GPT4 can improve itself](https://www.youtube.com/watch?v=5SgJKZLBrmg) is the video on that claim. [Lil Weng - Prompt Engineering](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/) is Lilian Weng's post on prompt engineering, also called in-context prompting: methods for steering an LLM toward a desired outcome without updating its weights, an empirical science whose effects vary a lot between models and need heavy experimentation. [Chip Huyen - Building LLM applicaations for production](https://huyenchip.com/2023/04/11/llm-engineering.html) is her post on taking those applications to production. [How to generate text using different decoding methods for language generation with transformers](https://huggingface.co/blog/how-to-generate) is the Hugging Face post on decoding methods, the step that turns probabilities into text. (great) [a gentle intro to LLMs and Langchain](https://medium.com/data-science/a-gentle-intro-to-chaining-llms-agents-and-utils-via-langchain-16cd385fca81) is the introduction to chaining LLMs, agents, and utilities with LangChain. [LLMs can explain NN of other LLMs](https://openai.com/research/language-models-can-explain-neurons-in-language-models) is OpenAI's work on language models explaining the neurons of language models. [Fine tuning LLMs](https://medium.com/@miloszivic99/finetuning-large-language-models-customize-llama-3-8b-for-your-needs-bfe0f43cd239) is a walkthrough of customizing Llama 3 8B for your needs. [Model size vs Computer overhead](https://www.harmdevries.com/post/model-size-vs-compute-overhead/) - The trade-off between model size and compute overhead and reveal there is significant room to reduce the compute-optimal model size with minimal compute overhead.

## Best Practices

A built application still has to be measured, and for RAG applications the judge is often another LLM. [Best Practices for LLM Evaluation of RAG Applications](https://www.databricks.com/blog/LLM-auto-eval-best-practices-RAG) is the Databricks post on evaluating retrieval-augmented generation with LLMs. [Announcing MLflow 2.8 LLM-as-a-judge metrics and Best Practices for LLM Evaluation of RAG Applications, Part 2](https://www.databricks.com/blog/announcing-mlflow-28-llm-judge-metrics-and-best-practices-llm-evaluation-rag-applications-part) is the follow-up: MLflow 2.8 automates evaluation with LLM judges to save time and cost, and it shows that data cleaning improves RAG performance.

 <figure><img src="../.gitbook/assets/image (48).png" alt=""><figcaption><p>Announcing MLflow 2.8 LLM-as-a-judge metrics and Best Practices for LLM Evaluation of RAG Applications, Part 2</p></figcaption></figure>

## Guardrails

Evaluation tells you what went wrong afterward; guardrails stop it at run time. [GuardrailsAI](https://hub.guardrailsai.com/) is the Guardrails Hub from Guardrails AI.

 <figure><img src="../.gitbook/assets/image (50).png" alt=""><figcaption><p>GuardrailsAI</p></figcaption></figure>

[Safeguarding LLMs with Guardrails](https://medium.com/data-science/safeguarding-llms-with-guardrails-4f5d9f57cff2) is Aparna Dhinakaran's article, co-authored with Hakan Tekgul.

 <figure><img src="../.gitbook/assets/image (47).png" alt=""><figcaption><p>Safeguarding LLMs with Guardrails</p></figcaption></figure>

On a managed platform, [Databricks GR](https://www.databricks.com/blog/implementing-llm-guardrails-safe-and-responsible-generative-ai-deployment-databricks) - Implementing LLM Guardrails for Safe and Responsible Generative AI Deployment on Databricks.

## Reinforcement Learning for LLM

Guardrails filter outputs; reinforcement learning from human feedback changes the model so it produces better ones in the first place.

The same notes are in [RLHF](../decision-intelligence/reinforcement-learning.md#rlhf).

[RLHF: Reinforcement Learning from Human Feedback](https://huyenchip.com/2023/05/02/rlhf.html) is Chip Huyen's explanation. [Yoav on RL](https://gist.github.com/yoavg/6bff0fecd65950898eba1bb321cfbd81) is the rl-for-llms gist. [John Schulman](https://www.youtube.com/watch?v=hhiLw5Q_UFg) - Reinforcement Learning from Human Feedback: Progress and Challenges.

## Metrics

Whatever trained the model, its text output still needs a score.

The same notes are in [LANGUAGE TRANSLATION](../language-ai/language-detection-identification-generation-nld-nli-nlg.md#language-translation), [Perplexity](../evals/evaluation-metrics.md#perplexity), and [Summarization](../language-ai/summarization.md).

For summaries, [Understanding ROUGE](https://dataman-ai.medium.com/understand-rouge-9ade61b0e0bc) - a family of metrics that evaluate the performance of a LLM in text summarization, i.e., ROUGE-1, ROUGE-2, ROUGE-L, for unigrams, bi grams, LCS, respectively.

## Use Cases

The last step is putting a model to work beyond its context window. [Enhancing ChatGPT With Infinite External Memory Using Vector Database and ChatGPT Retrieval Plugin](https://betterprogramming.pub/enhancing-chatgpt-with-infinite-external-memory-using-vector-database-and-chatgpt-retrieval-plugin-b6f4ea16ab8) gives ChatGPT external memory through a vector database and the retrieval plugin.

Guardrails come back as a use case too: Wenqi Glantz's "NeMo Guardrails, the Ultimate Open-Source LLM Security Toolkit", filed here as Safeguarding LLMs with Guardrails, explores the practical use cases of NeMo Guardrails, at [https://towardsdatascience.com/safeguarding-llms-with-guardrails-4f5d9f57cff2](https://towardsdatascience.com/safeguarding-llms-with-guardrails-4f5d9f57cff2)
# GPT

GPT is the family most of the above grew from, so it gets its own path: the methods that led to it, the embedding tools built on it, the articles that explain and compare its alignment, the hackathons, and a catalog of assistants.

The same notes are in [Chat Bots](chat-bots.md), [Decoding Algorithms For NLP](../language-ai/decoding-algorithms-for-nlp.md), [GPT2](../language-ai/pretrained-language-models.md#gpt2), [GPT3](../language-ai/pretrained-language-models.md#gpt3), [Large Language Models (LLMs)](large-language-models-llms.md), [Prompt](prompt.md), and [Tokenization](../language-ai/tokenization.md).

## Precursor

Instruction-following GPT came from three steps, each building on the last:

1. [Proximal Policy Optimization](https://openai.com/research/openai-baselines-ppo) (PPO) - an RL algorithm, PPO is better than state-of-the-art approaches while being much simpler to implement and tune and is the default reinforcement learning algorithm at OpenAI.
2. [Learning from human preference](https://openai.com/research/learning-from-human-preferences) (human in the loop) - a method used to infer what humans want by being told which of two proposed behaviors is better.
3. [instructGPT](https://openai.com/research/instruction-following) - arguably better at following user intentions than GPT-3 while also making them more truthful and less toxic, using human in the loop.

## Tools

Beyond chat, GPT models also produce sentence embeddings for search. [sentence embedding for semantic search](https://github.com/Muennighoff/sgpt) is SGPT, GPT sentence embeddings for semantic search. [GPT 3 Dense sentence embeddings](https://medium.com/@nils_reimers/openai-gpt-3-text-embeddings-really-a-new-state-of-the-art-in-dense-text-embeddings-6571fe3ec9d9) is Nils Reimers asking whether OpenAI's GPT-3 text embeddings are really a new state of the art in dense text embeddings.

## Articles

With the tools named, the articles explain how the model works and how its alignment compares. [what is chatGPT doing and why does it work?](https://writings.stephenwolfram.com/2023/02/what-is-chatgpt-doing-and-why-does-it-work/) is Stephen Wolfram's broader picture. [Karpathy on building GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=191s) is Andrej Karpathy's "Let's build GPT: from scratch, in code, spelled out." On alignment, [Is DPO Superior to PPO for LLM Alignment](https://arxiv.org/pdf/2404.10719)? A Comprehensive Study - answers with the PPO precursor above:

 > PPO is able to surpass other alignment methods in all cases and achieve state-of-the-art results in challenging code competitions.

## Competitions

Hackathons show what people actually build with it. The GPT 4 [Hackathon code results](https://docs.google.com/spreadsheets/d/1tmfn8jKb7T1x7PpyO7rD023tH2zc_WDg_OHh0aVXIrw/edit#gid=174517450) and the [LangChain Gen Hackathon](https://docs.google.com/spreadsheets/d/1GqwPo1FpAbe_awmNZW5ZMH69yc5QtEr7ZYw-ckaz_mQ/edit#gid=795016726) sheets list the results.

## Virtual assistants

The same builds end up as assistants people share. [flowGPT](https://flowgpt.com/) is an open platform to discover prompts and AI apps built by a global community, or to publish your own.

## Deprecated links

{% hint style="warning" %}
These links and images no longer work. The original wording is kept here. A same-resource copy, when one was checked, is used above.
{% endhint %}

- An amazing tutorial in Youtube by Patrick Loeber. This address no longer opens: https://www.python-engineer.com/posts/langchain-crash-course/
- (great) a gentle intro to LLMs and Langchain. This address no longer opens: https://towardsdatascience.com/a-gentle-intro-to-chaining-llms-agents-and-utils-via-langchain-16cd385fca81
