# Prompt

This page collects prompt-engineering articles, papers, and techniques.
The sections below cover prompt engineering, step-back prompting, chain of thought, tuning, hacking examples, papers, and articles.

The same notes are in [Articles](large-language-models-llms.md#articles), [GPT](large-language-models-llms.md), [GPT3 is ZERO, ONE, FEW](../problem-framing/n-shot-learning.md#gpt3-is-zero-one-few), and [Large Language Models (LLMs)](large-language-models-llms.md).

## Prompt Engineering

This section covers automatic and context-aware prompt design.

1. [Large Language Models Are Human-Level Prompt Engineers](https://arxiv.org/abs/2211.01910) - optimizing over a set of candidate that were proposed by an LLM in order to maximize a score function. contributes to improvement of responses.
2. [Demystifying Prompts in Language Models via Perplexity](https://arxiv.org/pdf/2212.04037.pdf) - given the variability of the quality of results, how do we pick the best prompts automatically? using GPT3 and back translation to choose the lowest perplexity prompts that give the most gain in performance.
- SituatedQA. SituatedQA. [SituatedQA Incorporating linguistic context into Question Answering](https://situatedqa.github.io/)

## Step back prompting

This section describes step-back prompting (STP).

1. [STP](https://cobusgreyling.medium.com/a-new-prompt-engineering-technique-has-been-introduced-called-step-back-prompting-b00e8954cacb) - Step-Back Prompting (STP) is prompt approach in which we teach the model to answer a global questions, i.e., the original question is transformed into a stepback question, and the answer to the stepback question is used to formulate the final response.

## Chain Of Thought

This section covers chain-of-thought prompting and self-consistency.

1. [Chain Of Thought](https://arxiv.org/pdf/2201.11903.pdf) Prompting Elicits Reasoning in Large Language Models - COT is a series of intermediate reasoning steps that significantly improves the ability of large language models to perform complex reasoning, by Jason Wei Xuezhi Wang Dale Schuurmans Maarten Bosma Brian Ichter Fei Xia Ed H. Chi Quoc V. Le Denny Zhou Google Research, Brain Team.

 <figure><img src="../.gitbook/assets/image (12).png" alt=""><figcaption><p>COT, Google brain.</p></figcaption></figure>

- Self-Consistency Improves Chain of Thought Reasoning in Language Models. [self consistency improve chain of though reasoning in language models](https://arxiv.org/abs/2203.11171)

 > samples a diverse set of reasoning paths instead of only taking the greedy one, and then selects the most consistent answer by marginalizing out the sampled reasoning paths

## Prompt Tuning

This section covers parameter-efficient soft prompts and related guides.

The same notes are in [Methods](methods.md).

- The page covers the Power of Scale for Parameter-Efficient Prompt Tuning. [The power of scale for parameter efficient prompt tuning](https://arxiv.org/abs/2104.08691)
- Posted by Brian Lester, AI Resident and Noah Constant, Senior Staff Software Engineer, Google Research Large pre-trained language models, which are... [Guiding Frozen Language Models with Learned Soft Prompts](https://ai.googleblog.com/2022/02/guiding-frozen-language-models-with.html)
- Fine-tuning pre-trained models is common in NLP, but forking the model for each task can be a burden. [tweet](https://twitter.com/GoogleAI/status/1491915977138720770)
- A Comprehensive Overview of Prompt Engineering. (amazing) [prompt engineering guides](https://www.promptingguide.ai/techniques)
- 🐙 Guides, papers, lessons, notebooks and resources for prompt engineering, context engineering, RAG, and AI Agents. [github](https://github.com/dair-ai/Prompt-Engineering-Guide)

 <figure><img src="../.gitbook/assets/image (52).png" alt=""><figcaption><p>prompt engineering guides</p></figcaption></figure>

## Prompt Hacking Examples

This section points at an example of prompt hacking.

- Well, that was fast…

I just helped create the first jailbreak for ChatGPT-4 that gets around the content filters every time

credit to @vaibhavk97 for the idea, I just generalized it to make it work on ChatGPT

here's GPT-4 writing instructions on how to hack someone's computer. [Alex bert](https://twitter.com/alexalbert__/status/1636488551817965568)

## Papers

This section is the papers heading for the prompt-engineering themes covered in the sections above.

## Articles

This section lists guides and curated prompt collections.

1. Brex on [prompt engineering](https://github.com/brexhq/prompt-engineering), but goes through the history of language models which is amazing
2. [prompt engineering examples](https://www.promptingguide.ai/), a good summary of all the techniques
3. [Lilian-Weng on prompt engineering](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/) - this is a very thorough review of the topic
- GitHub - f/prompts.chat: f.k.a. Awesome ChatGPT Prompts. Share, discover, and collect prompts from the community. Free and open source — self-host for your organization with complete privacy. [Awesome Prompts on github](https://github.com/f/awesome-chatgpt-prompts)

[Using LLMs as black box classifiers](https://cohenori.medium.com/understanding-scikit-llm-86441a5af370) (May 2023) uses the prompting on this page.

[LLM Token Economy & Optimization](https://cohenori.medium.com/llm-token-economy-optimization-d1c3feea880b) (April 2023) is the cost note for a prompt.
