# Prompt

This page collects prompt-engineering articles, papers, and techniques.

The same notes are in [Articles](large-language-models-llms.md#articles), [GPT](gpt.md), [GPT3 is ZERO, ONE, FEW](../types-of-machine-learning/n-shot-learning.md#gpt3-is-zero-one-few), and [Large Language Models (LLMs)](large-language-models-llms.md).

## Articles

This section lists guides and curated prompt collections.

1. Brex on [prompt engineering](https://github.com/brexhq/prompt-engineering), but goes through the history of language models which is amazing
2. [prompt engineering examples](https://www.promptingguide.ai/), a good summary of all the techniques
3. [Lilian-Weng on prompt engineering](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/) - this is a very thorough review of the topic
4. [Awesome Prompts on github](https://github.com/f/awesome-chatgpt-prompts)

## Papers

This section groups papers on prompt engineering, tuning, chain of thought, hacking, and step-back prompting.

### Prompt Engineering

This subsection covers automatic and context-aware prompt design.

1. [Large Language Models Are Human-Level Prompt Engineers](https://arxiv.org/abs/2211.01910) - optimizing over a set of candidate that were proposed by an LLM in order to maximize a score function. contributes to improvement of responses.
2. [Demystifying Prompts in Language Models via Perplexity](https://arxiv.org/pdf/2212.04037.pdf) - given the variability of the quality of results, how do we pick the best prompts automatically? using GPT3 and back translation to choose the lowest perplexity prompts that give the most gain in performance.
3. [SituatedQA Incorporating linguistic context into Question Answering](https://situatedqa.github.io/)

### Prompt Tuning

This subsection covers parameter-efficient soft prompts and related guides.

The same notes are in [Methods](methods.md).

1. [The power of scale for parameter efficient prompt tuning](https://arxiv.org/abs/2104.08691) - it becomes more competitive at scale.
2. [Guiding Frozen Language Models with Learned Soft Prompts](https://ai.googleblog.com/2022/02/guiding-frozen-language-models-with.html), [tweet](https://twitter.com/GoogleAI/status/1491915977138720770)
3. (amazing) [prompt engineering guides](https://www.promptingguide.ai/techniques), [github](https://github.com/dair-ai/Prompt-Engineering-Guide)

   <figure><img src="../.gitbook/assets/image (52).png" alt=""><figcaption><p>prompt engineering guides</p></figcaption></figure>

### Chain Of Thought

This subsection covers chain-of-thought prompting and self-consistency.

1. [Chain Of Thought](https://arxiv.org/pdf/2201.11903.pdf) Prompting Elicits Reasoning in Large Language Models - COT is a series of intermediate reasoning steps that significantly improves the ability of large language models to perform complex reasoning, by Jason Wei Xuezhi Wang Dale Schuurmans Maarten Bosma Brian Ichter Fei Xia Ed H. Chi Quoc V. Le Denny Zhou Google Research, Brain Team.

   <figure><img src="../.gitbook/assets/image (12).png" alt=""><figcaption><p>COT, Google brain.</p></figcaption></figure>

2. [self consistency improve chain of though reasoning in language models](https://arxiv.org/abs/2203.11171)

   > samples a diverse set of reasoning paths instead of only taking the greedy one, and then selects the most consistent answer by marginalizing out the sampled reasoning paths

### Prompt Hacking Examples

This subsection points at an example of prompt hacking.

1. [Alex bert](https://twitter.com/alexalbert__/status/1636488551817965568)

### Step back prompting

This subsection describes step-back prompting (STP).

1. [STP](https://cobusgreyling.medium.com/a-new-prompt-engineering-technique-has-been-introduced-called-step-back-prompting-b00e8954cacb) - Step-Back Prompting (STP) is prompt approach in which we teach the model to answer a global questions, i.e., the original question is transformed into a stepback question, and the answer to the stepback question is used to formulate the final response.
