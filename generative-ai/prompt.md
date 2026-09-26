# Prompt

A prompt is the only control most users have over a large language model, and the quality of the answer changes with it. This page starts with picking prompts automatically, then moves to techniques that make the model reason (step-back prompting and chain of thought), to learned soft prompts, to prompts that break the model's filters, and ends with the papers and the guides to keep open.

The same notes are in [Articles](large-language-models-llms.md#articles), [GPT](large-language-models-llms.md), [GPT3 is ZERO, ONE, FEW](../problem-framing/n-shot-learning.md#gpt3-is-zero-one-few), and [Large Language Models (LLMs)](large-language-models-llms.md).

## Prompt Engineering

Writing prompts by hand does not scale, so the first question is whether a model can choose them for you.

1. [Large Language Models Are Human-Level Prompt Engineers](https://arxiv.org/abs/2211.01910) - optimizing over a set of candidate that were proposed by an LLM in order to maximize a score function. contributes to improvement of responses.
2. [Demystifying Prompts in Language Models via Perplexity](https://arxiv.org/pdf/2212.04037.pdf) - given the variability of the quality of results, how do we pick the best prompts automatically? using GPT3 and back translation to choose the lowest perplexity prompts that give the most gain in performance.

A prompt also depends on the context the question was asked in. [SituatedQA Incorporating linguistic context into Question Answering](https://situatedqa.github.io/) is the SituatedQA project on that context.

## Step back prompting

Choosing a better prompt is one lever; asking the model to answer a more general question first is another. [STP](https://cobusgreyling.medium.com/a-new-prompt-engineering-technique-has-been-introduced-called-step-back-prompting-b00e8954cacb) - Step-Back Prompting (STP) is prompt approach in which we teach the model to answer a global questions, i.e., the original question is transformed into a stepback question, and the answer to the stepback question is used to formulate the final response.

## Chain Of Thought

Stepping back is one way to structure reasoning; writing the reasoning out step by step is the better-known one. [Chain Of Thought](https://arxiv.org/pdf/2201.11903.pdf) Prompting Elicits Reasoning in Large Language Models - COT is a series of intermediate reasoning steps that significantly improves the ability of large language models to perform complex reasoning, by Jason Wei Xuezhi Wang Dale Schuurmans Maarten Bosma Brian Ichter Fei Xia Ed H. Chi Quoc V. Le Denny Zhou Google Research, Brain Team.

 <figure><img src="../.gitbook/assets/image (12).png" alt=""><figcaption><p>COT, Google brain.</p></figcaption></figure>

The figure is the chain-of-thought example from Google Brain. One reasoning path can still go wrong, so [self consistency improve chain of though reasoning in language models](https://arxiv.org/abs/2203.11171) replaces the naive greedy decoding used in chain-of-thought prompting with a new decoding strategy, self-consistency, which:

 > samples a diverse set of reasoning paths instead of only taking the greedy one, and then selects the most consistent answer by marginalizing out the sampled reasoning paths

## Prompt Tuning

All of the above are written prompts. Prompt tuning learns the prompt instead, as a few vectors trained while the model stays frozen.

The same notes are in [Methods](methods.md).

[The power of scale for parameter efficient prompt tuning](https://arxiv.org/abs/2104.08691) is the paper: "soft prompts" are learned through backpropagation to condition frozen language models on specific downstream tasks, unlike the discrete text prompts used by GPT-3, and they can absorb signal from any number of labeled examples. [Guiding Frozen Language Models with Learned Soft Prompts](https://ai.googleblog.com/2022/02/guiding-frozen-language-models-with.html) is the Google Research blog post by Brian Lester and Noah Constant. The Google AI [tweet](https://twitter.com/GoogleAI/status/1491915977138720770) gives the short version: fine-tuning pre-trained models is common in NLP, but forking the model for each task can be a burden, and prompt tuning adds a small set of learnable vectors to the input that can match fine-tuning quality while every task shares the same frozen model.

For the whole range of techniques, written and learned, (amazing) [prompt engineering guides](https://www.promptingguide.ai/techniques) is a comprehensive overview of prompt engineering, and its source is on [github](https://github.com/dair-ai/Prompt-Engineering-Guide) as dair-ai's guides, papers, lessons, notebooks, and resources for prompt engineering, context engineering, RAG, and AI agents.

 <figure><img src="../.gitbook/assets/image (52).png" alt=""><figcaption><p>prompt engineering guides</p></figcaption></figure>

## Prompt Hacking Examples

The same control that steers a model can also be used against its filters. [Alex bert](https://twitter.com/alexalbert__/status/1636488551817965568) is Alex Albert's post: "Well, that was fast… I just helped create the first jailbreak for ChatGPT-4 that gets around the content filters every time. credit to @vaibhavk97 for the idea, I just generalized it to make it work on ChatGPT. here's GPT-4 writing instructions on how to hack someone's computer."

## Papers

The papers for these themes are the ones already linked in the sections above, from automatic prompt engineering through chain of thought to prompt tuning.

## Articles

After the papers, these are the guides and collections to keep open while writing prompts.

1. Brex on [prompt engineering](https://github.com/brexhq/prompt-engineering), but goes through the history of language models which is amazing
2. [prompt engineering examples](https://www.promptingguide.ai/), a good summary of all the techniques
3. [Lilian-Weng on prompt engineering](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/) - this is a very thorough review of the topic. Lilian Weng treats prompt engineering, also called in-context prompting, as steering an LLM toward a desired outcome without updating its weights, an empirical science that needs heavy experimentation because its effects vary a lot between models.
4. [Awesome Prompts on github](https://github.com/f/awesome-chatgpt-prompts) is f/prompts.chat, formerly Awesome ChatGPT Prompts, a free and open-source place to share, discover, and collect community prompts that an organization can self-host.

Two notes put these prompts to work. [Using LLMs as black box classifiers](https://cohenori.medium.com/understanding-scikit-llm-86441a5af370) (May 2023) uses the prompting on this page.

[LLM Token Economy & Optimization](https://cohenori.medium.com/llm-token-economy-optimization-d1c3feea880b) (April 2023) is the cost note for a prompt.
