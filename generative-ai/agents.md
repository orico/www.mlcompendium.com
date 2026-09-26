# Agents

An agent is an LLM that uses tools, talks to other agents, and acts on its own, and building one starts with picking a framework. The list below goes from a multi-agent framework, to routing queries between agents inside a product, to deploying autonomous agents in a browser, and closes with two notes on what agents do to systems and organizations.

The same notes are in [Chat Bots](chat-bots.md) and [Tools](large-language-models-llms.md#tools).

1. [AutoGen](https://github.com/microsoft/autogen) by [microsoft](https://microsoft.github.io/autogen/) is a programming framework for agentic AI. In its own words, it

   > is a framework that enables the development of LLM applications using multiple agents that can converse with each other to solve tasks. AutoGen agents are customizable, conversable, and seamlessly allow human participation. They can operate in various modes that employ combinations of LLMs, human inputs, and tools.

   <figure><img src="../.gitbook/assets/image (49).png" alt=""><figcaption><p>AutoGen</p></figcaption></figure>

2. Once there are several agents, something has to decide which one gets the query. [Routing tools/agents](https://www.linkedin.com/blog/engineering/generative-ai/musings-on-building-a-generative-ai-product) - an overall multi tool article, with a routing example - decides if the query is in scope or not, and which AI agent to forward it to. Examples of agents are: job assessment, company understanding, takeaways for posts, etc.
3. To try autonomous agents without building the plumbing, AgentGPT - [Assemble, configure, and deploy autonomous AI Agents in your browser](https://github.com/reworkd/AgentGPT) is reworkd's 🤖 tool for doing exactly that. An agent-operations tool for evals, observability, and replays used to be listed here; that address no longer opens and is kept at the end of the page.

Once agents run in production, they change the systems and teams around them. Two notes take that up. See also [Self-Healing Agentic Systems](https://cohenori.medium.com/the-rise-of-self-healing-systems-fe653869b7fc) (January 2026), which starts from modern distributed, event-driven, hybrid and multi-cloud systems that generate more signals, dependencies, and failure modes than human operators can manage in real time. See also [Agents-Driven Organizations](https://cohenori.medium.com/agents-driven-organizations-4bc300fc5283) (January 2026), by Dr. Ori Cohen, which describes integrating agents as team members rather than as a subordinate copilot, and examines what that enables, where it creates pressure, and the organizational dynamics it introduces.

## Deprecated links

{% hint style="warning" %}
These links and images no longer work. The original wording is kept here. A same-resource copy, when one was checked, is used above.
{% endhint %}

- AgentOps - Build your next agent with evals, observability, and replays. This address no longer opens: https://app.agentops.ai/start
