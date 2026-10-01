# Generative AI

Generation is how the model meets a user who did not bring a labeled table.
After this chapter the reader can trace a prompt, a retrieval step, an agent, and a generated modality back to the model that produces it.
The chapter starts with the methods and models that generate, moves to the systems built around them, and ends with the surfaces where a user actually meets the output.

## Methods

The first stop is how a generative model is adapted and what kinds of model generate at all. [Methods](methods.md) is the parameter-efficient fine-tuning entry point, the way to adapt a large model without retraining all of it. [Diffusion Models](diffusion-models.md) is the introductory reading on diffusion, the model family behind most image generation. [Large Language Models (LLMs)](large-language-models-llms.md) is the page for the text models themselves. [Prompt](prompt.md) is how you steer those models once they exist, from prompt writing to prompt tuning.

## Systems

Once a model can generate, the next step is to wrap it in a system that brings in knowledge and takes actions. [RAG](rag.md) collects retrieval-augmented generation tutorials and Graph RAG notes, the way a model answers from your documents. [Agents](agents.md) collects frameworks and products for tool-using and multi-agent LLM setups. [Research and Reviewer Agents](research-reviewer-agents.md) sorts agents that do scientific research, agents that also review what they produced, and agents that only review.

## Surfaces

The systems finally reach a user through a modality or an interface. [Speech](speech.md) covers speech recognition models and how to evaluate them with word error rate. [Vision](vision.md) points at DINOv2, a self-supervised vision foundation model. [Mix N Match](mix-n-match.md) is where vision, language, and speech tools are combined into one system. [GenAI Applications](genai-applications.md) collects applications built with generative models, from DreamBooth personalization to chatting with a PDF. [Chat UI/UX](chat-ui-ux.md) is the ready chat interfaces that sit over an LLM backend. [Chat Bots](chat-bots.md) is a chatbot survey with cross-links into the GPT and intent-recognition notes. [Conversation](conversation.md) points at Cornell ConvoKit for analyzing the conversations those bots take part in. [Gen AI Industry](generative-ai.md) closes the chapter with overviews of the generative-AI landscape.


See also [Understanding PandasAI](https://cohenori.medium.com/understanding-pandasai-fc135b871e84) (May 2023), which looks at Pandas-AI, a Python library that brings generative AI into Pandas so data frames become conversational, and at how any LLM can be steered to answer data questions.
