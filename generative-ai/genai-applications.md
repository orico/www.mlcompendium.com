# GenAI Applications

Models, prompts, and retrieval only matter once they become something a person can use. This page walks through applications built with generative models, starting with personalized image generation, then text-to-SQL and chatting with documents, then domain and multimodal applications, and ending with private document chat, security, and messaging bots.

The first application is personalization: generating a specific subject in new scenes. [DreamBooth](https://dreambooth.github.io/) is the project that does it for text-to-image diffusion.

The same notes are in [Diffusion Models](diffusion-models.md).

 > Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation

 > It’s like a photo booth, but once the subject is captured, it can be synthesized wherever your dreams take you…

Text generation turns the same idea toward data and documents. [Analytical bot, ask a question get an sql query response](https://www.youtube.com/watch?v=fiQy276sd18&list=PL1FoIGqsXJ_Dd0twE-V9bwzw2FeZHZmtF) is the Machine & Deep Learning Israel talk "Democratize data and information with text-to-code models (Hebrew)". For documents, [how to chat with a document](https://www.youtube.com/watch?v=ih9PBGVVOO4) Youtube - In this video, we'll demonstrate how to use OpenAI's GPT-4 API to interact with a 56-page PDF of a Supreme Court legal case. You'll learn about GPT-4's enhanced capabilities, including processing up to 25,000 words and handling more complex instructions. We'll cover using LangChain to assemble chatbot components and Pinecone to store documents as numerical vectors. Finally, we'll show how to create a chat interface to display results alongside the source documents. This approach can be applied to building chatbots for various file formats like PDFs, websites, and Excel sheets.

The same notes are in [RAG](rag.md).

Images come back in consumer form: [Upload a photo of your room to generate your dream room with AI.](https://github.com/Nutlope/roomGPT) is Nutlope's roomGPT. In a scientific domain, [BioGPT](https://github.com/microsoft/BioGPT) is Microsoft's repository for [Generative Pre-trained Transformer for Biomedical Text Generation and Mining](https://academic.oup.com/bib/advance-article/doi/10.1093/bib/bbac409/6713511), by Renqian Luo, Liai Sun, Yingce Xia, Tao Qin, Sheng Zhang, Hoifung Poon and Tie-Yan Liu. Generation can also produce structure: [Extrapolating knowledge graphs from unstructured text using GPT-3](https://github.com/varunshenoy/GraphGPT) is varunshenoy's GraphGPT 🕵️‍♂️.

The same notes are in [Knowledge Graphs](../language-ai/knowledge-graphs.md).

Chatting with documents is the application that repeats most, in different shapes. [Interact with your documents using the power of GPT, 100% privately, no data leaks](https://github.com/zylon-ai/private-gpt) is zylon-ai's private-gpt, now a complete API layer for private AI applications on local models, with RAG, skills, tools, MCP, text-to-sql, and more, working with any OpenAI-compatible inference server. [GPT4 & LangChain Chatbot for large PDF docs](https://github.com/mayooear/gpt4-pdf-chatbot-langchain) is an AI PDF chatbot agent built with LangChain and LangGraph. [GPT-powered chat for documentation, chat with your documents](https://github.com/arc53/DocsGPT) is DocsGPT, now a private AI platform for agents, assistants, and enterprise search, with an agent builder, deep research, document analysis, multi-model support, and API connectivity for agents. [PDF GPT allows you to chat with the contents of your PDF file by using GPT capabilities. The most effective open source solution to turn your pdf files in a chatbot!](https://github.com/bhaskatripathi/pdfGPT) is bhaskatripathi's pdfGPT.

The same pattern extends to sound, security, and messaging. [AudioGPT: Understanding and Generating Speech, Music, Sound, and Talking Head](https://github.com/AIGC-Audio/AudioGPT) is the AIGC-Audio repository for that model. [A GPT-empowered penetration testing tool](https://github.com/GreyDGL/PentestGPT) is PentestGPT, an automated penetration testing agentic framework powered by large language models. [Whatsapp GPT](https://github.com/danielgross/whatsapp-gpt) is danielgross's repository for putting GPT inside WhatsApp.

[Developing A Virtual Psychologist With Gen-AI](https://pub.towardsai.net/developing-a-virtual-psychologist-with-gen-ai-f2e87c7d7c28) (October 2024) is an application on this page, and the constraint on it lives in Responsible AI. It describes Gen-AI services built to make mental-health support less daunting and out of reach: a virtual chatbot designed to provide therapeutic assistance and emotional support.

## Deprecated links

{% hint style="warning" %}
These links and images no longer work. The original wording is kept here. A same-resource copy, when one was checked, is used above.
{% endhint %}

- Generative Pre-trained Transformer for Biomedical Text Generation and Mining, by Renqian Luo, Liai Sun, Yingce Xia, Tao Qin, Sheng Zhang, Hoifung Poon and Tie-Yan Liu. This address no longer opens: https://academic.oup.com/bib/advance-article/doi/10.1093/bib/bbac409/6713511?guestAccessKey=a66d9b5d-4f83-4017-bb52-405815c907b9
