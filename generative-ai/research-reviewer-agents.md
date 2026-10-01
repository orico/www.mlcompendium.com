# Research and Reviewer Agents

A research agent proposes questions, searches the literature, and runs experiments, and a reviewer agent checks a manuscript, a claim, or a proof. This page sorts those systems into research only, research plus a reviewer, and reviewer only.

## Research

[CORAL](https://arxiv.org/abs/2604.01658) (Ao Qu, Han Zheng, Zijian Zhou, and colleagues, MIT and the National University of Singapore, 2026) is a multi-agent system that evolves open-ended discoveries, with code at [Human-Agent-Society/CORAL](https://github.com/Human-Agent-Society/CORAL).

[Robin](https://www.nature.com/articles/s41586-026-10652-y) (Ghareeb and colleagues, FutureHouse, published 19 May 2026; preprint [arXiv:2505.13400](https://arxiv.org/abs/2505.13400)) is a multi-agent system that proposes biological hypotheses, designs experiments, and analyzes the data, with code at [Future-House/robin](https://github.com/Future-House/robin) and a write-up at [FutureHouse](https://www.futurehouse.org/research/demonstrating-end-to-end-scientific-discovery-with-robin-a-multi-agent-system).

[ERA](https://www.nature.com/articles/s41586-026-10658-6) (Aygün and colleagues, Google, published 19 May 2026; preprint [arXiv:2509.06503](https://arxiv.org/abs/2509.06503)) is a system that searches programs with a language model so the code maximizes a scientific quality metric, described at [Google Research](https://research.google/pubs/an-ai-system-to-help-scientists-write-expert-level-empirical-software/).

[Biomni](https://www.science.org/doi/10.1126/science.adz4351) (Kexin Huang, Serena Zhang, Hanchen Wang, and colleagues, Stanford, 2025) is a general-purpose biomedical agent that composes tools, databases, and protocols, with code at [snap-stanford/biomni](https://github.com/snap-stanford/biomni) and a demo at [biomni.stanford.edu](https://biomni.stanford.edu).

[The Virtual Biotech](https://www.science.org/doi/10.1126/science.aeg6779) (Harrison G. Zhang, Peter Eckmann, Jiacheng Miao, Andrew B. Mahon, and James Zou, Science, 17 September 2026) is a multi-agent framework that organizes therapeutic discovery the way a small drug company does.

[OpenResearch](https://github.com/alphaXiv/OpenResearch) (alphaXiv, 2026) is a local workspace that turns coding agents such as Claude Code, Codex, Cursor, and Antigravity into research agents that search the literature and keep a project memory.

[GIANTS-4B](https://arxiv.org/abs/2604.09793) (Joy He-Yueya, Anikait Singh, and colleagues, 2026) is a 4-billion-parameter model that predicts a later paper’s insight from its parent papers, with a site at [giants-insights.github.io](https://giants-insights.github.io/) and weights at [giants2026/GIANTS-4B](https://huggingface.co/giants2026/GIANTS-4B).

[Paper2Agent](https://www.nature.com/articles/s41586-026-11044-y) (Jiacheng Miao, Joe R. Davis, Yaohui Zhang, Jonathan K. Pritchard, and James Zou, Nature, 16 September 2026) turns a research paper into an agent that can run the paper’s tools, with a news piece the same day at [Nature](https://www.nature.com/articles/d41586-026-02899-2).

[GPT-Rosalind](https://share.google/XZhuqaCye7fxsdaGu) (OpenAI, announced 3 June 2026) is a life-sciences model for drug-discovery workflows, including binding, genomics, and experimental design.

[Science Skills](https://github.com/google-deepmind/science-skills) (Google DeepMind, 2026) is a pack of tool skills, covering UniProt, AlphaFold, AlphaGenome, and related databases, written up in the [Science Skills report](https://storage.googleapis.com/deepmind-media/papers/google_deepmind_science_skills_for_antigravity_towards_efficient_and_reliable_scientific_workflows.pdf) and named again in the [Gemini for Science](https://share.google/tHZKsVN6c5kq9sTqC) post.

[PaperVizAgent](https://arxiv.org/abs/2601.23265) (Dawei Zhu, Rui Meng, Yale Song, and colleagues, Google, 2026), also called PaperBanana, is a five-agent system that draws scientific figures and has a critic that checks those figures, with code at [google-research/papervizagent](https://github.com/google-research/papervizagent) and a shared announcement with ScholarPeer at [Google Research](https://research.google/blog/improving-the-academic-workflow-introducing-two-ai-agents-for-better-figures-and-peer-review/).

[OpenScholar](https://www.nature.com/articles/s41586-025-10072-4) (Akari Asai, Jacqueline He, Rulin Shao, Weijia Shi, and colleagues, University of Washington and the Allen Institute for AI, published online 4 February 2026) synthesizes the scientific literature with retrieval and citation-backed answers, with code at [AkariAsai/OpenScholar](https://github.com/AkariAsai/OpenScholar) and a demo at [open-scholar.allen.ai](https://open-scholar.allen.ai/).

[PaperQA2](https://arxiv.org/abs/2409.13740) (Michael D. Skarlinski, Sam Cox, and colleagues, FutureHouse, 2024) is an agent that retrieves and synthesizes scientific papers, with code at [Future-House/paper-qa](https://github.com/future-house/paper-qa) and a write-up at [FutureHouse](https://www.futurehouse.org/research/engineering-blog-journey-to-superhuman-performance-on-scientific-tasks).

[literature_reviewer_agent](https://github.com/jykr/literature_reviewer_agent) (2026) is a concierge that ranks recent papers in a field and returns a literature review.

[NotebookLM](https://share.google/OUUUStYSbRZBZ92Xy) (Trond Wuellner and Usama Bin Shafqat, Google, 8 June 2026) adds an agent that runs code and searches the web over a user’s sources.

[LeapSpace](https://www.elsevier.com/products/leapspace) (Elsevier) is a research workspace whose agentic features include Claim Radar and a Writing Coach over Elsevier’s peer-reviewed literature, announced in the [LeapSpace press release](https://www.elsevier.com/about/press-releases/elsevier-expands-leapspace-with-new-agentic-capabilities-for-tasks-across-the-complete-research-workflow).

[Scholar Labs](https://scholar.googleblog.com/2025/11/scholar-labs-ai-powered-scholar-search.html) (Sam Yuan, Alex Verstak, Hanshen Wang, Akash Sethi, Namit Shetty, and Anurag Acharya, Google Scholar, 18 November 2025) is an AI search that answers a research question from the Scholar index, also described at [blog.google](https://blog.google/products-and-platforms/products/education/google-scholar-labs/).

## Research and review

[ScientistTwo](https://arxiv.org/abs/2609.19644) (Jaehyun Nam, Jinsung Yoon, Yanzhou Pan, Yubo Wang, Rui Meng, Parthasarathy Ranganathan, and Tomas Pfister, Google Cloud AI Research, 2026) runs an autonomous discovery cycle and then a simulated peer-review rebuttal, with a site at [scientist-two.github.io](https://scientist-two.github.io/).

[The AI Scientist](https://www.nature.com/articles/s41586-026-10265-5) (Chris Lu, Cong Lu, Robert Tjarko Lange, Yutaro Yamada, and colleagues, Sakana AI, Nature 2026) writes a full paper and then reviews it, with the 2024 system at [SakanaAI/AI-Scientist](https://github.com/SakanaAI/AI-Scientist) ([arXiv:2408.06292](https://arxiv.org/abs/2408.06292)), the 2025 tree-search system at [SakanaAI/AI-Scientist-v2](https://github.com/SakanaAI/AI-Scientist-v2) ([arXiv:2504.08066](https://arxiv.org/abs/2504.08066)), a 2026 continuation at [findalexli/ai-scientist-v3](https://github.com/findalexli/ai-scientist-v3) and the [Hugging Face write-up](https://huggingface.co/blog/alexshengzhili/aiscientist), and an essay at [The Conversation](https://share.google/zdyBISXMTBREjorxs).

[PaperOrchestra](https://arxiv.org/abs/2604.05018) (Yiwen Song, Yale Song, Tomas Pfister, and Jinsung Yoon, Google Cloud AI Research, 2026) writes a paper from experiment logs and keeps a revision only when a simulated review score rises, with code at [google-research/paper-orchestra](https://github.com/google-research/paper-orchestra) and a community skill pack at [Ar9av/PaperOrchestra](https://github.com/Ar9av/PaperOrchestra).

[academic-research-skills](https://github.com/Imbad0202/academic-research-skills) (2026, [Zenodo 10.5281/zenodo.20696614](https://doi.org/10.5281/zenodo.20696614)) is a Claude Code skill pack that runs a research team, a writing team, and a seven-agent review panel of a journal-fit check, three reviewers, and a devil’s advocate.

[Brain Researcher](https://arxiv.org/abs/2608.19902) (Zijiao Chen, Nicholas Lu, Russell A. Poldrack, and colleagues, Stanford, 2026) is a neuroimaging harness whose scientific-review stage classifies each claim as accepted, qualified, revised, blocked, rejected, or deferred.

[CycleResearcher](https://arxiv.org/abs/2411.00816) (Yixuan Weng, Minjun Zhu, Guangsheng Bao, and colleagues, Westlake University, ICLR 2025) writes papers and scores them with CycleReviewer, with code at [zhu-minjun/Researcher](https://github.com/zhu-minjun/Researcher), the same repository as DeepReviewer.

[Co-Scientist](https://www.nature.com/articles/s41586-026-10644-y) (Juraj Gottweis, Wei-Hung Weng, and colleagues, Google DeepMind, Nature 2026; preprint [arXiv:2502.18864](https://arxiv.org/abs/2502.18864)) generates and critiques research hypotheses with a Reflection agent, with a 2026 validation follow-up at [arXiv:2608.26701](https://arxiv.org/abs/2608.26701), community code at [Kaimen-Inc/Co-Scientist](https://github.com/Kaimen-Inc/Co-Scientist), a news piece at [Nature](https://www.nature.com/articles/d41586-026-02931-5), and the DeepMind essay [Conjecture Machines](https://share.google/Ug9D3erJK3dcIalHq).

[AutoScientists](https://arxiv.org/abs/2605.28655) (Shanghua Gao, Ada Fang, and Marinka Zitnik, Harvard, 2026) is a self-organizing lab whose peers critique experiment proposals before compute is spent, with code at [mims-harvard/AutoScientists](https://github.com/mims-harvard/AutoScientists) and a site at [autoscientists.openscientist.ai](https://autoscientists.openscientist.ai).

[Agent Laboratory](https://arxiv.org/abs/2501.04227) (Samuel Schmidgall, Yusheng Su, and colleagues, AMD and Johns Hopkins, EMNLP Findings 2025) is a set of research-assistant agents whose report stage calls a reviewer, with code at [SamuelSchmidgall/AgentLaboratory](https://github.com/SamuelSchmidgall/AgentLaboratory).

[The AI co-mathematician](https://arxiv.org/abs/2605.06651) (Daniel Zheng, Ingrid von Glehn, Yori Zwols, and colleagues, Google DeepMind, 2026) is a mathematics workbench whose adversarial review loops check proofs.

[Claude Science](https://share.google/M4AKxjHS4SXmZIqsX) (Anthropic, announced 30 June 2026) is a research workbench whose fact-checker checks citations and calculations before a result is published.

## Review

[The Paper Assistant Tool](https://arxiv.org/abs/2606.28277) (Rajesh Jayaram, Drew Tyler, David Woodruff, Corinna Cortes, Yossi Matias, Vahab Mirrokni, and Vincent Cohen-Addad, Google Research, 2026) reviews full manuscripts and was piloted at STOC and ICML, and it is also named in the [Gemini for Science](https://share.google/tHZKsVN6c5kq9sTqC) post.

[ScholarPeer](https://arxiv.org/abs/2601.22638) (Palash Goyal, Mihir Parmar, Yiwen Song, Hamid Palangi, Tomas Pfister, and Jinsung Yoon, Google, 2026) runs a historian, a baseline scout, and a question-and-answer engine and then writes a review, with the [Google Research page](https://research.google/pubs/scholarpeer-a-multi-agent-framework-for-automated-peer-review/), the [shared blog post](https://research.google/blog/improving-the-academic-workflow-introducing-two-ai-agents-for-better-figures-and-peer-review/), and community code at [amirkiarafiei/open-scholar-peer](https://github.com/amirkiarafiei/open-scholar-peer).

[ProReviewer](https://arxiv.org/abs/2606.13349) (Haishuo Fang, Yue Feng, and Iryna Gurevych, UKP Lab, 11 June 2026) writes a review step by step and records the path in a review log, with code at [UKPLab/arxiv2026-ProReviewer](https://github.com/UKPLab/arxiv2026-ProReviewer), weights at [UKPLab/ProReviewer-8B](https://huggingface.co/UKPLab/ProReviewer-8B), and data at [UKPLab/ProReviewer-Dataset](https://huggingface.co/datasets/UKPLab/ProReviewer-Dataset).

[MARG](https://arxiv.org/abs/2401.04259) (Mike D’Arcy, Tom Hope, Larry Birnbaum, and Doug Downey, Northwestern and the Allen Institute for AI, 2024) generates review comments on experiments, clarity, and impact, with code at [allenai/marg-reviewer](https://github.com/allenai/marg-reviewer).

[DeepReview](https://arxiv.org/abs/2503.08569) (Minjun Zhu, Yixuan Weng, Linyi Yang, and Yue Zhang, Westlake University and UCL, ACL 2025) is a trained reviewer that retrieves related literature while it reviews, with code in [zhu-minjun/Researcher](https://github.com/zhu-minjun/Researcher), and [DeepReviewer 2.0](https://arxiv.org/abs/2604.09590) (2026) continues it at [ResearAI/DeepReviewer-v2](https://github.com/ResearAI/DeepReviewer-v2) and [deepscientist.cc](https://deepscientist.cc).

[ReviewGrounder](https://github.com/EigenTom/ReviewGrounder) (Zhuofeng Li, Yi Lu, and colleagues, Texas A&M, ACL 2026) writes a rubric-guided review with literature tools, with a site at [eigentom.github.io/ReviewGrounder](https://eigentom.github.io/ReviewGrounder/).

[AgentReview](https://github.com/ahren09/agentreview) (Yiqiao Jin, Qinlin Zhao, and colleagues, Georgia Tech, EMNLP 2024) simulates a review process with reviewer, author, and area-chair agents, with a site at [agentreview.github.io](https://agentreview.github.io/).

[reviewing-agents](https://github.com/togethercomputer/reviewing-agents) (Bianchi, Zou, and colleagues, Together AI and Stanford, 2025) is the reviewer stack used for Agents4Science, with a main reviewer plus checkers for correctness, format, jailbreaks, and references.

[The Stanford Agentic Reviewer](https://paperreview.ai/) (Yixing Jiang and Andrew Ng, Stanford, 2025) reviews a paper against the arXiv literature, with a [technical overview](https://paperreview.ai/tech-overview).

[QED Science](https://www.qedscience.com/) (2026) breaks a life-science manuscript into claims and scores originality and validity.

[Agentic_Paper](https://github.com/albertogerli/Agentic_Paper) (Alberto G. Gerli, 2026) runs twelve reviewer agents in parallel and then a coordinator and an editor, and it checks citations through OpenAlex.

[OpenReviewer](https://github.com/AliManjotho/open-reviewer) (Ali Asghar Manjotho, Mehran University, 2026) is a multi-agent pre-submission review of a manuscript.

[OpenReviewer](https://github.com/maxidl/openreviewer) (Maximilian Idahl and Zahra Ahmadi, NAACL 2025 demonstration) is a model tuned to write critical reviews of computer-science papers.

[coarse](https://github.com/Davidvandijcke/coarse) (2026) is an open-source review pipeline that reads the PDF, runs a three-judge panel and section agents, and checks quoted text against the source, with a site at [coarse.vercel.app](https://coarse.vercel.app).

[Reviewer2](https://arxiv.org/abs/2402.10886) (Zhaolin Gao, Kianté Brantley, and Thorsten Joachims, Cornell, 2024) generates peer-review text by searching over prompts, with code at [ZhaolinGao/Reviewer2](https://github.com/ZhaolinGao/Reviewer2).

[paper-agents-manuscript](https://github.com/bdsp-core/paper-agents-manuscript) (2026) reviews a draft with a panel of agents, two of which query Google Scholar for missing and mismatched citations.

[Nature Research Assistant](https://researchassistant.nature.com/library/manuscript) (Springer Nature) gives writing feedback on style, structure, and substance for an uploaded manuscript, and Springer Nature states that the advice is separate from editorial decisions.

Clinical agents, benchmarks, and essays from the same survey sit outside these three groups.

The same notes are in [Agents](agents.md) and [Large Language Models (LLMs)](large-language-models-llms.md#use-cases).
