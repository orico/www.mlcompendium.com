# Responsible AI

A shipped score can be miscalibrated, opaque, or unfair, and a prompt can be too.
After this chapter the reader can point at calibration, explanation, fairness, unlearning, and federated training as the constraints on a model that already exists.

The chapter starts with the score itself. [Calibration](calibration.md) explains why predicted probabilities need calibration and how to adjust them, from classical methods to neural-net approaches such as temperature scaling. Once the numbers can be trusted, the next question is why the model produced them: [Interpretable & Explainable AI (XAI)](interpretable-and-explainable-ai-xai.md) gathers the courses, libraries such as LIME, Anchor, and SHAP, and articles on explaining and interpreting ML models.

An explained model can still treat people unequally. [Fairness, Accountability, and Transparency](fairness-accountability-and-transparency.md) covers regulation, FAT research, bias, debiasing, fairness tools, and privacy. The same concerns reach language models through their inputs, so [Fairness, Accountability, and Transparency In Prompts](fairness-accountability-and-transparency-in-prompts.md) collects papers on debiasing with prompts and on LLM hallucinations.

The last two pages are about the data behind the model. [Unlearning](unlearning.md) is the challenge of erasing a data point's influence on the input-output mapping of an ML model, and [Federated Learning](federated-learning.md) points to a guide on training across centralized and decentralized setups.


See also [Developing A Virtual Psychologist With Gen-AI](https://pub.towardsai.net/developing-a-virtual-psychologist-with-gen-ai-f2e87c7d7c28) (October 2024), a walk-through of building a Gen-AI chatbot that offers therapeutic assistance and emotional support, a case where every one of these constraints applies at once.
