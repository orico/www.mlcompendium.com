# Methods

A large generative model is too big to retrain every time a task changes, so the practical question is how to adapt it cheaply. This page answers with one entry point, parameter-efficient fine-tuning (PEFT), and then points to the related notes on datasets, fine-tuning, prompt tuning, and transfer learning.

The same notes are in [DATASET SELECTION](../data/datasets.md#dataset-selection), [Fine tuning](../deep-learning/deep-neural-nets.md#fine-tuning), [Prompt Tuning](prompt.md#prompt-tuning), [TRAINING METHODOLOGIES](../data/datasets.md#training-methodologies), and [Transfer Learning using CNN](../deep-learning/convolutional-nets.md#transfer-learning-using-cnn).

Those pages explain why adapting a pretrained model works; the library that does it for generative models is Hugging Face's 🤗 PEFT. [State-of-the-art Parameter-Efficient Fine-Tuning (PEFT) methods](https://github.com/huggingface/peft) is the huggingface/peft repository, which collects state-of-the-art parameter-efficient fine-tuning methods in one place.
