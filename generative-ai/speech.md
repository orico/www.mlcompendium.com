# Speech

Generating from speech starts with turning audio into text, and a speech model is only useful once you can measure how often it gets the words wrong. The page goes from the model, Whisper, to the evaluation method, word error rate.

The same notes are in [Basics](../predictive-ml/audio-basics.md), [Deep Neural Audio](../deep-learning/deep-neural-audio.md), [Mix N Match](mix-n-match.md), and [Other Tools](../predictive-ml/audio-algorithms.md#other-tools).

## Models

The model to start from is OpenAI's Whisper. [Whisper](https://openai.com/research/whisper) is the "Introducing Whisper" announcement, and the code is on [GitHub](https://github.com/openai/whisper) as openai/whisper, "Robust Speech Recognition via Large-Scale Weak Supervision".

## Evaluation Methods

A transcript from that model still has to be scored against a reference.

1. [Jiwer](https://github.com/jitsi/jiwer) - Evaluate your speech-to-text system with similarity measures such as word error rate (WER)
