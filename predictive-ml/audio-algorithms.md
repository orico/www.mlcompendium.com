# Algorithms

This page collects algorithms and tools for sound event detection, source separation, embeddings, and related audio tasks. Use it after the terminology and feature pages when you need concrete methods and packages.


## Sound Event Detection

This section is about Sound Event Detection models and labels.

The same notes are in [Terminology](audio-terminology.md).

- Models and examples built with TensorFlow. [YamNet](https://github.com/tensorflow/models/tree/master/research/audioset/yamnet)
- and real-time sound event detection. [GitHub](https://github.com/robertanto/Real-Time-Sound-Event-Detection)
- [Event types labels list](https://github.com/robertanto/Real-Time-Sound-Event-Detection/blob/main/keras_yamnet/yamnet_class_map.csv) and real time sound event detection Relevant labels 420 430

## Query-based separation

This section is about query-based audio source separation.

1. [Zero Shot Audio Source Separation](https://github.com/RetroCirce/Zero_Shot_Audio_Source_Separation), [paper](https://arxiv.org/abs/2112.07891), [interface](https://replicate.com/retrocirce/zero_shot_audio_source_separation) — a three-component pipeline that lets you train an audio source separator to separate any source from the track. All you need is a mixture audio to separate, and a given source sample as a query. Then the model will separate your specified source from the track.

## Audio Source Separation

This section lists open-domain and Wave-U-Net source separation models.

The same notes are in [Terminology](audio-terminology.md).

1. [Audio Sep](https://github.com/Audio-AGI/AudioSep) — AudioSep is a foundation model for open-domain sound separation with natural language queries. AudioSep demonstrates strong separation performance and impressive zero-shot generalization ability on numerous tasks such as audio event separation, musical instrument separation, and speech enhancement.
2. Wave-U-Net
 - Implementation of the Wave-U-Net for audio source separation - f90/Wave-U-Net. [Original](https://github.com/f90/Wave-U-Net)
 - Improved Wave-U-Net implemented in Pytorch. [Pytorch](https://github.com/f90/Wave-U-Net-Pytorch)
 - This repository implements the Wave-U-net architecture in TensorFlow 2 - satvik-venkatesh/Wave-U-net-TF2. [TF2 / Keras](https://github.com/satvik-venkatesh/Wave-U-net-TF2)
 - Improved speech enhancement with the Wave-U-Net, a deep convolutional neural network architecture for audio source separation, implemented for the task of speech enhancement in the time-domain. [For speech enhancements](https://github.com/craigmacartney/Wave-U-Net-For-Speech-Enhancement)

## Blind Source Separation

This section covers blind source separation tools and EM methods.

1. [Deep Audio Prior](https://github.com/adobe/Deep-Audio-Prior) — Our deep audio prior can enable several audio applications: blind sound source separation, interactive mask-based editing, audio textual synthesis, and audio watermarker removal.
2. BSS ([EM source separation](https://github.com/fgnt/pb_bss)) — This repository covers EM algorithms to separate speech sources in multi-channel recordings. In particular, the repository contains methods to integrate Deep Clustering (a neural network-based source separation algorithm) with a probabilistic spatial mixture model as proposed in the Interspeech paper "Tight integration of spatial and spectral features for BSS with Deep Clustering embeddings" presented at Interspeech 2017 in Stockholm.

## Image embeddings and others

This section lists embedding, pitch, speaker, and VAD tools, plus a MathWorks pretrained-model list.

The same notes are in [Embedding](../deep-learning/representations.md).

1. [Openl3](https://github.com/marl/openl3) — OpenL3: Open-source deep audio and image embeddings
- CREPE: A Convolutional REpresentation for Pitch Estimation -- pre-trained model (ICASSP 2018) - marl/crepe. [Pitch estimation](https://github.com/marl/crepe)
3. [Speaker recognition](https://github.com/Anwarvic/Speaker-Recognition) — Speaker recognition is the identification of a person given an audio file. It is used to answer the question "Who is speaking?" Speaker verification (also called speaker authentication) is similar to speaker recognition, but instead of returning the speaker who is speaking, it returns whether the speaker (who is claiming to be a certain one) is truthful or not. Speaker Verification is considered to be a little easier than speaker recognition.
- Silero VAD: pre-trained enterprise-grade Voice Activity Detector - snakers4/silero-vad. [Voice activity detector](https://github.com/snakers4/silero-vad)
- Taken from [here](https://www.mathworks.com/help/audio/referencelist.html?type=function&category=pretrained-models&s_tid=CRUX_topnav)

## Other Tools

This section collects toolkits and write-ups for separation, classification, speech recognition, and annotation.

The same notes are in [Deep Neural Audio](../deep-learning/deep-neural-audio.md) and [Speech](../generative-ai/speech.md).

1. [KALDI](https://kaldi-asr.org/models.html) speech recognition toolkit with many SOTA models.
- [Isolating instruments from stereo music using Convolutional Neural Networks](https://towardsdatascience.medium.com/audio-ai-isolating-vocals-from-stereo-music-using-convolutional-neural-networks-210532383785)
- [part 2](https://towardsdatascience.medium.com/audio-ai-isolating-instruments-from-stereo-music-using-convolutional-neural-networks-584ababf69de)
- [Sound classification using CNN, loading and normalizing sounds using librosa, converting to a 2d spectrogram image, using CNN on top.](https://medium.com/@mikesmales/sound-classification-using-deep-learning-8bc2aa1990b7)
4. [Speech recognition with DL](https://medium.com/@ageitgey/machine-learning-is-fun-part-6-how-to-do-speech-recognition-with-deep-learning-28293c162f7a) — how to convert sounds to vectors, feeding into an RNN.
- (Great) [Jonathan Hui on speech recognition](https://medium.com/@jonathan_hui/speech-recognition-series-71fd6784551a) Great series
6. [Gecko](https://medium.com/gong-tech-blog/introducing-gecko-an-open-source-solution-for-effective-annotation-of-conversations-2ecec0909941) — ([github.com/gong-io/gecko](https://github.com/gong-io/gecko)) [YouTube](https://www.youtube.com/watch?v=CBYA0YC1NBI), is an open-source tool for the annotation of the linguistic content of conversations. It can be used for segmentation, diarization, and transcription. With Gecko, you can create and perfect audio-based datasets, compare the results of multiple models simultaneously, and highlight differences between transcriptions.
