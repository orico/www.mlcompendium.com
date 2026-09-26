# Algorithms

Once the task names and the features are clear, the remaining question is which model or package actually does the job. The page goes from sound event detection to query-based and open-domain source separation, then blind source separation, then embeddings and other pretrained tools, and ends with toolkits and write-ups for classification, speech recognition, and annotation.


## Sound Event Detection

Detection comes first because it is the most direct use of a pretrained audio model: say which sound events occur and when.

The same notes are in [Terminology](audio-terminology.md).

[YamNet](https://github.com/tensorflow/models/tree/master/research/audioset/yamnet) sits in the TensorFlow models repository of models and examples built with TensorFlow, and it is the pretrained model the rest of this section builds on. The robertanto repository on [GitHub](https://github.com/robertanto/Real-Time-Sound-Event-Detection) is the Python implementation of a sound event detection system working in real time, built on YamNet, and real-time sound event detection is exactly what it is for. Its [Event types labels list](https://github.com/robertanto/Real-Time-Sound-Event-Detection/blob/main/keras_yamnet/yamnet_class_map.csv) is the YamNet class map used by that real time sound event detection code; the relevant labels are 420 430.

## Query-based separation

Detection says a sound is there; query-based separation pulls that one sound out of the mix when you give an example of it.

[Zero Shot Audio Source Separation](https://github.com/RetroCirce/Zero_Shot_Audio_Source_Separation) is the official code repo for the AAAI 2022 work, the [paper](https://arxiv.org/abs/2112.07891) is "Zero-shot Audio Source Separation through Query-based Learning from Weakly-labeled Data" on arXiv, and the [interface](https://replicate.com/retrocirce/zero_shot_audio_source_separation) on Replicate runs zero shot sound separation by arbitrary query samples — a three-component pipeline that lets you train an audio source separator to separate any source from the track. All you need is a mixture audio to separate, and a given source sample as a query. Then the model will separate your specified source from the track.

## Audio Source Separation

Querying by a sample is one way to pick a source; the models here separate open-domain sources, including by natural language, or learn separation directly on the waveform.

The same notes are in [Terminology](audio-terminology.md).

[Audio Sep](https://github.com/Audio-AGI/AudioSep) — AudioSep is a foundation model for open-domain sound separation with natural language queries. AudioSep demonstrates strong separation performance and impressive zero-shot generalization ability on numerous tasks such as audio event separation, musical instrument separation, and speech enhancement.

The waveform route is Wave-U-Net, and it comes in several implementations. The [Original](https://github.com/f90/Wave-U-Net) is f90/Wave-U-Net, the implementation of the Wave-U-Net for audio source separation. The [Pytorch](https://github.com/f90/Wave-U-Net-Pytorch) version is the improved Wave-U-Net implemented in Pytorch. [TF2 / Keras](https://github.com/satvik-venkatesh/Wave-U-net-TF2) is satvik-venkatesh/Wave-U-net-TF2, which implements the Wave-U-net architecture in TensorFlow 2. [For speech enhancements](https://github.com/craigmacartney/Wave-U-Net-For-Speech-Enhancement) is improved speech enhancement with the Wave-U-Net, a deep convolutional neural network architecture for audio source separation, implemented for the task of speech enhancement in the time-domain.

## Blind Source Separation

The models above learn what sources sound like from labeled data; blind source separation works without that, from priors or from the spatial structure of multi-channel recordings.

[Deep Audio Prior](https://github.com/adobe/Deep-Audio-Prior) — Our deep audio prior can enable several audio applications: blind sound source separation, interactive mask-based editing, audio textual synthesis, and audio watermarker removal.

BSS ([EM source separation](https://github.com/fgnt/pb_bss)) — This repository covers EM algorithms to separate speech sources in multi-channel recordings. In particular, the repository contains methods to integrate Deep Clustering (a neural network-based source separation algorithm) with a probabilistic spatial mixture model as proposed in the Interspeech paper "Tight integration of spatial and spectral features for BSS with Deep Clustering embeddings" presented at Interspeech 2017 in Stockholm.

## Image embeddings and others

Separation is not the only downstream task; many audio pipelines need a pretrained representation or a single-purpose detector instead.

The same notes are in [Embedding](../deep-learning/representations.md).

[Openl3](https://github.com/marl/openl3) — OpenL3: Open-source deep audio and image embeddings. For pitch, [Pitch estimation](https://github.com/marl/crepe) is marl/crepe, CREPE: A Convolutional REpresentation for Pitch Estimation, a pre-trained model (ICASSP 2018).

[Speaker recognition](https://github.com/Anwarvic/Speaker-Recognition) — Speaker recognition is the identification of a person given an audio file. It is used to answer the question "Who is speaking?" Speaker verification (also called speaker authentication) is similar to speaker recognition, but instead of returning the speaker who is speaking, it returns whether the speaker (who is claiming to be a certain one) is truthful or not. Speaker Verification is considered to be a little easier than speaker recognition.

Before any of these runs, a pipeline often needs to know whether anyone is speaking at all. [Voice activity detector](https://github.com/snakers4/silero-vad) is snakers4/silero-vad, Silero VAD, a pre-trained enterprise-grade Voice Activity Detector. The list of pretrained audio models these tools were drawn from is taken from [here](https://www.mathworks.com/help/audio/referencelist.html?type=function&category=pretrained-models&s_tid=CRUX_topnav), the MathWorks pretrained-model list.

## Other Tools

Beyond single models, the remaining shelf is full toolkits and write-ups for separation, classification, speech recognition, and annotation.

The same notes are in [Deep Neural Audio](../deep-learning/deep-neural-audio.md) and [Speech](../generative-ai/speech.md).

[KALDI](https://kaldi-asr.org/models.html) speech recognition toolkit with many SOTA models is the toolkit to start from.

Separation also has an end-to-end story in the Audio AI series by Alejandro Koretzky. [Isolating instruments from stereo music using Convolutional Neural Networks](https://towardsdatascience.medium.com/audio-ai-isolating-vocals-from-stereo-music-using-convolutional-neural-networks-210532383785) is the first article, on isolating vocals from stereo music: a medium quality mp3 of We Can Work it Out by The Beatles goes in, and the isolated vocals come out of the model. [part 2](https://towardsdatascience.medium.com/audio-ai-isolating-instruments-from-stereo-music-using-convolutional-neural-networks-584ababf69de) is the follow-up on isolating instruments, which recommends reading the vocal isolation article first.

For classification, Mike Smales' Udacity capstone on classifying urban sounds walks the full path: [Sound classification using CNN, loading and normalizing sounds using librosa, converting to a 2d spectrogram image, using CNN on top.](https://medium.com/@mikesmales/sound-classification-using-deep-learning-8bc2aa1990b7)

For speech, [Speech recognition with DL](https://medium.com/@ageitgey/machine-learning-is-fun-part-6-how-to-do-speech-recognition-with-deep-learning-28293c162f7a) — how to convert sounds to vectors, feeding into an RNN. (Great) [Jonathan Hui on speech recognition](https://medium.com/@jonathan_hui/speech-recognition-series-71fd6784551a) is a Great series that covers the basics, like phonetics, and the machine learning models used in speech recognition, and then applies deep learning to it.

Speech models also need labeled conversations. [Gecko](https://medium.com/gong-tech-blog/introducing-gecko-an-open-source-solution-for-effective-annotation-of-conversations-2ecec0909941) — ([github.com/gong-io/gecko](https://github.com/gong-io/gecko)) [YouTube](https://www.youtube.com/watch?v=CBYA0YC1NBI), is an open-source tool for the annotation of the linguistic content of conversations, introduced by Golan Levy on the Gong tech blog, where automatic speech recognition is at the core of the product. It can be used for segmentation, diarization, and transcription. With Gecko, you can create and perfect audio-based datasets, compare the results of multiple models simultaneously, and highlight differences between transcriptions.
