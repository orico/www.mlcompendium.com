# Algorithms

Sound Event Detection

1. [YamNet](https://github.com/tensorflow/models/tree/master/research/audioset/yamnet), and Real-time sound event detection [github](https://github.com/robertanto/Real-Time-Sound-Event-Detection), [Event types labels list](https://github.com/robertanto/Real-Time-Sound-Event-Detection/blob/main/keras\_yamnet/yamnet\_class\_map.csv) - Relevant labels: 420:430

Query-based separation

1. [Zero Shot Audio Source Separation](https://github.com/RetroCirce/Zero\_Shot\_Audio\_Source\_Separation), [paper](https://arxiv.org/abs/2112.07891), [interface](https://replicate.com/retrocirce/zero\_shot\_audio\_source\_separation) - is a three-component pipeline that allows you to train an audio source separator to separate any source from the track. All you need is a mixture audio to separate, and a given source sample as a query. Then the model will separate your specified source from the track.&#x20;

Audio Source Separation

1. [Audio Sep](https://github.com/Audio-AGI/AudioSep) - AudioSep is a foundation model for open-domain sound separation with natural language queries. AudioSep demonstrates strong separation performance and impressive zero-shot generalization ability on numerous tasks such as audio event separation, musical instrument separation, and speech enhancement"
2. Wave-U-net&#x20;
   1. [Original](https://github.com/f90/Wave-U-Net) Version 4y old
   2. [Pytorch](https://github.com/f90/Wave-U-Net-Pytorch) Version 3y old
   3. [TF2 / Keras](https://github.com/satvik-venkatesh/Wave-U-net-TF2) Version 2y old
   4. [For speech enhancements](https://github.com/craigmacartney/Wave-U-Net-For-Speech-Enhancement)

Blind Source Separation

1. [Deep Audio Prior](https://github.com/adobe/Deep-Audio-Prior) - Our deep audio prior can enable several audio applications: blind sound source separation, interactive mask-based editing, audio textual synthesis, and audio watermarker removal.
2. BSS ([EM source separation](https://github.com/fgnt/pb\_bss)) - This repository covers EM algorithms to separate speech sources in multi-channel recordings. In particular, the repository contains methods to integrate Deep Clustering (a neural network-based source separation algorithm) with a probabilistic spatial mixture model as proposed in the Interspeech paper "Tight integration of spatial and spectral features for BSS with Deep Clustering embeddings" presented at Interspeech 2017 in Stockholm.

Image embeddings and others

1. [Openl3](https://github.com/marl/openl3) - OpenL3: Open-source deep audio and image embeddings
2. [Pitch estimation](https://github.com/marl/crepe)
3. [Speaker recognition](https://github.com/Anwarvic/Speaker-Recognition) - Speaker recognition is the identification of a person given an audio file. It is used to answer the question "Who is speaking?" Speaker verification (also called speaker authentication) is similar to speaker recognition, but instead of returning the speaker who is speaking, it returns whether the speaker (who is claiming to be a certain one) is truthful or not. Speaker Verification is considered to be a little easier than speaker recognition.
4. [Voice activity detector](https://github.com/snakers4/silero-vad)&#x20;
5. Taken from [here](https://www.mathworks.com/help/audio/referencelist.html?type=function\&category=pretrained-models\&s\_tid=CRUX\_topnav)

Other Tools

1. [KALDI](https://kaldi-asr.org/models.html) speech recognition toolkit with many SOTA models.&#x20;
2. [isolating instruments from stereo music using Convolutional Neural Networks](https://towardsdatascience.com/audio-ai-isolating-vocals-from-stereo-music-using-convolutional-neural-networks-210532383785), [part 2](https://towardsdatascience.com/audio-ai-isolating-instruments-from-stereo-music-using-convolutional-neural-networks-584ababf69de)
3. [Sound classification using cnn, loading and normalizing sounds using librosa, converting to a 2d spectrogram image, using cnn on top.](https://medium.com/@mikesmales/sound-classification-using-deep-learning-8bc2aa1990b7)
4. [speech recognition with DL -](https://medium.com/@ageitgey/machine-learning-is-fun-part-6-how-to-do-speech-recognition-with-deep-learning-28293c162f7a) how to convert sounds to vectors, feeding into an RNN.
5. (Great) [Jonathan Hui on speech recognition](https://medium.com/@jonathan\_hui/speech-recognition-series-71fd6784551a) - series.
6. [Gecko](https://medium.com/gong-tech-blog/introducing-gecko-an-open-source-solution-for-effective-annotation-of-conversations-2ecec0909941) -  ([github.com/gong-io/gecko](https://github.com/gong-io/gecko)) [youtube](https://www.youtube.com/watch?v=CBYA0YC1NBI), is an open-source tool for the annotation of the linguistic content of conversations. It can be used for segmentation, diarization, and transcription. With Gecko, you can create and perfect audio-based datasets, compare the results of multiple models simultaneously, and highlight differences between transcriptions.
7.
