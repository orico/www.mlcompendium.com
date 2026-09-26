# Basics

Before engineering audio features or picking an algorithm, you need to know what sound is once it is digitized and why spectrograms sit at the center of audio deep learning. The page follows Ketan Doshi's series in order: sound and spectrograms, Mel spectrograms, data preparation, sound classification, speech recognition, and finally beam search.

Series by Ketan Doshi:

Part 1, [State-of-the-Art Techniques](https://towardsdatascience.medium.com/audio-deep-learning-made-simple-part-1-state-of-the-art-techniques-da1d3dff2504), is the grounding: what is sound and how it is digitized, what problems is audio deep learning solving in our daily lives, and what are spectrograms and why they are all-important. Once the spectrogram is the input, part 2, [Why Mel Spectrograms perform better](https://towardsdatascience.medium.com/audio-deep-learning-made-simple-part-2-why-mel-spectrograms-perform-better-aad889a93505), moves to processing audio data in Python: what are Mel Spectrograms and how to generate them.

The same notes are in [Feature Engineering](audio-feature-engineering.md).

With Mel spectrograms in hand, part 3, [Data Preparation and Augmentation](https://towardsdatascience.medium.com/audio-deep-learning-made-simple-part-3-data-preparation-and-augmentation-24c6e1f6b52), shows how to enhance spectrograms features for optimal performance by hyper-parameter tuning and data augmentation. Part 4, [Sound Classification](https://towardsdatascience.medium.com/audio-deep-learning-made-simple-sound-classification-step-by-step-cebc936bbe5), puts that data to work in an end-to-end example and architecture to classify ordinary sounds, a foundational application for a range of scenarios. Part 5, [Automatic Speech Recognition](https://towardsdatascience.medium.com/audio-deep-learning-made-simple-automatic-speech-recognition-asr-how-it-works-716cfce4c706), turns from sounds to words: voice assistants such as Google Home, Amazon Echo, Siri, and Cortana start from a clip of spoken audio and extract the spoken words as text, and the article walks the Speech-to-Text algorithm and architecture, using CTC Loss and Decoding for aligning sequences.

The same notes are in [Speech](../generative-ai/speech.md).

Decoding is where the series ends. Part 6, [Beam Search](https://towardsdatascience.medium.com/foundations-of-nlp-explained-visually-beam-search-how-it-works-1586b9849a24), is the algorithm commonly used by Speech-to-Text and NLP applications to enhance predictions.

The same notes are in [Decoding Algorithms For NLP](../language-ai/decoding-algorithms-for-nlp.md).
