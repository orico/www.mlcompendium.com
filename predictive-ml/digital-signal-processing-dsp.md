# Digital Signal Processing (DSP)

Once a signal has been decomposed into frequencies, the practical question is which helpers actually filter it, find its peaks, and detect beats. This page is a toolbox shelf: SciPy for time-domain signal processing and peak finding first, then beat detection with Librosa and a real-time detector, then two modeling notes on activity recognition and self-attention for sound.

The same notes are in [Electronic Network Frequency Analysis](../ai-product/electronic-network-frequency-analysis.md) and [Feature Engineering](audio-feature-engineering.md).

The base layer is SciPy. [Scipy signal processing](https://docs.scipy.org/doc/scipy/reference/signal.html) is the signal processing module (scipy.signal) in the SciPy v1.18.0 manual, and [Script find peaks](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html) is its find_peaks reference, the function to reach for when the question is where a signal spikes.

Peaks in audio lead directly to beat detection. [Real time bpm beat det](https://github.com/shunfu/python-beat-detector) is a Python beat detector that does real-time detection of beats for audio, calculates BPM, and flashes an LED strip in time with music. Librosa also covers time-domain processing and beat detection (and temp), but its original docs addresses no longer open and are kept at the end of the page.

The same time-series thinking carries over to sensors. [Mastery on Human activity recognition, smartphones](https://machinelearningmastery.com/cnn-models-for-human-activity-recognition-time-series-classification/) is MachineLearningMastery.com on 1D convolutional neural network models for human activity recognition from smartphone data.

(Out of place) — Christopher Dossman's [using self-attention for sound signal processing](https://medium.com/ai%C2%B3-theory-practice-business/toward-interpretable-music-tagging-with-self-attention-67a8136048d0) is Toward Interpretable Music Tagging with Self-Attention. Out place on a DSP shelf, it is a model rather than a signal helper.

## Deprecated links

{% hint style="warning" %}
These links and images no longer work. The original wording is kept here. A same-resource copy, when one was checked, is used above.
{% endhint %}

- Librosa. This address no longer opens: https://librosa.org/doc/latest/core.html#time-domain-processing
- Beat detection (and temp). This address no longer opens: https://librosa.org/doc/latest/beat.html#beat
