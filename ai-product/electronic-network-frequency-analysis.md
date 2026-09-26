# Electronic Network Frequency Analysis

Audio can be checked against the power grid, because every recording near mains power carries its hum. The page first defines electrical network frequency (ENF) analysis and then points to a Python tool that extracts it.

The same notes are in [Digital Signal Processing (DSP)](../predictive-ml/digital-signal-processing-dsp.md) and [Fourier Transform](../predictive-ml/fourier-transform.md).

Electrical network frequency (ENF) analysis is an [audio forensics](https://en.wikipedia.org/wiki/Audio_forensics) technique for validating [audio recordings](https://en.wikipedia.org/wiki/Audio_recording) by comparing frequency changes in background [mains hum](https://en.wikipedia.org/wiki/Mains_hum) in the recording with long-term high-precision historical records of [mains frequency](https://en.wikipedia.org/wiki/Mains_frequency) changes from a database. Each of those four terms links to its Wikipedia article; mains frequency is covered there under utility frequency, and audio recording under sound recording and reproduction.

To try it on real files, the first step is extracting the ENF signal from the recording. [pyenf](https://github.com/deerajnagothu/pyenf_extraction) — Python-based ENF extraction from video and audio recordings.
