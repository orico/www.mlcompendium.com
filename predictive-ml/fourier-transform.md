# Fourier Transform

A signal recorded over time often hides its structure in frequency, so before modeling it you need a way to see which waves it is made of. The page starts with the Fourier transform, which decomposes a signal into sine and cosine waves, and then moves to wavelets, which keep track of when those frequencies happen.

The same notes are in [Electronic Network Frequency Analysis](../ai-product/electronic-network-frequency-analysis.md) and [Feature Engineering](audio-feature-engineering.md).

## Fourier Transform

The Fourier transform is a technique to decompose signals into sine and cosine waves. The applied entry point is Khairul Omar's [Deconstructing time series using FT](https://medium.com/@khairulomar/deconstructing-time-series-using-fourier-transform-e52dd535a44e), which starts from time series as data captured at equally spaced periods, the kind of data that weather measurements, stock markets, mobile transmission, and now Internet of Things devices generate continuously.

The math behind it is a Medium series, mostly on the math, in five parts. Part [1](https://medium.com/sho-jp/fourier-transform-101-part-1-b69ea3cb4837) opens the series. Part [2](https://medium.com/sho-jp/fourier-transform-101-part-2-complex-fourier-series-934a885b3921) recaps the real Fourier series from Part 1, where a periodic signal f(t) is approximated by P(t), and moves to the complex Fourier series. Part [3](https://medium.com/sho-jp/fourier-transform-101-part-3-fourier-transform-6def0bd2ca9b) relates the complex Fourier series to the continuous Fourier transform before stepping into what that transform means. Part [4](https://medium.com/sho-jp/fourier-transform-101-part-4-discrete-fourier-transform-8fc3fbb763f3) is the discrete Fourier transform, and part [5](https://medium.com/sho-jp/fourier-transform-101-part-5-fast-fourier-transform-fft-38c22e05ead3) is the fast Fourier transform (FFT).

## Wavelets

A Fourier view says which frequencies are present but not when, and wavelets are the next step for data science on signals. Two Medium pieces used to open this reading, one on what a wavelet is and how we use it for DS and one on multiple time series classification using continuous wavelet transformation and scalograms; both addresses no longer open and are kept at the end of the page.

The explanation that remains is Shaw Talebi's [Medium on the wavelet transform](https://medium.com/data-science/the-wavelet-transform-e9cfa85d7b34), The Wavelet Transform. The same Medium on the wavelet transform article also sits at its original address, [https://towardsdatascience.com/the-wavelet-transform-e9cfa85d7b34](https://towardsdatascience.com/the-wavelet-transform-e9cfa85d7b34).

To run wavelets in code, [pywavelets](https://pywavelets.readthedocs.io/en/latest/) — PyWavelets — is open source wavelet transform software for [Python](http://python.org/), the official home of the Python programming language. It combines a simple high level interface with low level C and Cython performance.

## Deprecated links

{% hint style="warning" %}
These links and images no longer work. The original wording is kept here. A same-resource copy, when one was checked, is used above.
{% endhint %}

- Medium on What is a wavelet and how do we use it for DS. This address no longer opens: https://towardsdatascience.com/what-is-wavelet-and-how-we-use-it-for-data-science-d19427699cef
- Medium multiple time series classification using continuous wavelet transformation and scalograms. This address no longer opens: https://towardsdatascience.com/multiple-time-series-classification-by-using-continuous-wavelet-transformation-d29df97c0442
