# Probability

This page collects notes on probability density functions and kernel density estimation, with Wikipedia crib figures.

The same notes are in [Distribution](distribution.md) and [Probability & Statistics](probability-and-statistics.md).

## Probability crib figures

These figures are crib sheets from Wikipedia on basic probability ideas.

<figure><img src="../.gitbook/assets/gimg-c0758b95a577.png" alt=""><figcaption><p>Probability crib figure.</p><p>Credit: by <a href="https://en.wikipedia.org/">en.wikipedia.org</a>.</p></figcaption></figure>

<figure><img src="../.gitbook/assets/gimg-548a6386a5ac.png" alt=""><figcaption><p>Probability crib figure.</p><p>Credit: by <a href="https://en.wikipedia.org/">en.wikipedia.org</a>.</p></figcaption></figure>

<figure><img src="../.gitbook/assets/gimg-17d46a314d25.png" alt=""><figcaption><p>Probability crib figure.</p><p>Credit: by <a href="https://en.wikipedia.org/">en.wikipedia.org</a>.</p></figcaption></figure>

<figure><img src="../.gitbook/assets/gimg-e31bb8b7775a.png" alt=""><figcaption><p>Probability crib figure.</p><p>Credit: by <a href="https://en.wikipedia.org/">en.wikipedia.org</a>.</p></figcaption></figure>

## PDF (probability density function)

This section points at tutorials on PDFs in SciPy and related univariate methods.

1. [Tutorial in scipy](https://oneau.wordpress.com/2011/02/28/simple-statistics-with-scipy/)
2. [Array-based tutorial in python with PDF and KDE](http://firsttimeprogrammer.blogspot.co.il/2015/01/how-to-estimate-probability-density.html)
3. [Summary of univariate distribution including pdf methods](https://www.johndcook.com/blog/distributions_scipy/)

## Kernel Density Estimation

This section explains why KDE helps when histograms are sparse, and lists tools and videos.

This tutorial actually explains why we should use KDE over a Histogram, it explains the cons of histograms and how KDE helps solve some issue that we usually encounter in ‘Sparse’ histograms where the distribution is hard to figure out.

- Supposedly a better [implementation](https://github.com/Daniel-B-Smith/KDE-for-SciPy) of KDE than SciPy

How to use KDE? A [tutorial](http://pythonhosted.org/PyQt-Fit/KDE_tut.html) about kernel density and how to use it in python. Has several good graphs and shows use cases.

### Video tutorials about Kernel Density

These YouTube links are video tutorials on KDE and related nonparametric estimation.

1. [KDE](https://www.youtube.com/watch?v=gPWsDh59zdo)
2. Non parametric [Kernel Regression Estimation](https://www.youtube.com/watch?v=ncF7ArjJFqM)
3. Non parametric [Sieve Estimation](https://www.youtube.com/watch?v=cqecz-DL-jI)
4. [Semi- nonparametric estimation](https://www.youtube.com/watch?v=G1N53K530To)

[Udacity Video Tutorial](https://www.youtube.com/watch?v=MEP35FcrQGs&list=PLAwxTw4SYaPn-ttWkPiUL7NP3lLRdUniJ&index=80) - pretty good

1. IMPORTANT: [Comparison and benchmarks of various KDE algo’s](https://jakevdp.github.io/blog/2013/12/01/kernel-density-estimation/)
2. [Histograms and density plots](https://medium.com/data-science/histograms-and-density-plots-in-python-f6bda88f5ac0)
3. [SK LEARN](http://scikit-learn.org/stable/modules/density.html#kernel-density-estimation)
4. [Gaussian KDE in scipy, version 2](https://www.youtube.com/watch?v=MEP35FcrQGs&list=PLAwxTw4SYaPn-ttWkPiUL7NP3lLRdUniJ&index=80)

## Deprecated links

{% hint style="warning" %}
These links and images no longer work. The original wording is kept here. A same-resource copy, when one was checked, is used above.
{% endhint %}

- This tutorial. This address no longer opens: https://mglerner.github.io/posts/histograms-and-kernel-density-estimation-kde-2.html?p=28
