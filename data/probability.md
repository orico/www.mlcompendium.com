# Probability

Before a distribution or a density estimate can be read, the basic probability pictures have to be clear, and so does the PDF those pictures become. The page starts with crib figures, moves to the probability density function, and ends with kernel density estimation and the videos that teach it.

The same notes are in [Distribution](distribution.md) and [Probability & Statistics](probability-and-statistics.md).

## Probability crib figures

The first step is visual. These four crib sheets from [en.wikipedia.org](https://en.wikipedia.org/) put the basic probability ideas on one screen before any density function appears.

<figure><img src="../.gitbook/assets/gimg-c0758b95a577.png" alt=""><figcaption><p>Probability crib figure.</p><p>Credit: by <a href="https://en.wikipedia.org/">en.wikipedia.org</a>.</p></figcaption></figure>

<figure><img src="../.gitbook/assets/gimg-548a6386a5ac.png" alt=""><figcaption><p>Probability crib figure.</p><p>Credit: by <a href="https://en.wikipedia.org/">en.wikipedia.org</a>.</p></figcaption></figure>

<figure><img src="../.gitbook/assets/gimg-17d46a314d25.png" alt=""><figcaption><p>Probability crib figure.</p><p>Credit: by <a href="https://en.wikipedia.org/">en.wikipedia.org</a>.</p></figcaption></figure>

<figure><img src="../.gitbook/assets/gimg-e31bb8b7775a.png" alt=""><figcaption><p>Probability crib figure.</p><p>Credit: by <a href="https://en.wikipedia.org/">en.wikipedia.org</a>.</p></figcaption></figure>

## PDF (probability density function)

Once the pictures are familiar, the next question is how to get a PDF out of real sample data in code. The [Tutorial in scipy](https://oneau.wordpress.com/2011/02/28/simple-statistics-with-scipy/) is the "Simple statistics with SciPy" post on Comfort at 1 AU, the starting point for doing this in SciPy. The [Array-based tutorial in python with PDF and KDE](http://firsttimeprogrammer.blogspot.co.il/2015/01/how-to-estimate-probability-density.html) shows how to estimate a probability density function from sample data with Python, on a blog of statistics and machine-learning experiments in Python, R, and Java. The [Summary of univariate distribution including pdf methods](https://www.johndcook.com/blog/distributions_scipy/) is a set of notes on probability distribution functions in Python using SciPy, useful as a reference once the first estimate works.

## Kernel Density Estimation

A PDF from a histogram breaks down when the histogram is sparse, and that is where KDE comes in. A tutorial that used to sit here actually explains why we should use KDE over a Histogram: it explains the cons of histograms and how KDE helps solve some issue that we usually encounter in ‘Sparse’ histograms where the distribution is hard to figure out. That address no longer opens and is kept at the end of the page.

For the code itself, there is supposedly a better [implementation](https://github.com/Daniel-B-Smith/KDE-for-SciPy), a repository of code meant to be integrated with SciPy's KDE module. Supposedly better KDE than SciPy is the author's note on it, not a benchmark.

How to use KDE? A [tutorial](http://pythonhosted.org/PyQt-Fit/KDE_tut.html) about kernel density and how to use it in python. Has several good graphs and shows use cases.

### Video tutorials about Kernel Density

Reading about KDE is slower than watching the bandwidth change, so the next step is video. Anders Munk-Nielsen's series starts with [KDE](https://www.youtube.com/watch?v=gPWsDh59zdo) itself, the Kernel Density Estimation lecture. The same series then widens to Non parametric [Kernel Regression Estimation](https://www.youtube.com/watch?v=ncF7ArjJFqM), then Non parametric [Sieve Estimation](https://www.youtube.com/watch?v=cqecz-DL-jI), and ends with [Semi- nonparametric estimation](https://www.youtube.com/watch?v=G1N53K530To), Robinson's double residual method.

The [Udacity Video Tutorial](https://www.youtube.com/watch?v=MEP35FcrQGs&list=PLAwxTw4SYaPn-ttWkPiUL7NP3lLRdUniJ&index=80) - pretty good - is Udacity's "Inspecting Distributions" lesson from Model Building and Validation.

After the videos, the written references compare and apply the methods. Jake VanderPlas's Kernel Density Estimation in Python post on Pythonic Perambulations is marked IMPORTANT: [Comparison and benchmarks of various KDE algo’s](https://jakevdp.github.io/blog/2013/12/01/kernel-density-estimation/). Will Koehrsen's [Histograms and density plots](https://medium.com/data-science/histograms-and-density-plots-in-python-f6bda88f5ac0) starts from the simple histogram, which shows the location, spread, and shape of the data, and moves to density plots for the cases where a histogram fails. The [SK LEARN](http://scikit-learn.org/stable/modules/density.html#kernel-density-estimation) density-estimation guide frames the topic as walking the line between unsupervised learning, feature engineering, and data modeling, with mixture models among the most popular techniques. The same Udacity lesson on inspecting distributions is linked once more as [Gaussian KDE in scipy, version 2](https://www.youtube.com/watch?v=MEP35FcrQGs&list=PLAwxTw4SYaPn-ttWkPiUL7NP3lLRdUniJ&index=80).

## Deprecated links

{% hint style="warning" %}
These links and images no longer work. The original wording is kept here. A same-resource copy, when one was checked, is used above.
{% endhint %}

- This tutorial. This address no longer opens: https://mglerner.github.io/posts/histograms-and-kernel-density-estimation-kde-2.html?p=28
