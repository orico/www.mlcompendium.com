# Other Architectures

Not every useful network is a standard deep net trained end to end with backpropagation. This page walks through four other families, each on its own: self-organizing maps that cluster by competition, neuro-evolution that grows networks with genetic algorithms, radial basis function networks that classify by distance to prototypes, and signal-processing networks built on Fourier transforms and wavelets.

## SELF ORGANIZING MAPS (SOM)

The first family is the self-organizing map, a clustering network also known as the Kohonen map, and the fastest way into it is code.

On Git, [Sompy](https://github.com/sevamoo/SOMPY) is a Python library for Self Organizing Map (SOM), and [minisom!](https://github.com/JustGlowing/minisom) is MiniSom, a minimalistic implementation of the Self Organizing Maps. [Many graph examples](https://medium.com/@s.ganjoo96/self-organizing-maps-b2cf58b74fdb) is Shivang Ganjoo's post on self-organizing maps as a super easy and very intuitive clustering technique, and the lightsalsa251/Self-Organizing-Map repo is one more [example](https://github.com/lightsalsa251/Self-Organizing-Map).

Once the code runs, the explanations follow. The author's notes list a step by step with examples, calculations (kept at the end of the page), one that adds intuition regarding “magnetism”’, and [Implementation and faces](https://medium.com/@navdeepsingh_2336/self-organizing-maps-for-machine-learning-algorithms-ad256a395fc5), which gives intuition towards each node and what it represents in a vision. I.e., each face resembles one of K clusters. They also point to Medium on kohonen networks, i.e., SOM, and to a SOM on iris that explains inference - averaging, and cons of the method; those sources are kept at the end of the page. [Simple explanation](https://medium.com/@valentinerutto/selforganizingmaps-in-english-35574f95b0ac) is Valentine's "Self Organizing Maps In English", and the last note in the run was the algorithm, formulas.

## NEURO EVOLUTION (GA/GP based)

A SOM still has a fixed grid. Neuro-evolution goes further and lets a genetic algorithm decide the network itself.

The same notes are in [Genetic Algorithms & Genetic Programming](../predictive-ml/genetic-algorithms-and-genetic-programming.md).

NEAT

[NEAT](http://www.cs.ucf.edu/~kstanley/neat.html) stands for NeuroEvolution of Augmenting Topologies. It is a method for evolving artificial neural networks with a genetic algorithm.

NEAT implements the idea that it is most effective to start evolution with small, simple networks and allow them to become increasingly complex over generations.

That way, just as organisms in nature increased in complexity since the first cell, so do neural networks in NEAT.

This process of continual elaboration allows finding highly sophisticated and complex neural networks. For a longer read on the same method, see [A great article about NEAT](http://hunterheidenreich.com/blog/neuroevolution-of-augmenting-topologies/).

HYPER-NEAT

NEAT evolves connections one by one. HyperNEAT computes the connectivity of its neural networks as a function of their geometry.

HyperNEAT is based on a theory of representation that hypothesizes that a good representation for an artificial neural network should be able to describe its pattern of connectivity compactly.

The encoding in HyperNEAT, called [compositional pattern producing networks](http://en.wikipedia.org/wiki/Compositional_pattern-producing_network), is designed to represent patterns with regularities such as symmetry, repetition, and repetition with variationץ

(WIKI) [Compositional pattern-producing networks](https://en.wikipedia.org/wiki/Compositional_pattern-producing_network) (CPPNs) are a variation of artificial neural networks (ANNs) that have an architecture whose evolution is guided by genetic algorithms. The figure below shows that pattern-producing idea.

<figure><img src="../.gitbook/assets/gimg-aa3cb1f189bb.png" alt=""><figcaption><p>NEURO EVOLUTION (GA/GP based)</p><p>Credit: <a href="https://lh6.googleusercontent.com/cAbcsLDWcDOMlX4K53ROOLyiAw6EhJ9ZRDuZmURFtBaje8JtwzU_KsOh4aeiC8ukdYgBYEm6zqWd7jZ3tStib3JJGYrmxM4wlrgyBJFhlnMHd_kIcxgO2reEsoE4RPjJLXr3O-R_">copied from the original hosted image</a>.</p></figcaption></figure>

A great HyperNeat tutorial on Medium used to sit here; it is kept at the end of the page with the HyperNEAT project page.

## Radial Basis Function Network (RBFN)

Evolution changes the shape of a network. An RBFN keeps a simple shape and changes what a neuron means: each one is a stored example.

For code, [RBF layer in Keras.](https://github.com/PetraVidnerova/rbf_keras/blob/master/test.py) is the test script of PetraVidnerova's rbf_keras, an RBF layer for Keras.

The [RBFN](http://mccormickml.com/2013/08/15/radial-basis-function-network-rbfn-tutorial/) approach is more intuitive than the MLP. An RBFN performs classification by measuring the input’s similarity to examples from the training set. Each RBFN neuron stores a “prototype”, which is just one of the examples from the training set. When we want to classify a new input, each neuron computes the Euclidean distance between the input and its prototype. Roughly speaking, if the input more closely resembles the class A prototypes than the class B prototypes, it is classified as class A. The simple architecture below shows that layout.

<figure><img src="../.gitbook/assets/gimg-6fdc79788c53.png" alt=""><figcaption><p>Architecture\_Simple</p><p>Credit: <a href="https://lh6.googleusercontent.com/5oVVPw02w2Pv1kqAGvQ6drOX6Nh7lA72cBDplTbqgd78u25ceNdjufDe8h4pKWNPKC350_r4V_TPUn1ionjck1IPJiW0Q4rwivL4sH4LJGaj7V7WZBss8eLSuqpZb5Rv525M4sQ1">copied from the original hosted image</a>.</p></figcaption></figure>

## SIGNAL PROCESSING NN (FFT, WAVELETS, SHAPELETS)

RBFNs compare raw inputs. Signal-processing networks first change the representation of the signal, into frequencies or wavelets, and learn on that.

The same notes are in [Feature Engineering](../predictive-ml/audio-feature-engineering.md).

The base is the [Fourier Transform](https://www.youtube.com/watch?v=spUNpyF58BY) - decomposing frequencies, here as 3Blue1Brown's visual introduction. Wavelets add time to frequency: [WAVELETS On youtube (4 videos)](https://www.youtube.com/watch?v=QX1-xGVFqmw) is MATLAB's "Understanding Wavelets" series. Wavelets are [used for denoising](https://www.youtube.com/watch?v=veCvP1mYpww), compression, detect edges, detect features with various orientation, analyse signal power, detect and localize transients, change points in time series data and detect optimal signal representation (peaks etc) of time freq analysis of images and data. They can also be used to [reconstruct time and frequencies](https://www.youtube.com/watch?v=veCvP1mYpww), analyse images in space, frequencies, orientation, identifying coherent time oscillation in time series, and to analyse signal variability and correlation.

## Deprecated links

{% hint style="warning" %}
These links and images no longer work. The original wording is kept here. A same-resource copy, when one was checked, is used above.
{% endhint %}
- Towards Data Science: hyperneat-powerful-indirect-neural-network-evolution-fba5c7c43b7b. This address no longer opens: https://towardsdatascience.com/hyperneat-powerful-indirect-neural-network-evolution-fba5c7c43b7b
- Towards Data Science: kohonen-self-organizing-maps-a29040d688da. This address no longer opens: https://towardsdatascience.com/kohonen-self-organizing-maps-a29040d688da
- Towards Data Science: self-organizing-maps-1b7d2a84e065. This address no longer opens: https://towardsdatascience.com/self-organizing-maps-1b7d2a84e065
- Towards Data Science: self-organizing-maps-ff5853a118d4. This address no longer opens: https://towardsdatascience.com/self-organizing-maps-ff5853a118d4
- Step by step with examples, calculations. This address no longer opens: https://mc.ai/self-organizing-mapsom/
- HyperNEAT. This address no longer opens: http://eplex.cs.ucf.edu/hyperNEATpage/
