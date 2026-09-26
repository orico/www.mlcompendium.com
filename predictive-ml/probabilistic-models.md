### PROBABILISTIC ALGORITHMS

This section lists naive Bayes, Bayesian networks, Markov models, HMMs, IOHMM, and CRF resources.

#### NAIVE BAYES

This subsection links introductory naive Bayes material.

1. Vidhya on NB
2. [Baysian tree](https://github.com/UBS-IB/bayesian_tree)
3. [NB, GNB, multi nominal NB](https://jakevdp.github.io/PythonDataScienceHandbook/05.05-naive-bayes.html)

#### BAYES, BAYESIAN BELIEF NETWORKS

This subsection covers Bayes theorem, belief networks, and maximum likelihood.

1. [Mastery on bayes theorem](https://machinelearningmastery.com/bayes-theorem-for-machine-learning/)
2. [Introduction To BBS](https://codesachin.wordpress.com/2017/03/10/an-introduction-to-bayesian-belief-networks/) - a very good blog post
3. A complementing SLIDE presentation that shows how to build the network’s tables
4. A very nice presentation regarding BBS
5. [Maximum Likelihood](http://mathworld.wolfram.com/MaximumLikelihood.html) (log likelihood) - proofs for bernoulli, normal, poisson.
6. [Another example](https://codesachin.wordpress.com/2017/03/10/an-introduction-to-bayesian-belief-networks/)

#### MARKOV MODELS

This subsection explains random versus stochastic wording and introductory Markov chains.

Random vs Stochastic ([here](https://math.stackexchange.com/questions/114373/whats-the-difference-between-stochastic-and-random) and [here](https://math.stackexchange.com/questions/569951/what-is-the-difference-between-a-random-vector-and-a-stochastic-process)):

- A variable is 'random'.
- A process is 'stochastic'.

Apart from this difference the two words are synonyms

In other words:

- A random vector is a generalization of a single random variables to many.
- A stochastic process is a sequence of random variables, or a sequence of random vectors (and then you have a vector-stochastic process).

(What is a Markov Model?) A Markov Model is a stochastic(random) model which models temporal or sequential data, i.e., data that are ordered.

- It provides a way to model the dependencies of current information (e.g. weather) with previous information.
- It is composed of states, transition scheme between states, and emission of outputs (discrete or continuous).
- Several goals can be accomplished by using Markov models:
   - Learn statistics of sequential data.
   - Do prediction or estimation.
   - Recognize patterns.

([sunny cloudy explanation](http://techeffigytutorials.blogspot.co.il/2015/01/markov-chains-explained.html)) Markov Chains is a probabilistic process, that relies on the current state to predict the next state.

- to be effective the current state has to be dependent on the previous state in some way
- if it looks cloudy outside, the next state we expect is rain.
- If the rain starts to subside into cloudiness, the next state will most likely be sunny.
- Not every process has the Markov Property, such as the Lottery, this weeks winning numbers have no dependence to the previous weeks winning numbers.

1. They show how to build an order 1 markov table of probabilities, predicting the next state given the current.
2. Then it shows the state diagram built from this table.
3. Then how to build a transition matrix from the 3 states, i.e., from the probabilities in the table
4. Then how to calculate the next state using the “current state vector” doing vec\*matrix multiplications.
5. Then it talks about the setting always into the rain prediction, and the solution is using two last states in a bigger table of order 2. He is not really telling us why the probabilities don't change if we add more states, it stays the same as in order 1, just repeating.

#### MARKOV MODELS / HIDDEN MARKOV MODEL

This subsection collects HMM tutorials, software, and explanatory videos.

The same notes are in [Timeseries](forecasting.md).

HMM tutorials

1. HMM tutorial
   1. Part 1, 2, 3, 4
2. Medium
   1. Intro to HMM / MM
   2. [Paper like example](https://medium.com/@kangeugine/hidden-markov-model-7681c22f5b9)
3. HMM with sklearn and networkx

HMM variants

1. [Stack exchange on hmm](https://datascience.stackexchange.com/questions/8460/python-library-to-implement-hidden-markov-models)
2. [HMM LEARN](https://github.com/hmmlearn/hmmlearn) (sklearn, still being developed)
3. [Pomegranate](https://pomegranate.readthedocs.io/en/latest/) (this is good)
   1. General mixture models
   2. Hmm
   3. Basyes classifiers and naive bayes
   4. Markov changes
   5. Bayesian networks
   6. Markov networks
   7. Factor graphs
4. [GHMM with python wrappers](http://ghmm.org/),
5. [Hmms](https://github.com/lopatovsky/HMMs) (old)

HMM ([what is? And why HIDDEN?)](https://youtu.be/jY2E6ExLxaw?t=27m38s) - the idea is that there are things that you CAN OBSERVE and there are things that you CAN'T OBSERVE. From the things you OBSERVE you want to INFER the things you CAN'T OBSERVE (HIDDEN). I.e., you play against someone else in a game, you don't see their choice of action, but you see the result.

1. Python [code](https://github.com/hmmlearn/hmmlearn), previously part of [sklearn ](http://scikit-learn.sourceforge.net/stable/modules/hmm.html)
2. Python [seqLearn](http://larsmans.github.io/seqlearn/reference.html) - supervised multinomial HMM

This youtube video [part1](https://www.youtube.com/watch?v=TPRoLreU9lA) - explains about the hidden markov model. It shows the visual representation of the model and how we go from that the formula: <figure><img src="../.gitbook/assets/gimg-be8663091250.png" alt=""><figcaption><p>Figure.</p><p>Credit: <a href="https://lh6.googleusercontent.com/H4cc7N9jYDubaIjtW7KKpJaGZ0vVa9BhLnzmCYtxtHzFoDiWm5V6oleAc9nV_3IxJ3sd8iIn1TixXhgMNNPIHSaY_Y5F3bXaFW1ujecr_wpHzqnS0mQF-cTIcmRnNAMWtbie1VI7">copied from the original hosted image</a>.</p></figcaption></figure>

It breaks down the formula to:

- transition probability formula - the probability of going from Zk to Zk+1
- emission probability formula - the probability of going from Zk to Xk
- (Pi) Initial distribution - the probability of Z1=i for i=1..m

<figure><img src="../.gitbook/assets/gimg-b1373985e3c7.png" alt=""><figcaption><p>Figure.</p><p>Credit: <a href="https://lh5.googleusercontent.com/4H0tKAQZosxj0cGmCcy98By6AqS3BooOvgBBLftz2Q85jeHWCUf2Ur9wGOa_OwvsC46lVOVk8i6j2uZHgRgf0DIeyOkLaY-m3NgLUUDaFVhqiFYtFlUdaYxSy0qwXPSJ2Je-zcfP">copied from the original hosted image</a>.</p></figcaption></figure>

In [part2](https://www.youtube.com/watch?v=M_IIW0VYMEA) of the video:

\* HMM in weka, with github, working on 7.3, not on 9.1

1. Probably the simplest explanation of Markov Models and HMM as a “game” - [link](http://www.fejes.ca/EasyHMM.html)
2. This [video](https://www.youtube.com/watch?v=jY2E6ExLxaw) explains that building blocks of the needed knowledge in HMM, starting probabilities P0, transitions and emissions (state probabilities)
3. This [post](https://www.quora.com/What-is-a-simple-explanation-of-the-Hidden-Markov-Model-algorithm), explains HMM and ties our understanding.

[A cute explanation on quora](https://www.quora.com/What-is-a-simple-explanation-of-the-Hidden-Markov-Model-algorithm):

<figure><img src="../.gitbook/assets/gimg-395ef142c343.png" alt=""><figcaption><p>Figure.</p><p>Credit: <a href="https://lh4.googleusercontent.com/NZOT7lKEm-kjQS4J_L161Pdu6vVA9SmamcNf2IISN2nl-uD35whZhjOH25t_JVePqB7dMh5q9nHRcThBc0iT0GHg326Attj5pAfROG9u1ZUaUObmFnGmPgYZTe_LXwghnhTQdvWI">copied from the original hosted image</a>.</p></figcaption></figure>

This is the iconic image of a Hidden Markov Model. There is some state (x) that changes with time (markov). And you want to estimate or track it. Unfortunately, you cannot directly observe this state (hidden). That's the hidden part. But, you can observe something correlated with the state (y).

OBSERVED DATA -> INFER -> what you CANT OBSERVE (HIDDEN).

<figure><img src="../.gitbook/assets/gimg-876498fdeeba.png" alt=""><figcaption><p>Figure.</p><p>Credit: <a href="https://lh3.googleusercontent.com/p3MzUK2Vwne89LbeUW_f49e3GuIO62OXDvXNGuZaLWeuTac0D5K5jXoTdJbhomJQqT6wsYSWzWeZ7G4ITvvoy958cHYrtojcjwF0ucQCrhwHekUZmXgB8HFGaAOX30xMf2oP3TRn">copied from the original hosted image</a>.</p></figcaption></figure>

Considering this model:

- where P(X0) is the initial state for happy or sad
- Where P(Xt | X t-1) is the transition model from time-1 to time
- Where P(Yt | Xt) is the observation model for happy and sad (X) in 4 situations (w, sad, crying, facebook)

<figure><img src="../.gitbook/assets/gimg-4b6256a7a198.png" alt=""><figcaption><p>Figure.</p><p>Credit: <a href="https://lh4.googleusercontent.com/5MOIyOwwg7VU39m2L2OqNM8VWatLz4bXCN3i1x6c9cQSJWaEeR6leubji6Bt0F-ptUJcXGYuIKjtTUmeh9iZCumgy6PPYESHzaBXOWk2fjeidWXaUIa2lNQsFW3wFhdP2BHWfKwW">copied from the original hosted image</a>.</p></figcaption></figure>

#### INPUT OUTPUT HMM (IOHMM)

This subsection links IOHMM code and a probabilistic-machine-learning reference.

1. [Incomplete python code](https://github.com/Mogeng/IOHMM) for unsupervised / semi-supervised / supervised IOHMM - training is there, prediction is missing.
2. Machine learning - a probabilistic approach, david barber.

#### CONDITIONAL RANDOM FIELDS (CRF)

This subsection lists CRF intros, comparisons, and Python wrappers.

The same notes are in [CRF for templatization](templatization.md#crf-for-templatization), [Named Entity Recognition (NER)](../language-ai/named-entity-recognition-ner.md), and [Timeseries](forecasting.md).

1. [Make sense intro to CRF, comparison against HMM ](https://medium.com/ml2vec/overview-of-conditional-random-fields-68a2a20fa541)
2. [HMM, CRF, MEMM](https://medium.com/@Alibaba_Cloud/hmm-memm-and-crf-a-comparative-analysis-of-statistical-modeling-methods-49fc32a73586)
3. [Another crf article](https://medium.com/@phylypo/nlp-text-segmentation-using-conditional-random-fields-e8ff1d2b6060)
4. Neural network CRF [NNCRF](https://medium.com/@Akhilesh_k_r/neural-networks-conditional-random-field-crf-973712a0fd30)
5. Another one
6. [scikit-learn inspired API for CRFsuite](https://github.com/TeamHG-Memex/sklearn-crfsuite)
7. [Sklearn wrapper](https://github.com/supercoderhawk/sklearn-crfsuite)
8. [Python crfsuite](https://github.com/scrapinghub/python-crfsuite) wrapper
9. [Pycrf suite vidahya](https://www.analyticsvidhya.com/blog/2018/08/nlp-guide-conditional-random-fields-text-classification/)

