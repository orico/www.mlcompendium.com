# Deep Learning Models

## **AUTOENCODERS**

1. [**How to use AE for dimensionality reduction + code**](https://statcompute.wordpress.com/2017/01/15/autoencoder-for-dimensionality-reduction/) **- using keras’ functional API**
2. [**Keras.io blog post about AE’s**](https://blog.keras.io/building-autoencoders-in-keras.html) **- regular, deep, sparse, regularized, cnn, variational**
   1. **A keras.io** [**replicate post**](https://towardsdatascience.com/deep-inside-autoencoders-7e41f319999f) **but explains AE quite nicely.**
3. [**Examples of vanilla, multi layer, CNN and sparse AE’s**](https://wiseodd.github.io/techblog/2016/12/03/autoencoders/)
4. [**Another example of CNN-AE**](https://hackernoon.com/autoencoders-deep-learning-bits-1-11731e200694)
5. [**Another AE tutorial**](https://towardsdatascience.com/how-to-reduce-image-noises-by-autoencoder-65d5e6de543)
6. [**Hinton’s coursera course**](https://www.coursera.org/learn/neural-networks/lecture/JiT1i/from-pca-to-autoencoders-5-mins) **on PCA vs AE, basically some info about what PCA does - maximizing variance and projecting and then what AE does and can do to achieve similar but non-linear dense representations**
7. [**A great tutorial on how does the clusters look like after applying PCA/ICA/AE**](https://www.kaggle.com/den3b81/2d-visualization-pca-ica-vs-autoencoders)
8. [**Another great presentation on PCA vs AE,**](https://web.cs.hacettepe.edu.tr/\~aykut/classes/fall2016/bbm406/slides/l25-kernel\_pca.pdf) **summarized in the KPCA section of this notebook. +**[**another one**](https://www.cs.toronto.edu/\~urtasun/courses/CSC411/14\_pca.pdf) **+**[**StackE**](https://stats.stackexchange.com/questions/261265/factor-analysis-vs-autoencoders)**xchange**
9. [**Autoencoder tutorial with python code and how to encode after**](https://ramhiser.com/post/2018-05-14-autoencoders-with-keras/)**,** [**mastery**](https://machinelearningmastery.com/encoder-decoder-attention-sequence-to-sequence-prediction-keras/)
10. [**Git code for low dimensional auto encoder**](https://github.com/Mylittlerapture/Low-Dimensional-Autoencoder)
11. [**Bart denoising AE**](https://arxiv.org/pdf/1910.13461.pdf)**, sequence to sequence pre training for NL generation translation and comprehension.**
12. [**Attention based seq to seq auto encoder**](https://wanasit.github.io/attention-based-sequence-to-sequence-in-keras.html)**,** [**git**](https://github.com/wanasit/katakana)

[**AE for anomaly detection, fraud detection**](https://medium.com/@curiousily/credit-card-fraud-detection-using-autoencoders-in-keras-tensorflow-for-hackers-part-vii-20e0c85301bd)

## **Variational AE**

1. **Unread -** [**Simple explanation**](https://medium.com/@dmonn/what-are-variational-autoencoders-a-simple-explanation-ea7dccafb0e3)
2. [**Pixel art VAE**](https://mlexplained.wordpress.com/category/generative-models/vae/)
3. [**Unread - another VAE**](https://towardsdatascience.com/teaching-a-variational-autoencoder-vae-to-draw-mnist-characters-978675c95776)
4. [**Pixel GAN VAE**](https://medium.com/@Synced/pixelgan-autoencoders-17496632b755)
5. [**Disentangled VAE**](https://www.youtube.com/watch?v=9zKuYvjFFS8) **- improves VAE**
6. **Optimus -** [**pretrained VAE**](https://github.com/ophiry/Optimus)**,** [**paper**](https://arxiv.org/abs/2004.04092)**,** [**Microsoft blog**](https://www.microsoft.com/en-us/research/blog/a-deep-generative-model-trifecta-three-advances-that-work-towards-harnessing-large-scale-power/)\*\*\*\*

![Optimus](<../.gitbook/assets/image (19).png>)

## **SELF ORGANIZING MAPS (SOM)**

1. **Git**
   1. [**Sompy**](https://github.com/sevamoo/SOMPY)**,**
   2. **\*\*\***[**minisom!**](https://github.com/JustGlowing/minisom)
   3. [**Many graph examples**](https://medium.com/@s.ganjoo96/self-organizing-maps-b2cf58b74fdb)**,** [**example**](https://github.com/lightsalsa251/Self-Organizing-Map)
2. [**Step by step with examples, calculations**](https://mc.ai/self-organizing-mapsom/)
3. [**Adds intuition regarding “magnetism”’**](https://towardsdatascience.com/self-organizing-maps-1b7d2a84e065)
4. [**Implementation and faces**](https://medium.com/@navdeepsingh\_2336/self-organizing-maps-for-machine-learning-algorithms-ad256a395fc5)**, intuition towards each node and what it represents in a vision. I.e., each face resembles one of K clusters.**
5. [**Medium on kohonen networks, i.e., SOM**](https://towardsdatascience.com/kohonen-self-organizing-maps-a29040d688da)
6. [**Som on iris**](https://towardsdatascience.com/self-organizing-maps-ff5853a118d4)**, explains inference - averaging, and cons of the method.**
7. [**Simple explanation**](https://medium.com/@valentinerutto/selforganizingmaps-in-english-35574f95b0ac)
8. [**Algorithm, formulas**](https://towardsdatascience.com/kohonen-self-organizing-maps-a29040d688da)

## **NEURO EVOLUTION (GA/GP based)**

**NEAT**

[**NEAT**](http://www.cs.ucf.edu/\~kstanley/neat.html) \*\*stands for NeuroEvolution of Augmenting Topologies. It is a method for evolving artificial neural networks with a genetic algorithm.

NEAT implements the idea that it is most effective to start evolution with small, simple networks and allow them to become increasingly complex over generations.\*\*

\*\*That way, just as organisms in nature increased in complexity since the first cell, so do neural networks in NEAT.

This process of continual elaboration allows finding highly sophisticated and complex neural networks.\*\*

[**A great article about NEAT**](http://hunterheidenreich.com/blog/neuroevolution-of-augmenting-topologies/)

**HYPER-NEAT**

[**HyperNEAT**](http://eplex.cs.ucf.edu/hyperNEATpage/) \*\*computes the connectivity of its neural networks as a function of their geometry.

HyperNEAT is based on a theory of representation that hypothesizes that a good representation for an artificial neural network should be able to describe its pattern of connectivity compactly.\*\*

**The encoding in HyperNEAT, called** [**compositional pattern producing networks**](http://en.wikipedia.org/wiki/Compositional\_pattern-producing\_network)\*\*, is designed to represent patterns with regularities such as symmetry, repetition, and repetition with variationץ

(WIKI) **\[Compositional pattern-producing networks]\(**[https://en.wikipedia.org/wiki/Compositional\_pattern-producing\_network](https://en.wikipedia.org/wiki/Compositional\_pattern-producing\_network)**)** (CPPNs) are a variation of artificial neural networks (ANNs) that have an architecture whose evolution is guided by genetic algorithms\*\*

![](https://lh6.googleusercontent.com/cAbcsLDWcDOMlX4K53ROOLyiAw6EhJ9ZRDuZmURFtBaje8JtwzU\_KsOh4aeiC8ukdYgBYEm6zqWd7jZ3tStib3JJGYrmxM4wlrgyBJFhlnMHd\_kIcxgO2reEsoE4RPjJLXr3O-R\_)

[**A great HyperNeat tutorial on Medium.**](https://towardsdatascience.com/hyperneat-powerful-indirect-neural-network-evolution-fba5c7c43b7b)

## **Radial Basis Function Network (RBFN)**

**+** [**RBF layer in Keras.**](https://github.com/PetraVidnerova/rbf\_keras/blob/master/test.py)

**The** [**RBFN**](http://mccormickml.com/2013/08/15/radial-basis-function-network-rbfn-tutorial/) **approach is more intuitive than the MLP.**

* **An RBFN performs classification by measuring the input’s similarity to examples from the training set.**
* **Each RBFN neuron stores a “prototype”, which is just one of the examples from the training set.**
* **When we want to classify a new input, each neuron computes the Euclidean distance between the input and its prototype.**
* **Roughly speaking, if the input more closely resembles the class A prototypes than the class B prototypes, it is classified as class A.**

![Architecture\_Simple](https://lh6.googleusercontent.com/5oVVPw02w2Pv1kqAGvQ6drOX6Nh7lA72cBDplTbqgd78u25ceNdjufDe8h4pKWNPKC350\_r4V\_TPUn1ionjck1IPJiW0Q4rwivL4sH4LJGaj7V7WZBss8eLSuqpZb5Rv525M4sQ1)

## **Bayesian Neural Network (BNN)**

[**BNN**](https://eng.uber.com/neural-networks-uncertainty-estimation/) **- (what is?)** [**Bayesian neural network (BNN)**](http://edwardlib.org/tutorials/bayesian-neural-network) **according to Uber - architecture that more accurately forecasts time series predictions and uncertainty estimations at scale. “how Uber has successfully applied this model to large-scale time series anomaly detection, enabling better accommodate rider demand during high-traffic intervals.”**

**Under the BNN framework, prediction uncertainty can be categorized into three types:**

1. **Model uncertainty captures our ignorance of the model parameters and can be reduced as more samples are collected.**
2. **model misspecification**
3. **inherent noise captures the uncertainty in the data generation process and is irreducible.**

**Note: in a series of articles, uber explains about time series and leads to a BNN architecture.**

1. [**Neural networks**](https://eng.uber.com/neural-networks/) **- training on multi-signal raw data, training X and Y are window-based and the window size(lag) is determined in advance.**

**Vanilla LSTM did not work properly, therefore an architecture of**

**Regarding point 1: ‘run prediction with dropout 100 times’**

**\*\*\*** [**MEDIUM with code how to do it.**](https://medium.com/hal24k-techblog/how-to-generate-neural-network-confidence-intervals-with-keras-e4c0b78ebbdf)

[**Why do we need a confidence measure when we have a softmax probability layer?**](https://hjweide.github.io/quantifying-uncertainty-in-neural-networks) **The blog post explains, for example, that with a CNN of apples, oranges, cat and dogs, a non related example such as a frog image may influence the network to decide its an apple, therefore we can’t rely on the probability as a confidence measure. The ‘run prediction with dropout 100 times’ should give us a confidence measure because it draws each weight from a bernoulli distribution.**

**“By applying dropout to all the weight layers in a neural network, we are essentially drawing each weight from a** [**Bernoulli distribution**](https://en.wikipedia.org/wiki/Bernoulli\_distribution)**. In practice, this mean that we can sample from the distribution by running several forward passes through the network. This is referred to as** [**Monte Carlo dropout**](http://arxiv.org/abs/1506.02158)**.”**

**Taken from Yarin Gal’s** [**blog post**](http://mlg.eng.cam.ac.uk/yarin/blog\_3d801aa532c1ce.html) **. In this figure we see how sporadic is the signal from a forward pass (black line) compared to a much cleaner signal from 100 dropout passes.**

![](https://lh5.googleusercontent.com/FlcvG689kstX36ya8JNaeIE6C5HeXhL7IKG3wMt5zTacLqJVmb9W6kqpby\_e3IMV6iWc7rrIJ8F6IMwKEM6hUiuHnLaJiLp4KBPkTird\_AB4GW8i5-5n\_DOOm-cZEQYUsM6TWotp)

**Is it applicable for time series? In the figure below he tried to predict the missing signal between each two dotted lines, A is a bad estimation, but with a dropout layer we can see that in most cases the signal is better predicted.**

![](https://lh6.googleusercontent.com/eNr1VJ6ahkfVOvZ0i3HIFqng\_hyCYueyZQ5jqb20mB55MtZwpd8EJ6Qhda7Ty0oRwLsNFUN4YSUN2sAUW768lA2PyAqIUiLOMULMXZtBJKlU54Me0p2CeVJIkOubgoNV-hnwD5Ip)

**Going back to uber, they are actually using this idea to predict time series with LSTM, using encoder decoder framework.**

![](https://lh6.googleusercontent.com/OoKHnEH6OcZVOBorLKp-rvUFWueY6qjwLW\_v0mHWLGKp1YSZeRscteXA59Ecqp77B-PWv5nB7v6Hyf-emOu6eABkNW6LTAGEVSUgwtPLBKKJZBSRHIy8JbiCqwcc3-RbyiFvtd8z)

**Note: this is probably applicable in other types of networks.**

[**Phd Thesis by Yarin**](http://mlg.eng.cam.ac.uk/yarin/blog\_2248.html?fref=gc\&dti=999449923520287)**, he talks about uncertainty in Neural networks and using BNNs. he may have proved this thesis, but I did not read it. This blog post links to his full Phd.**

**Old note:** [**The idea behind uncertainty is (**](http://mlg.eng.cam.ac.uk/yarin/blog\_3d801aa532c1ce.html)[**paper here**](https://arxiv.org/pdf/1506.02142.pdf)**) that in order to trust your network’s classification, you drop some of the neurons during prediction, you do this \~100 times and you average the results. Intuitively this will give you confidence in your classification and increase your classification accuracy, because only a partial part of your network participated in the classification, randomly, 100 times. Please note that Softmax doesn't give you certainty.**

[**Medium post on prediction with drop out**](https://towardsdatascience.com/is-your-algorithm-confident-enough-1b20dfe2db08)

**The** [**solution for keras**](https://github.com/keras-team/keras/issues/9412) **says to add trainable=true for every dropout layer and add another drop out at the end of the model. Thanks sam.**

**“import keras**

**inputs = keras.Input(shape=(10,))**

**x = keras.layers.Dense(3)(inputs)**

**outputs = keras.layers.Dropout(0.5)(x, training=True)**

**model = keras.Model(inputs, outputs)“**

## **CONVOLUTIONAL NEURAL NET**

![](https://lh5.googleusercontent.com/yw2GIv\_A\_BJLggUjAcF7K3NFbvf9BsGiMS4PQHgLjl6H5sAziuofhepBZOlsWvJnK296FbGTOGYsOdWCmkpyesvuO9BtqcReXIVQy2xT3SOCNIH4riyTrpjL7M2tOOlG6eH\_3SEN)

**(**[**an excellent and thorough explanation about LeNet**](https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/)**) -**

* **Convolution Layer primary purpose is to extract features from the input image. Convolution preserves the spatial relationship between pixels by learning image features using small squares of input data.**
* **ReLU (more in the activation chapter) - The purpose of ReLU is to introduce non-linearity in our ConvNet**
* **Spatial Pooling (also called subsampling or downsampling) reduces the dimensionality of each feature map but retains the most important information. Spatial Pooling can be of different types: Max, Average, Sum etc.**
* **Dense / Fully Connected - a traditional Multi Layer Perceptron that uses a softmax activation function in the output layer to classify. The output from the convolutional and pooling layers represent high-level features of the input image. The purpose of the Fully Connected layer is to use these features for classifying the input image into various classes based on the training dataset.**

**The overall training process of the Convolutional Network may be summarized as below:**

* **Step1: We initialize all filters and parameters / weights with random values**
* **Step2: The network takes a single training image as input, goes through the forward propagation step (convolution, ReLU and pooling operations along with forward propagation in the Fully Connected layer) and finds the output probabilities for each class.**
  * **Let's say the output probabilities for the boat image above are \[0.2, 0.4, 0.1, 0.3]**
  * **Since weights are randomly assigned for the first training example, output probabilities are also random.**
* **Step3: Calculate the total error at the output layer (summation over all 4 classes)**
  * **(L2) Total Error = ∑ ½ (target probability – output probability) ²**
* **Step4: Use Backpropagation to calculate the gradients of the error with respect to all weights in the network and use gradient descent to update all filter values / weights and parameter values to minimize the output error.**
  * **The weights are adjusted in proportion to their contribution to the total error.**
  * **When the same image is input again, output probabilities might now be \[0.1, 0.1, 0.7, 0.1], which is closer to the target vector \[0, 0, 1, 0].**
  * **This means that the network has learnt to classify this particular image correctly by adjusting its weights / filters such that the output error is reduced.**
  * **Parameters like number of filters, filter sizes, architecture of the network etc. have all been fixed before Step 1 and do not change during training process – only the values of the filter matrix and connection weights get updated.**
* **Step5: Repeat steps 2-4 with all images in the training set.**

**The above steps train the ConvNet – this essentially means that all the weights and parameters of the ConvNet have now been optimized to correctly classify images from the training set.**

**When a new (unseen) image is input into the ConvNet, the network would go through the forward propagation step and output a probability for each class (for a new image, the output probabilities are calculated using the weights which have been optimized to correctly classify all the previous training examples). If our training set is large enough, the network will (hopefully) generalize well to new images and classify them into correct categories.**

[**Illustrated 10 CNNS architectures**](https://towardsdatascience.com/illustrated-10-cnn-architectures-95d78ace614d)

[**A study that deals with class imbalance in CNN’s**](https://arxiv.org/pdf/1710.05381.pdf) **- we systematically investigate the impact of class imbalance on classification performance of convolutional neural networks (CNNs) and compare frequently used methods to address the issue**

1. **Over sampling**
2. **Undersampling**
3. **Thresholding probabilities (ROC?)**
4. **Cost sensitive classification -different cost to misclassification**
5. **One class - novelty detection. This is a concept learning technique that recognizes positive instances rather than discriminating between two classes**

**Using several imbalance scenarios, on several known data sets, such as MNIST**![](https://lh5.googleusercontent.com/dsLGbR3YBUjsDjRuOiC5FSrfef4MoK2Y1J-wPzn4NmIJWxg3wP7aY8TvP1EXr8p6a4T5wjcFqv2teT11KlXaMQFh3eWOYRT-5Vn-xlAlacyckL7DDsAx4sJG5lt\_tJC4rF2ytfhs)

**The results indication (loosely) that oversampling is usually better in most cases, and doesn't cause overfitting in CNNs.**

**CONV-1D**

1. [**How to setup a conv1d in keras, most importantly how to reshape your input vector**](https://stackoverflow.com/questions/43396572/dimension-of-shape-in-conv1d/43399308#43399308)
2. [**Mastery on Character ngram cnn for sentiment analysis**](https://machinelearningmastery.com/develop-n-gram-multichannel-convolutional-neural-network-sentiment-analysis/)

**1x1 CNN**

1. [**Mastery**](https://machinelearningmastery.com/introduction-to-1x1-convolutions-to-reduce-the-complexity-of-convolutional-neural-networks/) **on 1x1 cnn, for dim reduction, decreasing feature maps and other usages.**
   1. **“This is the most common application of this type of filter and in this way, the layer is often called a feature map pooling layer.”**
   2. **“In the paper, the authors propose the need for an MLP convolutional layer and the need for cross-channel pooling to promote learning across channels.”**
   3. **“the 1×1 filter was used explicitly for dimensionality reduction and for increasing the dimensionality of feature maps after pooling in the design of the inception module, used in the GoogLeNet model”**
   4. **“The 1×1 filter was used as a projection technique to match the number of filters of input to the output of residual modules in the design of the residual network “**
   5.

**MASKED R-CNN**

[**1. Using mask rnn for object detection**](https://machinelearningmastery.com/how-to-perform-object-detection-in-photographs-with-mask-r-cnn-in-keras/)

**Invariance in CNN**

1. [**Making cnn shift invariance**](https://richzhang.github.io/antialiased-cnns/) **- “Small shifts -- even by a single pixel -- can drastically change the output of a deep network (bars on left). We identify the cause: aliasing during downsampling. We anti-alias modern deep networks with classic signal processing, stabilizing output classifications (bars on right). We even observe accuracy increases (see plot below).**

**MAX AVERAGE POOLING**

[**Intuitions to the differences between max and average pooling:**](https://stats.stackexchange.com/questions/291451/feature-extracted-by-max-pooling-vs-mean-pooling)

1. **A max-pool layer compressed by taking the maximum activation in a block. If you have a block with mostly small activation, but a small bit of large activation, you will loose the information on the low activations. I think of this as saying "this type of feature was detected in this general area".**
2. **A mean-pool layer compresses by taking the mean activation in a block. If large activations are balanced by negative activations, the overall compressed activations will look like no activation at all. On the other hand, you retain some information about low activations in the previous example.**
3. **MAX pooling In other words: Max pooling roughly means that only those features that are most strongly triggering outputs are used in the subsequent layers. You can look at it a little like focusing the network’s attention on what’s most characteristic for the image at hand.**
4. [**GLOBAL MAX pooling**](https://alexisbcook.github.io/2017/global-average-pooling-layers-for-object-localization/)**: In the last few years, experts have turned to global average pooling (GAP) layers to minimize overfitting by reducing the total number of parameters in the model. Similar to max pooling layers, GAP layers are used to reduce the spatial dimensions of a three-dimensional tensor. However, GAP layers perform a more extreme type of dimensionality reduction,**
5. [**Hinton’s controversy thoughts on pooling**](https://mirror2image.wordpress.com/2014/11/11/geoffrey-hinton-on-max-pooling-reddit-ama/)

**Dilated CNN**

1. [**For improved performance**](https://stackoverflow.com/questions/41178576/whats-the-use-of-dilated-convolutions)
2. \*\*\*\*[**RESNET, DENSENET UNET**](https://medium.com/swlh/resnets-densenets-unets-6bbdbcfdf010) **- the trick behind them, concatenating both f(x) = x**

## **Graph Convolutional Networks**

[**Explaination here, with some examples**](https://tkipf.github.io/graph-convolutional-networks/)

## **CAPSULE NEURAL NETS**

1. [**The solution to CNN’s shortcomings**](https://hackernoon.com/capsule-networks-are-shaking-up-ai-heres-how-to-use-them-c233a0971952)**, where features can be identified without relations to each other in an image, i.e. changing the location of body parts will not affect the classification, and changing the orientation of the image will. The promise of capsule nets is that these two issues are solved.**
2. [**Understanding capsule nets - part 2,**](https://medium.com/ai%C2%B3-theory-practice-business/understanding-hintons-capsule-networks-part-ii-how-capsules-work-153b6ade9f66) **there are more parts to the series**

## **Transfer Learning using CNN**

1. **To Add keras book chapter 5 (i think)**
2. [**Mastery**](https://machinelearningmastery.com/how-to-use-transfer-learning-when-developing-convolutional-neural-network-models/) **on TL using CNN**
   1. **Classifier: The pre-trained model is used directly to classify new images.**
   2. **Standalone Feature Extractor: The pre-trained model, or some portion of the model, is used to pre-process images and extract relevant features.**
   3. **Integrated Feature Extractor: The pre-trained model, or some portion of the model, is integrated into a new model, but layers of the pre-trained model are frozen during training.**
   4. **Weight Initialization: The pre-trained model, or some portion of the model, is integrated into a new model, and the layers of the pre-trained model are trained in concert with the new model.**

## **VISUALIZE CNN**

1. [**How to**](https://towardsdatascience.com/how-to-visualize-convolutional-features-in-40-lines-of-code-70b7d87b0030)

## **Recurrent Neural Net (RNN)**

### **RNN**

&#x20;**a basic NN node with a loop, previous output is merged with current input (using tanh?), for the purpose of remembering history, for time series - to predict the next X based on the previous Y.**

**(What is RNN?) by Andrej Karpathy -** [**The Unreasonable Effectiveness of Recurrent Neural Networks**](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)**, basically a lot of information about RNNs and their usage cases 1 to N = frame captioning**

* **N to 1 = classification**
* **N to N = predict frames in a movie**
* **N\2 with time delay to N\2 = predict supply and demand**
* **Vanishing gradient is 100 times worse.**
* **Gate networks like LSTM solves vanishing gradient.**

**(how to initialize?)** [**Benchmarking RNN networks for text**](https://danijar.com/benchmarking-recurrent-networks-for-language-modeling) **- don't worry about initialization, use normalization and GRU for big networks.**

**\*\* Experimental improvements:**

[**Ref**](https://arxiv.org/abs/1709.02755) **- ”Simplified RNN, with pytorch implementation” - changing the underlying mechanism in RNNs for the purpose of parallelizing calculation, seems to work nicely in terms of speed, not sure about state of the art results.** [**Controversy regarding said work**](https://www.facebook.com/cho.k.hyun/posts/10208564563785149)**, author claims he already mentioned these ideas (QRNN)** [**first**](https://www.reddit.com/r/MachineLearning/comments/6zduh2/r\_170902755\_training\_rnns\_as\_fast\_as\_cnns/dmv9gnh/)**, a year before, however it seems like his ideas have also been reviewed as** [**incremental**](https://openreview.net/forum?id=H1zJ-v5xl) **(PixelRNN). Its probably best to read all 3 papers in chronological order and use the most optimal solution.**

[**RNNCELLS - recurrent shop**](https://github.com/farizrahman4u/recurrentshop)**, enables you to build complex rnns with keras. Details on their significance are inside the link**

**Masking for RNNs - the ideas is simple, we want to use variable length inputs, although rnns do use that, they require a fixed size input. So masking of 1’s and 0’s will help it understand the real size or where the information is in the input. Motivation: Padded inputs are going to contribute to our loss and we dont want that.**

[**Source 1**](https://www.quora.com/What-is-masking-in-a-recurrent-neural-network-RNN)**,** [**source 2**](https://r2rt.com/recurrent-neural-networks-in-tensorflow-iii-variable-length-sequences.html)**,**

**Visual attention RNNS - Same idea as masking but on a window-based cnn.** [**Paper**](https://papers.nips.cc/paper/5542-recurrent-models-of-visual-attention.pdf)

### **LSTM**

* [**The best, hands down, lstm post out there**](https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21)
* **LSTM -** [**what is?**](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) **the first reference for LSTM on the web, but you should know the background before reading.**
* ![](https://lh3.googleusercontent.com/7KJz\_beT-3kClxvDJHNVZP4gEMtn0oUK08yzh\_foRMwqjtrWh8EpC3Yp9oCmH0LOcBzBbA-8E9D-4Dd1TXdWipGjSHXW0GjgMBo4gs-1f8XLpXRjnwN29zhzpJPe2uKIyNXkkqy-)
* [**Hidden state vs cell state**](https://www.quora.com/How-is-the-hidden-state-h-different-from-the-memory-c-in-an-LSTM-cell) **- you have to understand this concept before you dive in. i.e, Hidden state is overall state of what we have seen so far. Cell state is selective memory of the past. The hidden state (h) carries the information about what an RNN cell has seen over the time and supply it to the present time such that a loss function is not just dependent upon the data it is seeing in this time instant, but also, data it has seen historically.**
* [**Illustrated rnn lstm gru**](https://towardsdatascience.com/animated-rnn-lstm-and-gru-ef124d06cf45)
* [**Paper**](https://arxiv.org/pdf/1503.04069.pdf) **- a comparison of many LSTMs variants and they are pretty much the same performance wise**
* [**Paper**](https://arxiv.org/pdf/1503.04069.pdf) **- comparison of lstm variants, vanilla is mostly the best, forget and output gates are the most important in terms of performance. Other conclusions in the paper..**
* **Master on** [**unrolling RNN’s introductory post**](https://machinelearningmastery.com/rnn-unrolling/)
* **Mastery on** [**under/over fitting lstms**](https://machinelearningmastery.com/diagnose-overfitting-underfitting-lstm-models/) **- but makes sense for all types of networks**
* **Mastery on r**[**eturn\_sequence and return\_state in keras LSTM**](https://machinelearningmastery.com/return-sequences-and-return-states-for-lstms-in-keras/)
  * **That return sequences return the hidden state output for each input time step.**
  * **That return state returns the hidden state output and cell state for the last input time step.**
  * **That return sequences and return state can be used at the same time.**
* **Mastery on** [**understanding stateful vs stateless**](https://machinelearningmastery.com/understanding-stateful-lstm-recurrent-neural-networks-python-keras/)**,** [**stateful stateless for time series**](https://machinelearningmastery.com/stateful-stateless-lstm-time-series-forecasting-python/)
* **Mastery on** [**timedistributed layer**](https://machinelearningmastery.com/timedistributed-layer-for-long-short-term-memory-networks-in-python/) **and seq2seq**
  * **TimeDistributed Layer - used to connect 3d inputs from lstms to dense layers, in order to utilize the time element. Otherwise it gets flattened when the connection is direct, nulling the lstm purpose. Note: nice trick that doesn't increase the dense layer structure multiplied by the number of dense neurons. It loops for each time step! I.e., The TimeDistributed achieves this trick by applying the same Dense layer (same weights) to the LSTMs outputs for one time step at a time. In this way, the output layer only needs one connection to each LSTM unit (plus one bias).**

**For this reason, the number of training epochs needs to be increased to account for the smaller network capacity. I doubled it from 500 to 1000 to match the first one-to-one example**

* **Sequence Learning Problem**
* **One-to-One LSTM for Sequence Prediction**
* **Many-to-One LSTM for Sequence Prediction (without TimeDistributed)**
* **Many-to-Many LSTM for Sequence Prediction (with TimeDistributed)**
*
  * **Mastery on** [**wrapping cnn-lstm with time distributed**](https://machinelearningmastery.com/cnn-long-short-term-memory-networks/)**, as a whole model wrap, or on every layer in the model which is equivalent and preferred.**
* **Master on** [**visual examples**](https://machinelearningmastery.com/sequence-prediction/) **for sequence prediction**
* **Unread - sentiment classification of IMDB movies using** [**Keras and LSTM**](http://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/)
* [**Very important - how to interpret LSTM neurons in keras**](https://yerevann.github.io/2017/06/27/interpreting-neurons-in-an-LSTM-network/)
* [**LSTM for time-series**](http://www.jakob-aungiers.com/articles/a/LSTM-Neural-Network-for-Time-Series-Prediction) **- (jakob) single point prediction, sequence prediction and shifted-sequence prediction with code.**

**Stateful vs Stateless: crucial for understanding how to leverage LSTM networks:**

1. [**A good description on what it is and how to use it.**](https://groups.google.com/forum/#!topic/keras-users/l1RV\_tthjoY)
2. [**ML mastery**](https://machinelearningmastery.com/stateful-stateless-lstm-time-series-forecasting-python/) _\*\*_
3. [**Philippe remy**](http://philipperemy.github.io/keras-stateful-lstm/) **on stateful vs stateless, intuition mostly with code, but not 100% clear**

**Machine Learning mastery:**

[**A good tutorial on LSTM:**](https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/) **important notes:**

**1. Scale to -1,1, because the internal activation in the lstm cell is tanh.**

**2.**[**stateful**](https://machinelearningmastery.com/understanding-stateful-lstm-recurrent-neural-networks-python-keras/) **- True, needs to reset internal states, False =stateless. Great info & results** [**HERE**](https://machinelearningmastery.com/stateful-stateless-lstm-time-series-forecasting-python/)**, with seeding, with training resets (and not) and predicting resets (and not) - note: empirically matching the shampoo input, network config, etc.**

[**Another explanation/tutorial about stateful lstm, should be thorough.**](http://philipperemy.github.io/keras-stateful-lstm/)

**3.** [**what is return\_sequence, return\_states**](https://machinelearningmastery.com/return-sequences-and-return-states-for-lstms-in-keras/)**, and how to use each one and both at the same time.**

**Return\_sequence is needed for stacked LSTM layers.**

**4.**[**stacked LSTM**](https://machinelearningmastery.com/stacked-long-short-term-memory-networks/) **- each layer has represents a higher level of abstraction in TIME!**

[**Keras Input shape**](https://stackoverflow.com/questions/44747343/keras-input-explanation-input-shape-units-batch-size-dim-etc) **- a good explanation about differences between input\_shape, dim, and what is. Additionally about layer calculation of inputs and output based on input shape, and sequence model vs API model.**

**A** [**comparison**](https://danijar.com/language-modeling-with-layer-norm-and-gru/) **of LSTM/GRU/MGU with batch normalization and various initializations, GRu/Xavier/Batch are the best and recommended for RNN**

[**Benchmarking LSTM variants**](http://proceedings.mlr.press/v37/jozefowicz15.pdf)**: - it looks like LSTM and GRU are competitive to mutation (i believe its only in pytorch) adding a bias to LSTM works (a bias of 1 as recommended in the** [**paper**](https://pdfs.semanticscholar.org/1154/0131eae85b2e11d53df7f1360eeb6476e7f4.pdf)**), but generally speaking there is no conclusive empirical evidence that says one type of network is better than the other for all tests, but the mutated networks tend to win over lstm\gru variants.**

[**BIAS 1 in keras**](https://keras.io/layers/recurrent/#lstm) **- unit\_forget\_bias: Boolean. If True, add 1 to the bias of the forget gate at initializationSetting it to true will also force bias\_initializer="zeros". This is recommended in** [**Jozefowicz et al.**](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)

![](https://lh3.googleusercontent.com/fiS0-IpAswRrHvrmnmFA-rrfd1h0rzoxmiZlPHQmBpcOrkbQXxzm9Z-5Q5HPsW26D\_qsxzmriQ2tMWCmlG6jP0W5riP-yKjME1vjX-empGjSgycHKyxZZgt916uqiUmuLk4aecb2)

[**Validation\_split arg**](https://www.quora.com/What-is-the-importance-of-the-validation-split-variable-in-Keras) **- The validation split variable in Keras is a value between \[0..1]. Keras proportionally split your training set by the value of the variable. The first set is used for training and the 2nd set for validation after each epoch.**

**This is a nice helper add-on by Keras, and most other Keras examples you have seen the training and test set was passed into the fit method, after you have manually made the split. The value of having a validation set is significant and is a vital step to understand how well your model is training. Ideally on a curve you want your training accuracy to be close to your validation curve, and the moment your validation curve falls below your training curve the alarm bells should go off and your model is probably busy over-fitting.**

**Keras is a wonderful framework for deep learning, and there are many different ways of doing things with plenty of helpers.**

[**Return\_sequence**](https://stackoverflow.com/questions/42755820/how-to-use-return-sequences-option-and-timedistributed-layer-in-keras)**: unclear.**

[**Sequence.pad\_sequences**](https://stackoverflow.com/questions/42943291/what-does-keras-io-preprocessing-sequence-pad-sequences-do) **- using maxlength it will either pad with zero if smaller than, or truncate it if bigger.**

[**Using batch size for LSTM in Keras**](https://machinelearningmastery.com/use-different-batch-sizes-training-predicting-python-keras/)

**Imbalanced classes? Use** [**class\_weight**](https://stackoverflow.com/questions/43459317/keras-class-weight-vs-sample-weights-in-the-fit-generator)**s, another explanation** [**here**](https://stackoverflow.com/questions/43459317/keras-class-weight-vs-sample-weights-in-the-fit-generator) **about class\_weights and sample\_weights.**

**SKlearn Formula for balanced class weights and why it works,** [**example**](https://stackoverflow.com/questions/50152377/in-sklearn-logistic-regression-class-balanced-helps-run-the-model-with-imbala/50154388)

[**number of units in LSTM**](https://www.quora.com/What-is-the-meaning-of-%E2%80%9CThe-number-of-units-in-the-LSTM-cell)

[**Calculate how many params are in an LSTM layer?**](https://stackoverflow.com/questions/38080035/how-to-calculate-the-number-of-parameters-of-an-lstm-network)

![](https://lh5.googleusercontent.com/niwCPHMxrR83JzXNLWT8J4dr9S\_GJ4\_Z4SEDMwPQFv6OghMu9S2X2A5cy9wUwTnaAehXU18IIVM4s--tRnANN8AxnMUOogOt6WjF5azZc0ootq5EIHgj9hfxL253oMCWaAm8ftQj)

[**Understanding timedistributed in Keras**](https://machinelearningmastery.com/timedistributed-layer-for-long-short-term-memory-networks-in-python/)**, but with focus on lstm one to one, one to many and many to many - here the timedistributed is applying a dense layer to each output neuron from the lstm, which returned\_sequence = true for that purpose.**

**This tutorial clearly shows how to manipulate input construction, lstm output neurons and the target layer for the purpose of those three problems (1:1, 1:m, m:m).**

**BIDIRECTIONAL LSTM**

**(what is?) Wiki - The basic idea of BRNNs is to connect two hidden layers of opposite directions to the same output. By this structure, the output layer can get information from past and future states.**

**BRNN are especially useful when the context of the input is needed. For example, in handwriting recognition, the performance can be enhanced by knowledge of the letters located before and after the current letter.**

[**Another**](https://machinelearningmastery.com/develop-bidirectional-lstm-sequence-classification-python-keras/) **explanation- It involves duplicating the first recurrent layer in the network so that there are now two layers side-by-side, then providing the input sequence as-is as input to the first layer and providing a reversed copy of the input sequence to the second.**

**.. It allows you to specify the merge mode, that is how the forward and backward outputs should be combined before being passed on to the next layer. The options are:**

* **‘sum‘: The outputs are added together.**
* **‘mul‘: The outputs are multiplied together.**
* **‘concat‘: The outputs are concatenated together (the default), providing double the number of outputs to the next layer.**
* **‘ave‘: The average of the outputs is taken.**

**The default mode is to concatenate, and this is the method often used in studies of bidirectional LSTMs.**

[**Another simplified example**](https://stackoverflow.com/questions/43035827/whats-the-difference-between-a-bidirectional-lstm-and-an-lstm)

### **BACK PROPAGATION**

[**A great Slide about back prop, on a simple 3 neuron network, with very easy to understand calculations.**](https://www.slideshare.net/AhmedGadFCIT/backpropagation-understanding-how-to-update-anns-weights-stepbystep)

### **UNSUPERVISED LSTM**

1. [**Paper**](ftp://ftp.idsia.ch/pub/juergen/icann2001unsup.pdf)**,** [**paper2**](https://arxiv.org/pdf/1502.04681.pdf)**,** [**paper3**](https://arxiv.org/abs/1709.02081)
2. [**In keras**](https://www.reddit.com/r/MachineLearning/comments/4adrie/unsupervised\_lstm\_using\_keras/)

### **GRU**

[**A tutorial about GRU**](https://towardsdatascience.com/understanding-gru-networks-2ef37df6c9be) **- To solve the vanishing gradient problem of a standard RNN, GRU uses, so called, update gate and reset gate. Basically, these are two vectors which decide what information should be passed to the output. The special thing about them is that they can be trained to keep information from long ago, without washing it through time or remove information which is irrelevant to the prediction.**

1. **update gate helps the model to determine how much of the past information (from previous time steps) needs to be passed along to the future.**
2. **Reset gate essentially, this gate is used from the model to decide how much of the past information to forget.**

**RECURRENT WEIGHTED AVERAGE (RNN-WA)**

**What is? (a type of cell that converges to higher accuracy faster than LSTM.**

**it implements attention into the recurrent neural network:**

**1. the keras implementation is available at** [**https://github.com/keisuke-nakata/rwa**](https://github.com/keisuke-nakata/rwa) _\*\*_

**2. the whitepaper is at** [**https://arxiv.org/pdf/1703.01253.pdf**](https://arxiv.org/pdf/1703.01253.pdf)

![](https://lh6.googleusercontent.com/OgNIg0\_EssPKTLuvrFf2cz3R89QeP4FYh7kLrk0J-\_AIDjcgaVirW\_d668aFDlPXW8mSF2CBtHDgCpiQoFDgc12bChOeePfbyWq1-ybMDdZSga6ezEdr16dKjiFEok8Oajn5XLFm)

### **QRNN**

[**Potential competitor to the transformer**](https://towardsdatascience.com/qrnn-a-potential-competitor-to-the-transformer-86b5aef6c137)

## **GRAPH NEURAL NETWORKS (GNN)**

1. **(amazing)** [**Why i am luke warm about GNN’s**](https://www.singlelunch.com/2020/12/28/why-im-lukewarm-on-graph-neural-networks/) **- really good insight to what they do (compressing data, vs adjacy graphs, vs graphs, high dim relations, etc.)**
2. (amazing) [Graphical intro to GNNs](https://distill.pub/2021/gnn-intro/)
3. [**Learning on graphs youtube - uriel singer**](https://www.youtube.com/watch?v=snLsWos\_1WU\&feature=youtu.be\&fbclid=IwAR0JlvF9aPgKMmeh2zGr3l3j\_8AebOTjknVGyMsz0Y2EvgcqrS0MmLkBTMU)
4. [**Benchmarking GNN’s, methodology, git, the works.**](https://graphdeeplearning.github.io/post/benchmarking-gnns/)
5. [**Awesome graph classification on github**](https://github.com/benedekrozemberczki/awesome-graph-classification)
6. **Octavian in medium on graphs,** [**A really good intro to graph networks, too long too summarize**](https://medium.com/octavian-ai/deep-learning-with-knowledge-graphs-3df0b469a61a)**, clever, mcgraph, regression, classification, embedding on graphs.**
7. [**Application of graph networks**](https://towardsdatascience.com/https-medium-com-aishwaryajadhav-applications-of-graph-neural-networks-1420576be574) _\*\*_
8. [**Recommender systems using GNN**](https://towardsdatascience.com/recommender-systems-applying-graph-and-nlp-techniques-619dbedd9ecc)**, w2v, pytorch w2v, networkx, sparse matrices, matrix factorization, dictionary optimization, part 1 here** [**(how to find product relations, important: creating negative samples)**](https://eugeneyan.com/2020/01/06/recommender-systems-beyond-the-user-item-matrix)
9. [**Transformers are GNN**](https://towardsdatascience.com/transformers-are-graph-neural-networks-bca9f75412aa)**, original:** [**Transformers are graphs, not the typical embedding on a graph, but a more holistic approach to understanding text as a graph.**](https://thegradient.pub/transformers-are-graph-neural-networks/)
10. [**Cnn for graphs**](https://towardsdatascience.com/how-to-do-deep-learning-on-graphs-with-graph-convolutional-networks-62acf5b143d0)
11. [**Staring with gnn**](https://medium.com/octavian-ai/how-to-get-started-with-machine-learning-on-graphs-7f0795c83763)
12. **Really good -** [**Basics deep walk and graphsage**](https://towardsdatascience.com/a-gentle-introduction-to-graph-neural-network-basics-deepwalk-and-graphsage-db5d540d50b3) _\*\*_
13. [**Application of gnn**](https://towardsdatascience.com/https-medium-com-aishwaryajadhav-applications-of-graph-neural-networks-1420576be574)
14. **Michael Bronstein’s** [**Central page for Graph deep learning articles on Medium**](https://towardsdatascience.com/graph-deep-learning/home) **(worth reading)**
15. [**GAT graphi attention networks**](https://petar-v.com/GAT/)**, paper, examples - The graph attentional layer utilised throughout these networks is computationally efficient (does not require costly matrix operations, and is parallelizable across all nodes in the graph), allows for (implicitly) assigning different importances to different nodes within a neighborhood while dealing with different sized neighborhoods, and does not depend on knowing the entire graph structure upfront—thus addressing many of the theoretical issues with approaches.**
16. **Medium on** [**Intro, basics, deep walk, graph sage**](https://towardsdatascience.com/a-gentle-introduction-to-graph-neural-network-basics-deepwalk-and-graphsage-db5d540d50b3)
17. [Struc2vec](https://leoribeiro.github.io/struc2vec.html), [youtube](https://www.youtube.com/watch?v=lu0xMOO48Xo\&embeds\_euri=https%3A%2F%2Fleoribeiro.github.io%2F\&source\_ve\_path=MjM4NTE\&feature=emb\_title): Learning Node Representations from Structural Identity- The _struc2vec_ algorithm learns continuous representations for nodes in any graph. struc2vec captures structural equivalence between nodes.

### GNN courses

1. [machine learning with graphs by Stanford](http://web.stanford.edu/class/cs224w/?fbclid=IwAR0nQR4lhyKCoTchsGQrcZ5E8EPBt2Bi4d8K8MYX-UN0ygQSxQ5bMoohhis), from ML to GNN.
2. [Graph deep learning course](https://geometricdeeplearning.com/lectures/) - graphs, sets, groups, GNNs. [youtube](https://www.youtube.com/watch?app=desktop\&v=w6Pw4MOzMuo)

### **Deep walk**

1. [**Git**](https://github.com/phanein/deepwalk)
2. [**Paper**](https://arxiv.org/abs/1403.6652)
3. [**Medium**](https://medium.com/@\_init\_/an-illustrated-explanation-of-using-skipgram-to-encode-the-structure-of-a-graph-deepwalk-6220e304d71b) **and medium on** [**W2v, deep walk, graph2vec, n2v**](https://towardsdatascience.com/graph-embeddings-the-summary-cc6075aba007)

### **Node2vec**

1. [**Git**](https://github.com/eliorc/node2vec)
2. [**Stanford**](https://snap.stanford.edu/node2vec/)
3. [**Elior on medium**](https://towardsdatascience.com/node2vec-embeddings-for-graph-data-32a866340fef)**,** [**youtube**](https://www.youtube.com/watch?v=828rZgV9t1g)
4. [**Paper**](https://cs.stanford.edu/\~jure/pubs/node2vec-kdd16.pdf)

### **Graphsage**

1. [**medium**](https://towardsdatascience.com/a-gentle-introduction-to-graph-neural-network-basics-deepwalk-and-graphsage-db5d540d50b3)

### **SDNE - structural deep network embedding**

1. [**medium**](https://towardsdatascience.com/graph-embeddings-the-summary-cc6075aba007)

### **Diff2vec**

1. [**Git**](https://github.com/benedekrozemberczki/diff2vec)
2. ![](https://lh6.googleusercontent.com/otaXffQv-FribLSm922jhO-904l0ZHD4QcWRJ0dgc7u4vW0HMP1cGP-QU63ohhJSLiUxpz5DTB9L6DsK1ettM0S1MRg76sZZhEjzezQpTDDrrXI6pnh5B-2aRrA8FxJrAJK\_fufn)

### **Splitter**

**,** [**git**](https://github.com/benedekrozemberczki/Splitter)**,** [**paper**](http://epasto.org/papers/www2019splitter.pdf)**, “Is a Single Embedding Enough? Learning Node Representations that Capture Multiple Social Contexts”**

**Recent interest in graph embedding methods has focused on learning a single representation for each node in the graph. But can nodes really be best described by a single vector representation? In this work, we propose a method for learning multiple representations of the nodes in a graph (e.g., the users of a social network). Based on a principled decomposition of the ego-network, each representation encodes the role of the node in a different local community in which the nodes participate. These representations allow for improved reconstruction of the nuanced relationships that occur in the graph a phenomenon that we illustrate through state-of-the-art results on link prediction tasks on a variety of graphs, reducing the error by up to 90%. In addition, we show that these embeddings allow for effective visual analysis of the learned community structure.**

![](https://lh3.googleusercontent.com/ZWvxCQ72uAo6J-nr2uojE4KYzqOvgm3dzzXSuKlP0nbry-qFhEbQVZIG4om\_SPLZpWZti3--aG1a6dYmOMnot--vFx0dnimMZDLz4LrjJQkRgAZY8ZospzEPKA9MrW\_\_We61ylD9)

![](https://lh5.googleusercontent.com/asBPQZ90fcBXUYlz3tT2uV2LbELCjHVm56nhjbvRFuW7UXFBDX8fy353dF\_6\_OFGHo7ioBmFOl5wwxsyfSJHhA2LIOS0LkOTIdI23WnTjHFIf-PFdr6tp5RG\_GaJF7BACv2RrJcK)

**16.** [**Self clustering graph embeddings**](https://github.com/benedekrozemberczki/GEMSEC)

![](https://lh5.googleusercontent.com/xLcNkor6PpkcSUl1sW9Ws36NxIrNr9kmdoBuhlPYnfCKlrC7zkaJwNIlSlIBDiXvL9OPi62lQ8q3ZA6oLXr\_pJfUJvUTmelHnEy7z2hivhQJxQN4Ppz8ZRCErtlLQzROyIoyZaV-)

**17.** [**Walklets**](https://github.com/benedekrozemberczki/walklets?fbclid=IwAR2ymD7lbgP\_sUde5UvKGZp7TYYYmACMFJS6UGNjqW29ethONHy7ibmDL0Q)**, similar to deep walk with node skips. - lots of improvements, works in scale due to lower size representations, improves results, etc.**

**Nodevectors**

[**Git**](https://github.com/VHRanger/nodevectors)**, The fastest network node embeddings in the west**![](https://lh3.googleusercontent.com/DwKfPhonL4At5xRePfv77SdSDjSZBYo\_Z0Qm1hAFNpLLEYtiGMQhN8QPLO\_5tNRr0NYvg3JRyYEECOUhjJkR6sK77k0M-Z1VVYcEwbBLU7cLqjlVN41IV5nGPt1yX8kYP-NlrqO9)

## **SIGNAL PROCESSING NN (FFT, WAVELETS, SHAPELETS)**

1. [**Fourier Transform**](https://www.youtube.com/watch?v=spUNpyF58BY) **- decomposing frequencies**
2. [**WAVELETS On youtube (4 videos)**](https://www.youtube.com/watch?v=QX1-xGVFqmw)**:**
   1. [**used for denoising**](https://www.youtube.com/watch?v=veCvP1mYpww)**, compression, detect edges, detect features with various orientation, analyse signal power, detect and localize transients, change points in time series data and detect optimal signal representation (peaks etc) of time freq analysis of images and data.**
   2. **Can also be used to** [**reconstruct time and frequencies**](https://www.youtube.com/watch?v=veCvP1mYpww)**, analyse images in space, frequencies, orientation, identifying coherent time oscillation in time series**
   3. **Analyse signal variability and correlation**
   4.

## **HIERARCHICAL RNN**

1. [**githubcode**](https://github.com/keras-team/keras/blob/master/examples/mnist\_hierarchical\_rnn.py)

## **NN-Sequence Analysis**

**(did not read)** [**A causal framework for explaining the predictions of black-box sequence-to-sequence models**](http://people.csail.mit.edu/tommi/papers/AlvJaa\_EMNLP2017.pdf) **- can this be applied to other time series prediction?**

## **SIAMESE NETWORKS (one shot)**

1. [**Siamese CNN, learns a similarity between images, not to classify**](https://medium.com/predict/face-recognition-from-scratch-using-siamese-networks-and-tensorflow-df03e32f8cd0)
2. [**Visual tracking, explains contrastive and triplet loss**](https://medium.com/intel-student-ambassadors/siamese-networks-for-visual-tracking-96262eaaba77)
3. [**One shot learning, very thorough, baseline vs siamese**](https://towardsdatascience.com/one-shot-learning-with-siamese-networks-using-keras-17f34e75bb3d)
4. [**What is triplet loss**](https://towardsdatascience.com/siamese-network-triplet-loss-b4ca82c1aec8)
5. **MULTI NETWORKS**
6. [**Google whitening black boxes using multi nets, segmentation and classification**](https://medium.com/health-ai/google-deepmind-might-have-just-solved-the-black-box-problem-in-medical-ai-3ed8bc21f636)
