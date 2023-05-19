# Deep Neural Nets Basics

## Perceptron

1. [perceptron](https://towardsdatascience.com/perceptrons-logical-functions-and-the-xor-problem-37ca5025790a) - logical functions and XOR
2. The chain rule
   1. [mastery on the chain rule for multi and univariate functions](https://machinelearningmastery.com/the-chain-rule-of-calculus-for-univariate-and-multivariate-functions/)
   2. [derivative of a sigmoid](https://towardsdatascience.com/understanding-the-derivative-of-the-sigmoid-function-cbfd46fb3716)
   3. [derivative for ML people](https://towardsdatascience.com/a-quick-introduction-to-derivatives-for-machine-learning-people-3cd913c5cf33)
3. [Step by step backpropagation example](https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/)&#x20;
4. [understanding backprop](https://towardsdatascience.com/understanding-backpropagation-algorithm-7bb3aa2f95fd)

## DNN

* [**Deep learning notes from Andrew NG’s course.**](https://www.slideshare.net/TessFerrandez/notes-from-coursera-deep-learning-courses-by-andrew-ng)
* **Jay Alammar on NN** [**Part 1**](http://jalammar.github.io/visual-interactive-guide-basics-neural-networks/)**,** [**Part 2**](http://jalammar.github.io/feedforward-neural-networks-visual-interactive/)
* [**NN in general**](http://briandolhansky.com/blog/?tag=neural+network#show-archive) **- 5 introductions tutorials.**
* [**Segmentation examples**](https://meetshah1995.github.io/semantic-segmentation/deep-learning/pytorch/visdom/2017/06/01/semantic-segmentation-over-the-years.html)

**MLP: fully connected, input, hidden layers, output. Gradient on the backprop takes a lot of time to calculate. Has vanishing gradient problem, because of multiplications when it reaches the first layers the loss correction is very small (0.1\*0.1\*01 = 0.001), therefore the early layers train slower than the last ones, and the early ones capture the basics structures so they are the more important ones.**

**AutoEncoder - unsupervised, drives the input through fully connected layers, sometime reducing their neurons amount, then does the reverse and expands the layer’s size to get to the input (images are multiplied by the transpose matrix, many times over), Comparing the predicted output to the input, correcting the cost using gradient descent and redoing it, until the networks learns the output.**

* **Convolutional auto encoder**
* **Denoiser auto encoder - masking areas in order to create an encoder that understands noisy images**
* **Variational autoencoder - doesnt rely on distance between pixels, rather it maps them to a function (gaussian), eventually the DS should be explained by this mapping, uses 2 new layers added to the network. Gaussian will create blurry images, but similar. Please note that it also works with CNN.**

**What are** [**logits**](https://stackoverflow.com/questions/41455101/what-is-the-meaning-of-the-word-logits-in-tensorflow) **in neural net - the vector of raw (non-normalized) predictions that a classification model generates, which is ordinarily then passed to a normalization function. If the model is solving a multi-class classification problem, logits typically become an input to the softmax function. The softmax function then generates a vector of (normalized) probabilities with one value for each possible class.**

[**WORD2VEC**](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/) **- based on autoencode, we keep only the hidden layer ,** [**Part 2**](http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/)

**RBM- restricted (no 2 nodes share a connection) boltzman machine**

**An Autoencoder of features, tries to encode its own structure.**

**Works best on pics, video, voice, sensor data. 2 layers, visible and hidden, error and bias calculated via KL Divergence.**

* **Also known as a shallow network.**
* **Two layers, input and output, goes back and forth until it learns its output.**

**DBN - deep belief networks, similar structure to multi layer perceptron. fully connected, input, hidden(s), output layers. Can be thought of as stacks of RBM. training using GPU optimization, accurate and needs smaller labelled data set to complete the training.**

**Solves the ‘vanishing gradient’ problem, imagine a fully connected network, advancing each 2 layers step by step until each boltzman network (2 layers) learns the output, keeps advancing until finished.. Each layer learns the entire input.**

**Next step is to fine tune using a labelled test set, improves performance and alters the net. So basically using labeled samples we fine tune and associate features and pattern with a name. Weights and biases are altered slightly and there is also an increase in performance. Unlike CNN which learns features then high level features.**

**Accurate and reasonable in time, unlike fully connected that has the vanishing gradient problem.**

**Transfer Learning = like Inception in Tensor flow, use a prebuilt network to solve many problems that “work” similarly to the original network.**

* [**CS course definition**](http://cs231n.github.io/transfer-learning/) **- also very good explanation of the common use cases:**
  * **Feature extraction from the CNN part (removing the fully connected layer)**
  * **Fine-tuning, everything or partial selection of the hidden layers, mainly good to keep low level neurons that know what edges and color blobs are, but not dog breeds or something not as general.**
* [**CNN checkpoints**](https://github.com/BVLC/caffe/wiki/Model-Zoo#cascaded-fully-convolutional-networks-for-biomedical-image-segmentation) **for many problems with transfer learning. Has several relevant references**
* **Such as this “**[**How transferable are features in deep neural networks?**](http://arxiv.org/abs/1411.1792) **“**
* **(the indian guy on facebook)** [**IMDB transfer learning using cnn vgg and word2vec**](https://spandan-madan.github.io/DeepLearningProject/)**, the word2vec is interesting, the cnn part is very informative. With python code, keras.**

**CNN, Convolutional Neural Net (**[**this link explains CNN quite well**](https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/)**,** [**2nd tutorial**](https://hackernoon.com/deep-learning-cnns-in-tensorflow-with-gpus-cba6efe0acc2) **- both explain about convolution, padding, relu - sparsity, max and avg pooling):**

* **Common Layers: input->convolution->relu activation->pooling to reduce dimensionality \*\*\*\* ->fully connected layer**
* **\*\*\*\*repeat several times over as this discover patterns but needs another layer -> fully connected layer**
* **Then we connect at the end a fully connected layer (fcl) to classify data samples.**
* **Good for face detection, images etc.**
* **Requires lots of data, not always possible in a real world situation**
* **Relu is quite resistant to vanishing gradient & allows for deactivating neurons and for sparsity.**

**RNN - what is RNN by Andrej Karpathy -** [**The Unreasonable Effectiveness of Recurrent Neural Networks**](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)**, basically a lot of information about RNNs and their usage cases**

* **basic NN node with a loop, previous output is merged with current input. for the purpose of remembering history, for time series, to predict the next X based on the previous Y.**
* **1 to N = frame captioning**
* **N to 1 = classification**
* **N to N = predict frames in a movie**
* **N\2 with time delay to N\2 = predict supply and demand**
* **Vanishing gradient is 100 times worse.**
* **Gate networks like LSTM solves vanishing gradient.**

[**SNN**](https://medium.com/@eliorcohen/selu-make-fnns-great-again-snn-8d61526802a9) **- SELU activation function is inside not outside, results converge better.**

**Probably useful for feedforward networks**

[**DEEP REINFORCEMENT LEARNING COURSE**](https://www.youtube.com/watch?v=QDzM8r3WgBw\&t=2958s) **(for motion planning)or**\
[**DEEP RL COURSE**](https://www.youtube.com/watch?v=PtAIh9KSnjo) **(Q-LEARNING?) - using unlabeled data, reward, and probably a CNN to solve games beyond human level.**

**A** [**brief survey of DL for Reinforcement learning**](https://arxiv.org/abs/1708.05866)

[**WIKI**](https://en.wikipedia.org/wiki/Recurrent\_neural\_network#Long\_short-term\_memory) **has many types of RNN networks (unread)**

**Unread and potentially good tutorials:**

1. [**deep learning python**](https://www.datacamp.com/community/tutorials/deep-learning-python)

**EXAMPLES of Using NN on images:**

[**Deep image prior / denoiser/ high res/ remove artifacts/ etc..**](https://dmitryulyanov.github.io/deep\_image\_prior)

## **GRADIENT DESCENT**

**(**[**What are**](http://machinelearningmastery.com/gentle-introduction-mini-batch-gradient-descent-configure-batch-size/)**?) batch, stochastic, and mini-batch gradient descent are and the benefits and limitations of each method.**

[**What is gradient descent, how to use it, local minima okay to use, compared to global. Saddle points, learning rate strategies and research points**](https://blog.paperspace.com/intro-to-optimization-in-deep-learning-gradient-descent/)

1. **Gradient descent is an optimization algorithm often used for finding the weights or coefficients of machine learning algorithms, such as artificial neural networks and logistic regression.**
2. **the model makes predictions on training data, then use the error on the predictions to update the model to reduce the error.**
3. **The goal of the algorithm is to find model parameters (e.g. coefficients or weights) that minimize the error of the model on the training dataset. It does this by making changes to the model that move it along a gradient or slope of errors down toward a minimum error value. This gives the algorithm its name of “gradient descent.”**

### **Stochastic**

* **calculate error and updates the model after every training sample**

### **Batch**

* **calculates the error for each example in the training dataset, but only updates the model after all training examples have been evaluated.**

### **Mini batch (most common)**

* **splits the training dataset into small batches, used to calculate model error and update model coefficients.**
* **Implementations may choose to sum the gradient over the mini-batch or take the average of the gradient (reduces variance of gradient) (unclear?)**

**+ Tips on how to choose and train using mini batch in the link above**

[**Dont decay the learning rate, increase batchsize - paper**](https://arxiv.org/abs/1711.00489) **(optimization of a network)**

![](https://lh5.googleusercontent.com/3UX6uh\_X7IhUv9gwKopvsWRTICf9T2Xm8xWHTZuetYCUQiVRCP7mvIRxfns8Rmx3vuUFMXHiW5x8pVLWhNsUP9h1ZFzkFi9YUZRZjEuugZ3urEAAoRrMNt78hX6wIyIYvZAINiGw)

![](https://lh5.googleusercontent.com/u6LIUt6HFxzUbztSkBRv5R6Sk53OdmC9R5\_BsSkci96Lr0VVDqrx7VW3UTCkPqz0GX7P4NV4GwKxvaZEQ1XEkVDUTdGFnyA\_GU4rSPeFs601g7HPtUZzVfiTQWiCW5rv4d3JggDU)

* [**Big batches are not the cause for the ‘generalization gap’ between mini and big batches, it is not advisable to use large batches because of the low update rate, however if you change that, authors claim its okay**](https://arxiv.org/abs/1705.08741)**.**
* [**So what is a batch size in NN (another source)**](https://stats.stackexchange.com/questions/153531/what-is-batch-size-in-neural-network) **- and how to find the “right” number. In general terms a good mini bach between 1 and all samples is a good idea. Figure it out empirically.**
* **one epoch = one forward pass and one backward pass of all the training examples**
* **batch size = the number of training examples in one forward/backward pass. The higher the batch size, the more memory space you'll need.**
* **number of iterations = number of passes, each pass using \[batch size] number of examples. To be clear, one pass = one forward pass + one backward pass (we do not count the forward pass and backward pass as two different passes).**

**Example: if you have 1000 training examples, and your batch size is 500, then it will take 2 iterations to complete 1 epoch.**

* [**How to balance and what is the tradeoff between batch size and the number of iterations.**](https://stats.stackexchange.com/questions/164876/tradeoff-batch-size-vs-number-of-iterations-to-train-a-neural-network)

![](https://lh6.googleusercontent.com/pFXmWXcOcfu1WkWxG17RlPrLsRIh6Ve2cFU0pYD8S2V4cRThGzQV98n\_tRcLkeSqAweAZ30K9p7n1iViaVunIzHeVUHBkzdZSoIKf3Gta4OpxBOk6a4MStFoLQET89X84i9nXtSn)

[**GD with Momentum**](https://towardsdatascience.com/stochastic-gradient-descent-with-momentum-a84097641a5d) **- explain**

## **Batch size**

**(**[**a good read)**](https://machinelearningmastery.com/use-different-batch-sizes-training-predicting-python-keras/) **about batch sizes in keras, specifically LSTM, read this first!**

**A sequence prediction problem makes a good case for a varied batch size as you may want to have a batch size equal to the training dataset size (batch learning) during training and a batch size of 1 when making predictions for one-step outputs.**

**power of 2: have some advantages with regards to vectorized operations in certain packages, so if it's close it might be faster to keep your batch\_size in a power of 2.**

**(**[**pushing batches of samples to memory in order to train)**](https://stats.stackexchange.com/questions/153531/what-is-batch-size-in-neural-network) **-**

**Batch size defines number of samples that going to be propagated through the network.**

**For instance, let's say you have 1050 training samples and you want to set up batch\_size equal to 100. Algorithm takes first 100 samples (from 1st to 100th) from the training dataset and trains network. Next it takes second 100 samples (from 101st to 200th) and train network again. We can keep doing this procedure until we will propagate through the networks all samples. The problem usually happens with the last set of samples. In our example we've used 1050 which is not divisible by 100 without remainder. The simplest solution is just to get final 50 samples and train the network.**

**Advantages:**

* **It requires less memory. Since you train network using less number of samples the overall training procedure requires less memory. It's especially important in case if you are not able to fit dataset in memory.**
* **Typically networks trains faster with mini-batches. That's because we update weights after each propagation. In our example we've propagated 11 batches (10 of them had 100 samples and 1 had 50 samples) and after each of them we've updated network's parameters. If we used all samples during propagation we would make only 1 update for the network's parameter.**

**Disadvantages:**

* **The smaller the batch the less accurate estimate of the gradient. In the figure below you can see that mini-batch (green color) gradient's direction fluctuates compare to the full batch (blue color).**

![enter image description here](https://lh3.googleusercontent.com/In\_QJSs\_c5iIJCuUmnaPJZSjeOIu3HvqOldEtdryCh4TKTNwru6LjdVRq6A02IzwCBYxWNyesrVZn462HHXPfoZUZCOJjZh1cg2qz2tzJ93khr4hYc20vz-8goU9JRyqFI8GIFmp)

[**Small batch size has an effect on validation accuracy.**](http://forums.fast.ai/t/batch-size-effect-on-validation-accuracy/413)

![](https://lh6.googleusercontent.com/-eOGc8ZDsqSJWbu8J18jTRZUHxNuPbvBpvImJVK\_zsYsk4GNtC7u-I0puhNbgIg0LzDS\_v3-ySi519U8uWOyPv0qcvbLsaeHS3JaVt8jrjGygT2S608ON2d\_QPZ2guCuqvwPq0Wq)**IMPORTANT: batch size in ‘.prediction’ is needed for some models,** [**only for technical reasons as seen here**](https://github.com/fchollet/keras/issues/3027)**, in keras.**

1. **(**[**unread**](https://www.quora.com/Intuitively-how-does-mini-batch-size-affect-the-performance-of-stochastic-gradient-descent)**) about mini batches and performance.**
2. **(**[**unread**](https://stats.stackexchange.com/questions/164876/tradeoff-batch-size-vs-number-of-iterations-to-train-a-neural-network)**) tradeoff between bath size and number of iterations**

[**Another observation, probably empirical**](https://stackoverflow.com/questions/35050753/how-big-should-batch-size-and-number-of-epochs-be-when-fitting-a-model-in-keras) **- to answer your questions on Batch Size and Epochs:**

**In general: Larger batch sizes result in faster progress in training, but don't always converge as fast. Smaller batch sizes train slower, but can converge faster. It's definitely problem dependent.**

**In general, the models improve with more epochs of training, to a point. They'll start to plateau in accuracy as they converge. Try something like 50 and plot number of epochs (x axis) vs. accuracy (y axis). You'll see where it levels out.**

## **BIAS**

[**The role of bias in NN**](https://stackoverflow.com/questions/2480650/role-of-bias-in-neural-networks) **- similarly to the ‘b’ in linear regression.**

![](https://lh4.googleusercontent.com/J2OMsHkzsj\_c2GqMXdumCZkCNLWbSB2oRlodc9kXts2gko4L8Uf92t46HCG4C4nh5KJAvStQ-o3syY5jAiDTMNZM8fX98xEyaKPCtWtnR5sXKMAsALwVrlLeQzt8zkFVtR1bso3Z)

![](https://lh6.googleusercontent.com/MfRZSVTUDmh1sHI5lmQG1rgf9mDaF6X5EmqRCncUcq7zG24M457rg2OZwVBi33RH6ImIIJshLg3z1NJ7nw-YCwrwTXATOMYgXpCxh-CDA8awb9wXRvWBJlknfZV\_9klTROdNr99F)

## **BATCH NORMALIZATION**

1. **The** [**best explanation**](https://blog.paperspace.com/busting-the-myths-about-batch-normalization/) **to what is BN and why to use it, including busting the myth that it solves internal covariance shift - shifting input distribution, and saying that it should come after activations as it makes more sense (it does),also a nice quote on where a layer ends is really good - it can end at the activation (or not). How to use BN in the test, hint: use a moving window. Bn allows us to use 2 parameters to control the input distribution instead of controlling all the weights.**
2. [**Medium on BN**](https://towardsdatascience.com/an-alternative-to-batch-normalization-2cee9051e8bc)
3. [**Medium on BN**](https://towardsdatascience.com/batch-normalization-theory-and-how-to-use-it-with-tensorflow-1892ca0173ad)
4. [**Ian goodfellow on BN**](https://www.youtube.com/watch?v=Xogn6veSyxA\&feature=youtu.be\&t=325)
5. [**Medium #2 - a better one on BN, and adding to VGG**](https://towardsdatascience.com/batch-normalization-in-neural-networks-1ac91516821c)
6. [**Reddit on BN, mainly on the paper saying to use it before, but best practice is to use after**](https://www.reddit.com/r/MachineLearning/comments/67gonq/d\_batch\_normalization\_before\_or\_after\_relu/)
7. [**Diff between batch and norm (weak explanation)**](https://www.quora.com/What-are-the-practical-differences-between-batch-normalization-and-layer-normalization-in-deep-neural-networks)
8. [**Weight normalization for keras and TF**](http://krasserm.github.io/2018/11/10/weightnorm-implementation-options/)
9. [**Layer normalization keras**](https://pypi.org/project/keras-layer-normalization/)
10. [**Instance normalization keras**](https://github.com/keras-team/keras-contrib/blob/master/keras\_contrib/layers/normalization/instancenormalization.py)
11. [**batch/layer/instance in TF with code**](https://towardsdatascience.com/implementing-spatial-batch-instance-layer-normalization-in-tensorflow-manual-back-prop-in-tf-77faa8d2c362)
12. **Layer** [**norm for rnn’s or whatever name it is in this post**](https://twimlai.com/new-layer-normalization-technique-speeds-rnn-training/) **with** [**code**](https://gist.github.com/udibr/7f46e790c9e342d75dcbd9b1deb9d940) **for GRU**

[**What is the diff between batch/layer/recurrent batch and back rnn normalization**](https://datascience.stackexchange.com/questions/12956/paper-whats-the-difference-between-layer-normalization-recurrent-batch-normal)

* **Layer normalization (Ba 2016): Does not use batch statistics. Normalize using the statistics collected from all units within a layer of the current sample. Does not work well with ConvNets.**
* **Recurrent Batch Normalization (BN) (Cooijmans, 2016; also proposed concurrently by Qianli Liao & Tomaso Poggio, but tested on Recurrent ConvNets, instead of RNN/LSTM): Same as batch normalization. Use different normalization statistics for each time step. You need to store a set of mean and standard deviation for each time step.**
* **Batch Normalized Recurrent Neural Networks (Laurent, 2015): batch normalization is only applied between the input and hidden state, but not between hidden states. i.e., normalization is not applied over time.**
* **Streaming Normalization (Liao et al. 2016) : it summarizes existing normalizations and overcomes most issues mentioned above. It works well with ConvNets, recurrent learning and online learning (i.e., small mini-batch or one sample at a time):**
* **Weight Normalization (Salimans and Kingma 2016): whenever a weight is used, it is divided by its L2 norm first, such that the resulting weight has L2 norm 1. That is, output y=x∗(w/|w|), where x and w denote the input and weight respectively. A scalar scaling factor g is then multiplied to the output y=y∗g. But in my experience g seems not essential for performance (also downstream learnable layers can learn this anyway).**
* **Cosine Normalization (Luo et al. 2017): weight normalization is very similar to cosine normalization, where the same L2 normalization is applied to both weight and input: y=(x/|x|)∗(w/|w|). Again, manual or automatic differentiation can compute appropriate gradients of x and w.**
* **Note that both Weight and Cosine Normalization have been extensively used (called normalized dot product) in the 2000s in a class of ConvNets called HMAX (Riesenhuber 1999) to model biological vision. You may find them interesting.**

[**More about Batch/layer/instance/group norm are different methods for normalizing the inputs to the layers of deep neural networks**](https://nealjean.com/ml/neural-network-normalization/)

1. **Layer normalization solves the rnn case that batch couldnt - Is done per feature within the layer and normalized features are replaced**
2. **Instance does it for (cnn?) using per channel normalization**
3. **Group does it for group of channels**
4. ![](https://lh3.googleusercontent.com/P3AL20iV863GBbN\_D07g1PBh2T3nEVrR0CYd\_MXi5Gecozo-dc4CzbPemj5Bbyl4SbiZXtu-k8Q4hBXyh6c8SC8jOu4fU9B2G1vi0UT5nyGjDGAxURHqyre9NNmCnm5SVZpuHskF)

[**Part1: intuitive explanation to batch normalization**](http://mlexplained.com/2018/01/10/an-intuitive-explanation-of-why-batch-normalization-really-works-normalization-in-deep-learning-part-1/)

**Part2:** [**batch/layer/weight normalization**](http://mlexplained.com/2018/01/13/weight-normalization-and-layer-normalization-explained-normalization-in-deep-learning-part-2/) **- This is a good resource for advantages for every layer**

* **Layer, per feature in a batch,**
* **weight - divided by the norm**

![](https://lh3.googleusercontent.com/IqvjdZcCmsI-rAJ4ye0aUIoyrYLXLJTE2XMeRAAMIi0MxRoSzpRaZ6Op6dWgZ1VkjvBNUcuS8Xr0V9jo7jIpE46-7ktlS9QTDf6vmM8LI4N9juxa3CaLY4B5Gkl9oNPd44DjN5Bs)

## **HYPER PARAM GRID SEARCHES**

1. [**A disciplined approach to neural network hyper-parameters: Part 1 -- learning rate, batch size, momentum, and weight decay**](https://arxiv.org/abs/1803.09820)

## **LOSS**

[**Very Basic advice**](https://stats.stackexchange.com/questions/232754/reference-to-learn-how-to-interpret-learning-curves-of-deep-convolutional-neural)**: You should probably switch train/validation repartition to something like 80% training and 20% validation. In most cases it will improve the classifier performance overall (more training data = better performance)**

**+If Training error and test error are too close (your system is unable to overfit on your training data), this means that your model is too simple. Solution: more layers or more neurons per layer.**

**Early stopping**

**If you have never heard about "early-stopping" you should look it up, it's an important concept in the neural network domain :** [**https://en.wikipedia.org/wiki/Early\_stopping**](https://en.wikipedia.org/wiki/Early\_stopping) **. To summarize, the idea behind early-stopping is to stop the training once the validation loss starts plateauing. Indeed, when this happens it almost always mean you are starting to overfitt your classifier. The training loss value in itself is not something you should trust, beacause it will continue to increase event when you are overfitting your classifier.**

**With** [**cross entropy**](https://www.quora.com/Loss-cross-entropy-is-decreasing-but-accuracy-remains-the-same-while-training-convolutional-neural-networks-How-can-it-happen) **there can be an issue where the accuracy is the same for two cases, one where the loss is decreasing and the other when the loss is not changing much.**

![](https://lh3.googleusercontent.com/f2R8DVu5A9g6LOGbNcmyIfayuVBYnpScO\_kNsAcuJ8lsiM-hnYwlqD04qyI1wPYTwmsr2KpFKJa19gMkkJd67y03iJquhRftQdBpfGEdw5OQHficHqgkxudLfgpZsSS7Cc2p9qDS)

[**How to read LOSS graphs (and accuracy on top)**](https://github.com/fchollet/keras/issues/3755)

![](https://lh6.googleusercontent.com/blj3natUcvqK-nEmNjv90zAIM74QbA4x7hQ\_F\_oPGcHxQcdhc0\_NrcPZhWDne2EEnUnJKNDOw4Xt\_cUkhv3cFTFMcqzzBT4NeOPPnmoTfTXLFrEnVwkrlc5PEsZDNCZXdOr0GRZj) ![](https://lh4.googleusercontent.com/o39Jcw1o7JeSsKuD\_q-9xGukmT6pWLGs-9sVIumxLRF7dPpf25w8o9e2OBnWbpPc\_p6t9e03D46r34N-8CYZa6fvfcWBVp\_7N06xE0kbrvIzBC5sGWcMymN\_KtPTfRKwHk1-gRcQ)

**This indicates that the model is overfitting. It continues to get better and better at fitting the data that it sees (training data) while getting worse and worse at fitting the data that it does not see (validation data).**

[**This is a very good example of a train/test loss and an accuracy behavior.**](https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/)

![](https://lh6.googleusercontent.com/GK\_rvndJY76-cgBbBetgSZfwTD7RTZW2UsXUtsEZRUvFW1ACpJw9FMhNwj3LBERvmmPvcuTkkwb5HUcXgi7ua42WqJwAZgFP-3NsyF1qEo9GmACXGQGWGSYh3AR7yY765Qm9QfiO) ![](https://lh4.googleusercontent.com/Q46fiZLm9mMhuQnOVJjyZWstXj6Aq1Ctev1cvIUsdrOWiOqxfvNlkJjcW08waf8qCERvvt1AkW-HjDrLvjHiVxKTFzxfX0BmVq4hRUERqrGsNLALeJb75Geb06X21Bgb8z2dA6iw)

[**Cross entropy formula with soft labels (probability) rather than classes.**](https://stats.stackexchange.com/questions/206925/is-it-okay-to-use-cross-entropy-loss-function-with-soft-labels)

[**Mastery on cross entropy, brier, roc auc, how to ‘game’ them and calibrate them**](https://machinelearningmastery.com/how-to-score-probability-predictions-in-python/)

[**Game changer paper - a general adaptive loss search in nn**](https://www.reddit.com/r/computervision/comments/bsd82j/a\_general\_and\_adaptive\_robust\_loss\_function/?utm\_medium=android\_app\&utm\_source=share)

## **LEARNING RATE REDUCTION**

[**Intro to Learning Rate methods**](https://medium.com/@chengweizhang2012/quick-notes-on-how-to-choose-optimizer-in-keras-9d3d12d09039) **- what they are doing and what they are fixing in other algos.**

[**Callbacks**](https://keras.io/callbacks/)**, especially ReduceLROnPlateau - this callback monitors a quantity and if no improvement is seen for a 'patience' number of epochs, the learning rate is reduced.**

[**Cs123**](http://cs231n.github.io/neural-networks-3/) **(very good): explains about many things related to CNN, but also about LR and adaptive methods.**

[**An excellent comparison of several learning rate schedule methods and adaptive methods:**](https://medium.com/towards-data-science/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1) **(**[**same here but not as good**](https://machinelearningmastery.com/using-learning-rate-schedules-deep-learning-models-python-keras/)**)**

![](https://lh5.googleusercontent.com/UtrDKeqV\_UfuPuot937svdmi-fzHp3K\_eRS5xFAgQI7CAXPFchkFCQO4YPYOFkWMG6tYDlAeATR0YUwOLKqLlDq17T-Row\_iBknUXchk9zT2\_0KBzE7BMipHBKPds-sFw\_0NDAjF)

**Adaptive gradient descent algorithms such as** [**Adagrad**](https://en.wikipedia.org/wiki/Stochastic\_gradient\_descent#AdaGrad)**, Adadelta,** [**RMSprop**](https://en.wikipedia.org/wiki/Stochastic\_gradient\_descent#RMSProp)**,** [**Adam**](https://en.wikipedia.org/wiki/Stochastic\_gradient\_descent#Adam)**, provide an alternative to classical SGD.**

**These per-parameter learning rate methods provide heuristic approach without requiring expensive work in tuning hyperparameters for the learning rate schedule manually.**

1. **Adagrad performs larger updates for more sparse parameters and smaller updates for less sparse parameter. It has good performance with sparse data and training large-scale neural network. However, its monotonic learning rate usually proves too aggressive and stops learning too early when training deep neural networks.**
2. **Adadelta is an extension of Adagrad that seeks to reduce its aggressive, monotonically decreasing learning rate.**
3. **RMSprop adjusts the Adagrad method in a very simple way in an attempt to reduce its aggressive, monotonically decreasing learning rate.**
4. [**Adam**](https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/) **is an update to the RMSProp optimizer which is like RMSprop with momentum.**

![](https://lh6.googleusercontent.com/ixb189Iy\_Z4PuSCZHn48vmBvRDNchESvmANzapkuTNMt5zYp7vl9NLznUzNQYaMuyUQzhLiQgpCPUho9klBdd4W09dcjsdx8D\_yIDvOcOK8Jo2\_p6nDMmLv3QL5ohm07-pJmIo48)

**adaptive learning rate methods demonstrate better performance than learning rate schedules, and they require much less effort in hyperparamater settings**

![](https://lh3.googleusercontent.com/rYknk8vLbQKYuLSKeItX59a6rdi84U5QaeNJoardmv\_jLgXqIMHj1BGbZsMh4l0Pli-mKYg29dNGDMKHS341t94fUScWELjPsIXWy7i1-\_zXiCOSR1J46gMODzPQrrX4x64P1ato)

[**Recommended paper**](https://arxiv.org/pdf/1206.5533v2.pdf)**: practical recommendation for gradient based DNN**

**Another great comparison -** [**pdf paper**](https://arxiv.org/abs/1609.04747) **and** [**webpage link**](http://ruder.io/optimizing-gradient-descent/) **-**

* **if your input data is sparse, then you likely achieve the best results using one of the adaptive learning-rate methods.**
* **An additional benefit is that you will not need to tune the learning rate but will likely achieve the best results with the default value.**
* **In summary, RMSprop is an extension of Adagrad that deals with its radically diminishing learning rates. It is identical to Adadelta, except that Adadelta uses the RMS of parameter updates in the numerator update rule. Adam, finally, adds bias-correction and momentum to RMSprop. Insofar, RMSprop, Adadelta, and Adam are very similar algorithms that do well in similar circumstances. Kingma et al. \[10] show that its bias-correction helps Adam slightly outperform RMSprop towards the end of optimization as gradients become sparser. Insofar, Adam might be the best overall choice**

## **TRAIN / VAL accuracy in NN**

**The second important quantity to track while training a classifier is the validation/training accuracy. This plot can give you valuable insights into the amount of overfitting in your model:**

![](https://lh5.googleusercontent.com/K8KuSlFCGaOO9qihQGVQf3Cckcy5A2V98Tt\_OKbscmv-ZmmemEVJFs2V9eeydc8Aa\_dk-TXXjsJhiPCD7UAqKcvaMc4xsP0RIJNl0EiZ7ybQ5HsrINup7AYJjSfayQELeOA3WS\_-)

* **The gap between the training and validation accuracy indicates the amount of overfitting.**
* **Two possible cases are shown in the diagram on the left. The blue validation error curve shows very small validation accuracy compared to the training accuracy, indicating strong overfitting (note, it's possible for the validation accuracy to even start to go down after some point).**
* **NOTE: When you see this in practice you probably want to increase regularization:**
  * **stronger L2 weight penalty**
  * **Dropout**
  * **collect more data.**
* **The other possible case is when the validation accuracy tracks the training accuracy fairly well. This case indicates that your model capacity is not high enough: make the model larger by increasing the number of parameters.**

## **INITIALIZERS**

**XAVIER GLOROT:**

[**Why’s Xavier initialization important?**](http://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization)

**In short, it helps signals reach deep into the network.**

* **If the weights in a network start too small, then the signal shrinks as it passes through each layer until it’s too tiny to be useful.**
* **If the weights in a network start too large, then the signal grows as it passes through each layer until it’s too massive to be useful.**

**Xavier initialization makes sure the weights are ‘just right’, keeping the signal in a reasonable range of values through many layers.**

**To go any further than this, you’re going to need a small amount of statistics - specifically you need to know about random distributions and their variance.**

[**When to use glorot uniform-over-normal initialization?**](https://datascience.stackexchange.com/questions/13061/when-to-use-he-or-glorot-normal-initialization-over-uniform-init-and-what-are)

**However, i am still not seeing anything empirical that says that glorot surpesses everything else under certain conditions (**[**except the glorot paper**](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)**), most importantly, does it really help in LSTM where the vanishing gradient is \~no longer an issue?**

[**He-et-al Initialization**](https://towardsdatascience.com/random-initialization-for-neural-networks-a-thing-of-the-past-bfcdd806bf9e)

**This method of initializing became famous through a paper submitted in 2015 by He et al, and is similar to Xavier initialization, with the factor multiplied by two. In this method, the weights are initialized keeping in mind the size of the previous layer which helps in attaining a global minimum of the cost function faster and more efficiently.**

**w=np.random.randn(layer\_size\[l],layer\_size\[l-1])\*np.sqrt(2/layer\_size\[l-1])**

## **ACTIVATION FUNCTIONS**

1. [**a bunch of observations, seems like a personal list**](http://sentiment-mining.blogspot.co.il/2015/08/the-difference-of-activation-function.html) **-**
   1. **Output layer - linear for regression, softmax for classification**
   2. **Hidden layers - hyperbolic tangent for shallow networks (less than 3 hidden layers), and ReLU for deep networks**
2. **ReLU - The purpose of ReLU is to introduce non-linearity, since most of the real-world data we would want our network to learn would be nonlinear (e.g. convolution is a linear operation – element wise matrix multiplication and addition, so we account for nonlinearity by introducing a nonlinear function like ReLU, e.g** [**here**](https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/) **- search for ReLU).**
   1. **Relu is quite resistant to vanishing gradient & allows for deactivating neurons and for sparsity.**
   2. **Other nonlinear functions such as tanh or sigmoid can also be used instead of ReLU, but ReLU has been found to perform better in most situations.**
3. [**Visual + description of activation functions**](https://medium.com/the-theory-of-everything/understanding-activation-functions-in-neural-networks-9491262884e0)
4. [**A very good explanation + figures about activations functions**](https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6)
5. [**Selu**](https://towardsdatascience.com/selu-make-fnns-great-again-snn-8d61526802a9) **- better than RELU? Possibly.**
6. [**Mish**](https://github.com/digantamisra98/Mish)**: A Self Regularized Non-Monotonic Neural Activation Function,** [**yam peleg’s code**](https://gist.github.com/ypeleg/3af35d07d7f659f387952c9843849772?fbclid=IwAR2x\_Hzlg79\_mo\_zQMJGFbQWORbpdydnllnHoA\_RmUlCLpqKdGwClBuJy8g)
7. [**Mish, Medium, Keras Code, with benchmarks, computationally expensive.**](https://towardsdatascience.com/mish-8283934a72df)
8. [Gelu](https://paperswithcode.com/method/gelu) (Used by OpenAI
9. [Deep Learning 101: Transformer Activation Functions Explainer - Sigmoid, ReLU, GELU, Swish](https://www.saltdatalabs.com/blog/deep-learning-101-transformer-activation-functions-explainer-relu-leaky-relu-gelu-elu-selu-softmax-and-more)

## **OPTIMIZERS**

**There are several optimizers, each had his 15 minutes of fame, some optimizers are recommended for CNN, Time Series, etc..**

**There are also what I call ‘experimental’ optimizers, it seems like these pop every now and then, with or without a formal proof. It is recommended to follow the literature and see what are the ‘supposedly’ state of the art optimizers atm.**

[**Adamod**](https://medium.com/@lessw/meet-adamod-a-new-deep-learning-optimizer-with-memory-f01e831b80bd) **deeplearning optimizer with memory**

[**Backstitch**](http://www.danielpovey.com/files/2017\_nips\_backstitch.pdf) **- September 17 - supposedly an improvement over SGD for speech recognition using DNN. Note: it wasnt tested with other datasets or other network types.**

**(how does it work?) take a negative step back, then a positive step forward. I.e., When processing a minibatch, instead of taking a single SGD step, we first take a step with −α times the current learning rate, for α > 0 (e.g. α = 0.3), and then a step with 1 + α times the learning rate, with the same minibatch (and a recomputed gradient). So we are taking a small negative step, and then a larger positive step. This resulted in quite large improvements – around 10% relative improvement \[37] – for our best speech recognition DNNs. The recommended hyper parameters are in the paper.**

**Drawbacks: takes twice to train, momentum not implemented or tested, dropout is mandatory for improvement, slow starter.**

[**Documentation about optimizers**](https://keras.io/optimizers/) **in keras**

* **SGD can be fine tuned**
* **For others Leave most parameters as they were**

[**Best description on optimizers with momentum etc, from sgd to nadam, formulas and intuition**](https://towardsdatascience.com/10-gradient-descent-optimisation-algorithms-86989510b5e9)

![](https://lh6.googleusercontent.com/-quQMukoMffONyGh-R-nuGssirsDgFz6YQyZAjQ22FyQFglTbpnN0kA7VNQ3UH\_o2DSus3SJs2ThnwMS0rnH3iIZN1cK8OzKb39oBj4c2lU-dE9k3c\_MDuiMr51IeghvAHLZh2t9)

## **DROPOUT LAYERS IN KERAS AND GENERAL**

[**A very influential paper about dropout and how beneficial it is - bottom line always use it.**](http://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf)

**OPEN QUESTIONs:**

1. **does a dropout layer improve performance even if an lstm layer has dropout or recurrent dropout.**
2. **What is the diff between a separate layer and inside the lstm layer.**
3. **What is the diff in practice and intuitively between drop and recurrentdrop**

[**Dropout layers in keras, or dropout regularization:**](https://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/)

* **Dropout is a technique where randomly selected neurons are ignored RANDOMLY during training.**
* **contribution to the activation of downstream neurons is temporally removed on the forward pass and any weight updates are not applied to the neuron on the backward pass.**
* **As a neural network learns, neuron weights settle into their context within the network.**
* **Weights of neurons are tuned for specific features providing some specialization. Neighboring neurons become to rely on this specialization, which if taken too far can result in a fragile model too specialized to the training data. (overfitting)**
* **This reliant on context for a neuron during training is referred to complex co-adaptations.**
* **After dropout, other neurons will have to step in and handle the representation required to make predictions for the missing neurons, which is believed to result in multiple independent internal representations being learned by the network.**
* **Thus, the effect of dropout is that the network becomes less sensitive to the specific weights of neurons.**
* **This in turn leads to a network with better generalization capability and less likely to overfit the training data.**

[**Another great answer about drop out**](https://www.quora.com/In-Keras-what-is-a-dense-and-a-dropout-layer) **-**

* **as a consequence of the 50% dropout, the neural network will learn different, redundant representations; the network can’t rely on the particular neurons and the combination (or interaction) of these to be present.**
* **Another nice side effect is that training will be faster.**
* **Rules:**
  * **Dropout is only applied during training,**
  * **Need to rescale the remaining neuron activations. E.g., if you set 50% of the activations in a given layer to zero, you need to scale up the remaining ones by a factor of 2.**
  * **if the training has finished, you’d use the complete network for testing (or in other words, you set the dropout probability to 0).**

[**Implementation of drop out in keras**](https://datascience.stackexchange.com/questions/18088/convolutional-layer-dropout-layer-in-keras/18098) **is “inverse dropout” - n the Keras implementation, the output values are corrected during training (by dividing, in addition to randomly dropping out the values) instead of during testing (by multiplying). This is called "inverted dropout".**

**Inverted dropout is functionally equivalent to original dropout (as per your link to Srivastava's paper), with a nice feature that the network does not use dropout layers at all during test and prediction. This is explained a little in this** [**Keras issue**](https://github.com/fchollet/keras/issues/3305)**.**

[**Dropout notes and rules of thumb aka “best practice” -**](http://blog.mrtanke.com/2016/10/09/Keras-Study-Notes-3-Dropout-Regularization-for-Deep-Networks/)

* **dropout value of 20%-50% of neurons with 20% providing a good starting point. (A probability too low has minimal effect and a value too high results in underlearning by the network.)**
* **Use a large network for better performance, i.e., when dropout is used on a larger network, giving the model more of an opportunity to learn independent representations.**
* **Use dropout on VISIBLE AND HIDDEN. Application of dropout at each layer of the network has shown good results.**
* **Unclear ? Use a large learning rate with decay and a large momentum. Increase your learning rate by a factor of 10 to 100 and use a high momentum value of 0.9 or 0.99.**
* **Unclear ? Constrain the size of network weights. A large learning rate can result in very large network weights. Imposing a constraint on the size of network weights such as max-norm regularization with a size of 4 or 5 has been shown to improve results.**

[**Difference between LSTM ‘dropout’ and ‘recurrent\_dropout’**](https://stackoverflow.com/questions/44924690/keras-the-difference-between-lstm-dropout-and-lstm-recurrent-dropout) **- vertical vs horizontal.**

**I suggest taking a look at (the first part of)** [**this paper**](https://arxiv.org/pdf/1512.05287.pdf)**. Regular dropout is applied on the inputs and/or the outputs, meaning the vertical arrows from x\_t and to h\_t. In you add it as an argument to your layer, it will mask the inputs; you can add a Dropout layer after your recurrent layer to mask the outputs as well. Recurrent dropout masks (or "drops") the connections between the recurrent units; that would be the horizontal arrows in your picture.**

**This picture is taken from the paper above. On the left, regular dropout on inputs and outputs. On the right, regular dropout PLUS recurrent dropout:**

![This picture is taken from the paper above. On the left, regular dropout on inputs and outputs. On the right, regular dropout PLUS recurrent dropout.](https://lh3.googleusercontent.com/RF9eawLdYCty8TSrEBsd3NvaxpFbQNG9s551Q-sX1OVlsC3MRZZ1q5s-xYZVv81Z\_-3SvK4JwtAwUirZuCE8MPIISw0ebchNTqY3IMEpc76jalJG-0oeRpDGrWMTnYtAELhs0c3-)

## **NEURAL NETWORK OPTIMIZATION TECHNIQUES**

**Basically do these after you have a working network**

1. [**Dont decay the learning rate, increase batchsize - paper**](https://arxiv.org/abs/1711.00489) **(optimization of a network)**
2. [**Add one neuron with skip connection, or to every layer in a binary classification network to get global minimum**](https://arxiv.org/abs/1805.08671)**.**
3. \*\*\*\*[**RESNET, DENSENET UNET**](https://medium.com/swlh/resnets-densenets-unets-6bbdbcfdf010) **- the trick behind them, concatenating both f(x) = x**
4.  \*\*\*\*[**skip connections**](https://www.analyticsvidhya.com/blog/2021/08/all-you-need-to-know-about-skip-connections/) \*\*\*\* by Siravam / Vidhya- \*\*"\*\*Skip Connections (or Shortcut Connections) as the name suggests skips some of the layers in the neural network and feeds the output of one layer as the input to the next layers.

    Skip Connections were introduced to solve different problems in different architectures. In the case of ResNets, skip connections solved the _degradation problem_ that we addressed earlier whereas, in the case of DenseNets, it ensured **feature reusability**. We’ll discuss them in detail in the following sections.

    Skip connections were introduced in literature even before residual networks. For example, [**Highway Networks**](https://arxiv.org/abs/1505.00387) (Srivastava et al.) had skip connections with gates that controlled and learned the flow of information to deeper layers. This concept is similar to the gating mechanism in LSTM. Although ResNets is actually a special case of Highway networks, the performance isn’t up to the mark comparing to ResNets. This suggests that it’s better to keep the gradient highways clear than to go for any gates – simplicity wins here!"

## **Fine tuning**

1. [**3 methods to fine tune, cut softmax layer, smaller learning rate, freeze layers**](https://flyyufelix.github.io/2016/10/03/fine-tuning-in-keras-part1.html)
2. [**Fine tuning on a sunset of data**](https://stats.stackexchange.com/questions/289036/fine-tuning-with-a-subset-of-the-same-data)

## **Deep Learning for NLP**

* **(did not fully read)** [**Yoav Goldberg’s course**](https://docs.google.com/document/d/1Xf\_dqjf7mWmSoYX0HTKnml2mssP5BjrKUs-4E17CbNo/edit) **syllabus with lots of relevant topics on DL4NLP, including bidirectional RNNS and tree RNNs.**
* **(did not fully read)** [**CS224d**](http://cs224d.stanford.edu/index.html)**: Deep Learning for Natural Language Processing, with** [**slides etc.**](http://cs224d.stanford.edu/syllabus.html)

[**Deep Learning using Linear Support Vector Machines**](http://deeplearning.net/wp-content/uploads/2013/03/dlsvm.pdf) **- 1-3% decrease in error by replacing the softmax layer with a linear support vector machine**

## **MULTI LABEL/OUTPUT**

1. **A machine learning framework for** [**multi-output/multi-label**](https://github.com/scikit-multiflow/scikit-multiflow) **and stream data. Inspired by MOA and MEKA, following scikit-learn's philosophy.** [**https://scikit-multiflow.github.io/**](https://scikit-multiflow.github.io/)
2. [**Medium on MO, sklearn and keras**](https://towardsdatascience.com/what-data-scientists-should-know-about-multi-output-and-multi-label-training-b9d4be620e11)
3. [**MO in keras, see functional API on how.**](https://www.pyimagesearch.com/2018/06/04/keras-multiple-outputs-and-multiple-losses/)

### **FUZZY MULTI LABEL**

1. [**Ie., probabilities or soft values instead of hard labels**](https://datascience.stackexchange.com/questions/48111/multilabel-classifcation-in-sklearn-with-soft-fuzzy-labels)

## **SIAMESE NETWORKS**

1. [**Siamese for conveyor belt fault prediction**](https://towardsdatascience.com/predictive-maintenance-with-lstm-siamese-network-51ee7df29767)
2. [**Burlow**](https://arxiv.org/abs/2103.03230)**,** [**fb post**](https://www.facebook.com/yann.lecun/posts/10157682573642143) **- Self-supervised learning (SSL) is rapidly closing the gap with supervised methods on large computer vision benchmarks. A successful approach to SSL is to learn representations which are invariant to distortions of the input sample. However, a recurring issue with this approach is the existence of trivial constant solutions. Most current methods avoid such solutions by careful implementation details. We propose an objective function that naturally avoids such collapse by measuring the cross-correlation matrix between the outputs of two identical networks fed with distorted versions of a sample, and making it as close to the identity matrix as possible. This causes the representation vectors of distorted versions of a sample to be similar, while minimizing the redundancy between the components of these vectors.**

## **Gated Multi-Layer Perceptron (GMLP)**

1. \*\*\*\*[**paper**](https://arxiv.org/abs/2105.08050)**,** [**git1**](https://github.com/jaketae/g-mlp)**,** [**git2**](https://github.com/lucidrains/g-mlp-pytorch) \*\*- "\*\*a simple network architecture, gMLP, based on MLPs with gating, and show that it can perform as well as Transformers in key language and vision applications. Our comparisons show that self-attention is not critical for Vision Transformers, as gMLP can achieve the same accuracy."

![](<../.gitbook/assets/image (14).png>)
