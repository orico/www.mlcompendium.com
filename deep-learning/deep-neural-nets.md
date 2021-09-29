# Deep Neural Nets

* [**Deep learning notes from Andrew NG’s course.**](https://www.slideshare.net/TessFerrandez/notes-from-coursera-deep-learning-courses-by-andrew-ng)
* **Jay Alammar on NN** [**Part 1**](http://jalammar.github.io/visual-interactive-guide-basics-neural-networks/)**,** [**Part 2**](http://jalammar.github.io/feedforward-neural-networks-visual-interactive/)
* [**NN in general**](http://briandolhansky.com/blog/?tag=neural+network#show-archive) **- 5 introductions  tutorials.**
* [**Segmentation examples**](https://meetshah1995.github.io/semantic-segmentation/deep-learning/pytorch/visdom/2017/06/01/semantic-segmentation-over-the-years.html)

**MLP: fully connected, input, hidden layers, output. Gradient on the backprop takes a lot of time to calculate. Has vanishing gradient problem, because of multiplications when it reaches the first layers the loss correction is very small \(0.1\*0.1\*01 = 0.001\), therefore the early layers train slower than the last ones, and the early ones capture the basics structures so they are the more important ones.**

**AutoEncoder - unsupervised, drives the input through fully connected layers, sometime reducing their neurons amount, then does the reverse and expands the layer’s size to get to the input \(images are multiplied by the transpose matrix, many times over\), Comparing the predicted output to the input, correcting the cost using gradient descent and redoing it, until the networks learns the output.**

* **Convolutional auto encoder**
* **Denoiser auto encoder - masking areas in order to create an encoder that understands noisy images**
* **Variational autoencoder - doesnt rely on distance between pixels, rather it maps them to a function \(gaussian\), eventually the DS should be explained by this mapping, uses 2 new layers added to the network. Gaussian will create blurry images, but similar. Please note that it also works with CNN.**

**What are** [**logits**](https://stackoverflow.com/questions/41455101/what-is-the-meaning-of-the-word-logits-in-tensorflow) **in neural net - the vector of raw \(non-normalized\) predictions that a classification model generates, which is ordinarily then passed to a normalization function. If the model is solving a multi-class classification problem, logits typically become an input to the softmax function. The softmax function then generates a vector of \(normalized\) probabilities with one value for each possible class.**

[**WORD2VEC**](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/) **- based on autoencode, we keep only the hidden layer ,** [**Part 2**](http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/)

**RBM- restricted \(no 2 nodes share a connection\) boltzman machine**

**An Autoencoder of features, tries to encode its own structure.**

**Works best on pics, video, voice, sensor data. 2 layers, visible and hidden, error and bias calculated via KL Divergence.**

* **Also known as a shallow network.**
* **Two layers, input and output, goes back and forth until it learns its output.**

**DBN - deep belief networks, similar structure to multi layer perceptron. fully connected, input, hidden\(s\), output layers. Can be thought of as stacks of RBM. training using GPU optimization, accurate and needs smaller labelled data set to complete the training.**

**Solves the ‘vanishing gradient’ problem, imagine a fully connected network, advancing each 2 layers step by step until each boltzman network \(2 layers\) learns the output, keeps advancing until finished.. Each layer learns the entire input.**

**Next step is to fine tune using a labelled test set, improves performance and alters the net. So basically using labeled samples we fine tune and associate features and pattern with a name. Weights and biases are altered slightly and there is also an increase in performance. Unlike CNN which learns features then high level features.**

**Accurate and reasonable in time, unlike fully connected that has the vanishing gradient problem.**

**Transfer Learning = like Inception in Tensor flow, use a prebuilt network to solve many problems that “work” similarly to the original network.**

* [**CS course definition**](http://cs231n.github.io/transfer-learning/) **- also very good explanation of the common use cases:**
  * **Feature extraction from the CNN part \(removing the fully connected layer\)**
  * **Fine-tuning, everything or partial selection of the hidden layers, mainly good to keep low level neurons that know what edges and color blobs are, but not dog breeds or something not as general.**
* [**CNN checkpoints**](https://github.com/BVLC/caffe/wiki/Model-Zoo#cascaded-fully-convolutional-networks-for-biomedical-image-segmentation) **for many problems with transfer learning. Has several relevant references**
* **Such as this “**[**How transferable are features in deep neural networks?**](http://arxiv.org/abs/1411.1792) **“**
* **\(the indian guy on facebook\)** [**IMDB transfer learning using cnn vgg and word2vec**](https://spandan-madan.github.io/DeepLearningProject/)**, the word2vec is interesting, the cnn part is very informative. With python code, keras.**

**CNN, Convolutional Neural Net \(**[**this link explains CNN quite well**](https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/)**,** [**2nd tutorial**](https://hackernoon.com/deep-learning-cnns-in-tensorflow-with-gpus-cba6efe0acc2) **- both explain about convolution, padding, relu - sparsity, max and avg pooling\):**

* **Common Layers: input-&gt;convolution-&gt;relu activation-&gt;pooling to reduce dimensionality \*\*\*\* -&gt;fully connected layer**
* **\*\*\*\*repeat several times over as this discover patterns but needs another layer -&gt; fully connected layer**
* **Then we connect at the end a fully connected layer \(fcl\) to classify data samples.**
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

[**DEEP REINFORCEMENT LEARNING COURSE**](https://www.youtube.com/watch?v=QDzM8r3WgBw&t=2958s) **\(for motion planning\)or**  
[**DEEP RL COURSE**](https://www.youtube.com/watch?v=PtAIh9KSnjo) **\(Q-LEARNING?\) - using unlabeled data, reward, and probably a CNN to solve games beyond human level.**

**A** [**brief survey of DL for Reinforcement learning**](https://arxiv.org/abs/1708.05866)

[**WIKI**](https://en.wikipedia.org/wiki/Recurrent_neural_network#Long_short-term_memory) **has many types of RNN networks \(unread\)**

**Unread and potentially good tutorials:**

1. [**deep learning python**](https://www.datacamp.com/community/tutorials/deep-learning-python)

**EXAMPLES of Using NN on images:**

[**Deep image prior / denoiser/ high res/ remove artifacts/ etc..**](https://dmitryulyanov.github.io/deep_image_prior)

## **GRADIENT DESCENT**

**\(**[**What are**](http://machinelearningmastery.com/gentle-introduction-mini-batch-gradient-descent-configure-batch-size/)**?\) batch, stochastic, and mini-batch gradient descent are and the benefits and limitations of each method.**

[**What is gradient descent, how to use it, local minima okay to use, compared to global. Saddle points, learning rate strategies and research points**](https://blog.paperspace.com/intro-to-optimization-in-deep-learning-gradient-descent/)

1. **Gradient descent is an optimization algorithm often used for finding the weights or coefficients of machine learning algorithms, such as artificial neural networks and logistic regression.**
2. **the model makes predictions on training data, then use the error on the predictions to update the model to reduce the error.**
3. **The goal of the algorithm is to find model parameters \(e.g. coefficients or weights\) that minimize the error of the model on the training dataset. It does this by making changes to the model that move it along a gradient or slope of errors down toward a minimum error value. This gives the algorithm its name of “gradient descent.”**

### **Stochastic**

* **calculate error and updates the model after every training sample**

### **Batch**

* **calculates the error for each example in the training dataset, but only updates the model after all training examples have been evaluated.**

### **Mini batch \(most common\)**

* **splits the training dataset into small batches, used to calculate model error and update model coefficients.** 
* **Implementations may choose to sum the gradient over the mini-batch or take the average of the gradient \(reduces variance of gradient\) \(unclear?\)**

**+ Tips on how to choose and train using mini batch in the link above**

[**Dont decay the learning rate, increase batchsize - paper**](https://arxiv.org/abs/1711.00489) **\(optimization of a network\)**

![](https://lh5.googleusercontent.com/3UX6uh_X7IhUv9gwKopvsWRTICf9T2Xm8xWHTZuetYCUQiVRCP7mvIRxfns8Rmx3vuUFMXHiW5x8pVLWhNsUP9h1ZFzkFi9YUZRZjEuugZ3urEAAoRrMNt78hX6wIyIYvZAINiGw)

![](https://lh5.googleusercontent.com/u6LIUt6HFxzUbztSkBRv5R6Sk53OdmC9R5_BsSkci96Lr0VVDqrx7VW3UTCkPqz0GX7P4NV4GwKxvaZEQ1XEkVDUTdGFnyA_GU4rSPeFs601g7HPtUZzVfiTQWiCW5rv4d3JggDU)

* [**Big batches are not the cause for the ‘generalization gap’ between mini and big batches, it is not advisable to use large batches because of the low update rate, however if you change that, authors claim its okay**](https://arxiv.org/abs/1705.08741)**.**
* [**So what is a batch size in NN \(another source\)**](https://stats.stackexchange.com/questions/153531/what-is-batch-size-in-neural-network) **- and how to find the “right” number. In general terms a good mini bach between 1 and all samples is a good idea. Figure it out empirically.** 
* **one epoch = one forward pass and one backward pass of all the training examples**
* **batch size = the number of training examples in one forward/backward pass. The higher the batch size, the more memory space you'll need.**
* **number of iterations = number of passes, each pass using \[batch size\] number of examples. To be clear, one pass = one forward pass + one backward pass \(we do not count the forward pass and backward pass as two different passes\).**

**Example: if you have 1000 training examples, and your batch size is 500, then it will take 2 iterations to complete 1 epoch.**

* [**How to balance and what is the tradeoff between batch size and the number of iterations.**](https://stats.stackexchange.com/questions/164876/tradeoff-batch-size-vs-number-of-iterations-to-train-a-neural-network)

![](https://lh6.googleusercontent.com/pFXmWXcOcfu1WkWxG17RlPrLsRIh6Ve2cFU0pYD8S2V4cRThGzQV98n_tRcLkeSqAweAZ30K9p7n1iViaVunIzHeVUHBkzdZSoIKf3Gta4OpxBOk6a4MStFoLQET89X84i9nXtSn)

[**GD with Momentum**](https://towardsdatascience.com/stochastic-gradient-descent-with-momentum-a84097641a5d) **- explain**

## **Batch size**

**\(**[**a good read\)**](https://machinelearningmastery.com/use-different-batch-sizes-training-predicting-python-keras/) **about batch sizes in keras, specifically LSTM, read this first!**

**A sequence prediction problem makes a good case for a varied batch size as you may want to have a batch size equal to the training dataset size \(batch learning\) during training and a batch size of 1 when making predictions for one-step outputs.**

**power of 2: have some advantages with regards to vectorized operations in certain packages, so if it's close it might be faster to keep your batch\_size in a power of 2.**

**\(**[**pushing batches of samples to memory in order to train\)**](https://stats.stackexchange.com/questions/153531/what-is-batch-size-in-neural-network) **-**

**Batch size defines number of samples that going to be propagated through the network.**

**For instance, let's say you have 1050 training samples and you want to set up batch\_size equal to 100. Algorithm takes first 100 samples \(from 1st to 100th\) from the training dataset and trains network. Next it takes second 100 samples \(from 101st to 200th\) and train network again. We can keep doing this procedure until we will propagate through the networks all samples. The problem usually happens with the last set of samples. In our example we've used 1050 which is not divisible by 100 without remainder. The simplest solution is just to get final 50 samples and train the network.**

**Advantages:**

* **It requires less memory. Since you train network using less number of samples the overall training procedure requires less memory. It's especially important in case if you are not able to fit dataset in memory.**
* **Typically networks trains faster with mini-batches. That's because we update weights after each propagation. In our example we've propagated 11 batches \(10 of them had 100 samples and 1 had 50 samples\) and after each of them we've updated network's parameters. If we used all samples during propagation we would make only 1 update for the network's parameter.**

**Disadvantages:**

* **The smaller the batch the less accurate estimate of the gradient. In the figure below you can see that mini-batch \(green color\) gradient's direction fluctuates compare to the full batch \(blue color\).**

![enter image description here](https://lh3.googleusercontent.com/In_QJSs_c5iIJCuUmnaPJZSjeOIu3HvqOldEtdryCh4TKTNwru6LjdVRq6A02IzwCBYxWNyesrVZn462HHXPfoZUZCOJjZh1cg2qz2tzJ93khr4hYc20vz-8goU9JRyqFI8GIFmp)

[**Small batch size has an effect on validation accuracy.**](http://forums.fast.ai/t/batch-size-effect-on-validation-accuracy/413)

![](https://lh6.googleusercontent.com/-eOGc8ZDsqSJWbu8J18jTRZUHxNuPbvBpvImJVK_zsYsk4GNtC7u-I0puhNbgIg0LzDS_v3-ySi519U8uWOyPv0qcvbLsaeHS3JaVt8jrjGygT2S608ON2d_QPZ2guCuqvwPq0Wq)**IMPORTANT: batch size in ‘.prediction’ is needed for some models,** [**only for technical reasons as seen here**](https://github.com/fchollet/keras/issues/3027)**, in keras.**

1. **\(**[**unread**](https://www.quora.com/Intuitively-how-does-mini-batch-size-affect-the-performance-of-stochastic-gradient-descent)**\) about mini batches and performance.**
2. **\(**[**unread**](https://stats.stackexchange.com/questions/164876/tradeoff-batch-size-vs-number-of-iterations-to-train-a-neural-network)**\) tradeoff between bath size and number of iterations**

[**Another observation, probably empirical**](https://stackoverflow.com/questions/35050753/how-big-should-batch-size-and-number-of-epochs-be-when-fitting-a-model-in-keras) **- to answer your questions on Batch Size and Epochs:**

**In general: Larger batch sizes result in faster progress in training, but don't always converge as fast. Smaller batch sizes train slower, but can converge faster. It's definitely problem dependent.**

**In general, the models improve with more epochs of training, to a point. They'll start to plateau in accuracy as they converge. Try something like 50 and plot number of epochs \(x axis\) vs. accuracy \(y axis\). You'll see where it levels out.**

## **BIAS**

[**The role of bias in NN**](https://stackoverflow.com/questions/2480650/role-of-bias-in-neural-networks) **- similarly to the ‘b’ in linear regression.**

![](https://lh4.googleusercontent.com/J2OMsHkzsj_c2GqMXdumCZkCNLWbSB2oRlodc9kXts2gko4L8Uf92t46HCG4C4nh5KJAvStQ-o3syY5jAiDTMNZM8fX98xEyaKPCtWtnR5sXKMAsALwVrlLeQzt8zkFVtR1bso3Z)

![](https://lh6.googleusercontent.com/MfRZSVTUDmh1sHI5lmQG1rgf9mDaF6X5EmqRCncUcq7zG24M457rg2OZwVBi33RH6ImIIJshLg3z1NJ7nw-YCwrwTXATOMYgXpCxh-CDA8awb9wXRvWBJlknfZV_9klTROdNr99F)

## **BATCH NORMALIZATION**

1. **The** [**best explanation**](https://blog.paperspace.com/busting-the-myths-about-batch-normalization/) **to what is BN and why to use it, including busting the myth that it solves internal covariance shift - shifting input distribution, and saying that it should come after activations as it makes more sense \(it does\),also a nice quote on where a layer ends is really good - it can end at the activation \(or not\). How to use BN in the test, hint: use a moving window. Bn allows us to use 2 parameters to control the input distribution instead of controlling all the weights.**
2. [**Medium on BN**](https://towardsdatascience.com/an-alternative-to-batch-normalization-2cee9051e8bc)
3. [**Medium on BN**](https://towardsdatascience.com/batch-normalization-theory-and-how-to-use-it-with-tensorflow-1892ca0173ad)
4. [**Ian goodfellow on BN**](https://www.youtube.com/watch?v=Xogn6veSyxA&feature=youtu.be&t=325)
5. [**Medium \#2 - a better one on BN, and adding to VGG**](https://towardsdatascience.com/batch-normalization-in-neural-networks-1ac91516821c)
6. [**Reddit on BN, mainly on the paper saying to use it before, but best practice is to use after**](https://www.reddit.com/r/MachineLearning/comments/67gonq/d_batch_normalization_before_or_after_relu/)
7. [**Diff between batch and norm \(weak explanation\)**](https://www.quora.com/What-are-the-practical-differences-between-batch-normalization-and-layer-normalization-in-deep-neural-networks)
8. [**Weight normalization for keras and TF**](http://krasserm.github.io/2018/11/10/weightnorm-implementation-options/)
9. [**Layer normalization keras**](https://pypi.org/project/keras-layer-normalization/)
10. [**Instance normalization keras**](https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/layers/normalization/instancenormalization.py)
11. [**batch/layer/instance in TF with code**](https://towardsdatascience.com/implementing-spatial-batch-instance-layer-normalization-in-tensorflow-manual-back-prop-in-tf-77faa8d2c362)
12. **Layer** [**norm for rnn’s or whatever name it is in this post**](https://twimlai.com/new-layer-normalization-technique-speeds-rnn-training/) **with** [**code**](https://gist.github.com/udibr/7f46e790c9e342d75dcbd9b1deb9d940) **for GRU**

[**What is the diff between batch/layer/recurrent batch and back rnn normalization**](https://datascience.stackexchange.com/questions/12956/paper-whats-the-difference-between-layer-normalization-recurrent-batch-normal)

* **Layer normalization \(Ba 2016\): Does not use batch statistics. Normalize using the statistics collected from all units within a layer of the current sample. Does not work well with ConvNets.**
* **Recurrent Batch Normalization \(BN\) \(Cooijmans, 2016; also proposed concurrently by Qianli Liao & Tomaso Poggio, but tested on Recurrent ConvNets, instead of RNN/LSTM\): Same as batch normalization. Use different normalization statistics for each time step. You need to store a set of mean and standard deviation for each time step.**
* **Batch Normalized Recurrent Neural Networks \(Laurent, 2015\): batch normalization is only applied between the input and hidden state, but not between hidden states. i.e., normalization is not applied over time.**
* **Streaming Normalization \(Liao et al. 2016\) : it summarizes existing normalizations and overcomes most issues mentioned above. It works well with ConvNets, recurrent learning and online learning \(i.e., small mini-batch or one sample at a time\):**
* **Weight Normalization \(Salimans and Kingma 2016\): whenever a weight is used, it is divided by its L2 norm first, such that the resulting weight has L2 norm 1. That is, output y=x∗\(w/\|w\|\), where x and w denote the input and weight respectively. A scalar scaling factor g is then multiplied to the output y=y∗g. But in my experience g seems not essential for performance \(also downstream learnable layers can learn this anyway\).**
* **Cosine Normalization \(Luo et al. 2017\): weight normalization is very similar to cosine normalization, where the same L2 normalization is applied to both weight and input: y=\(x/\|x\|\)∗\(w/\|w\|\). Again, manual or automatic differentiation can compute appropriate gradients of x and w.**
* **Note that both Weight and Cosine Normalization have been extensively used \(called normalized dot product\) in the 2000s in a class of ConvNets called HMAX \(Riesenhuber 1999\) to model biological vision. You may find them interesting.**

[**More about Batch/layer/instance/group norm are different methods for normalizing the inputs to the layers of deep neural networks**](https://nealjean.com/ml/neural-network-normalization/)

1. **Layer normalization solves the rnn case that batch couldnt - Is done per feature within the layer and normalized features are replaced**
2. **Instance does it for \(cnn?\) using per channel normalization**
3. **Group does it for group of channels**
4. ![](https://lh3.googleusercontent.com/P3AL20iV863GBbN_D07g1PBh2T3nEVrR0CYd_MXi5Gecozo-dc4CzbPemj5Bbyl4SbiZXtu-k8Q4hBXyh6c8SC8jOu4fU9B2G1vi0UT5nyGjDGAxURHqyre9NNmCnm5SVZpuHskF)

[**Part1: intuitive explanation to batch normalization**](http://mlexplained.com/2018/01/10/an-intuitive-explanation-of-why-batch-normalization-really-works-normalization-in-deep-learning-part-1/)

**Part2:** [**batch/layer/weight normalization**](http://mlexplained.com/2018/01/13/weight-normalization-and-layer-normalization-explained-normalization-in-deep-learning-part-2/) **- This is a good resource for advantages for every layer**

* **Layer, per feature in a batch,** 
* **weight - divided by the norm**

![](https://lh3.googleusercontent.com/IqvjdZcCmsI-rAJ4ye0aUIoyrYLXLJTE2XMeRAAMIi0MxRoSzpRaZ6Op6dWgZ1VkjvBNUcuS8Xr0V9jo7jIpE46-7ktlS9QTDf6vmM8LI4N9juxa3CaLY4B5Gkl9oNPd44DjN5Bs)

## **HYPER PARAM GRID SEARCHES**

1. [**A disciplined approach to neural network hyper-parameters: Part 1 -- learning rate, batch size, momentum, and weight decay**](https://arxiv.org/abs/1803.09820)

## **LOSS**

[**Very Basic advice**](https://stats.stackexchange.com/questions/232754/reference-to-learn-how-to-interpret-learning-curves-of-deep-convolutional-neural)**: You should probably switch train/validation repartition to something like 80% training and 20% validation. In most cases it will improve the classifier performance overall \(more training data = better performance\)**

**+If Training error and test error are too close \(your system is unable to overfit on your training data\), this means that your model is too simple. Solution: more layers or more neurons per layer.**

**Early stopping**

**If you have never heard about "early-stopping" you should look it up, it's an important concept in the neural network domain :** [**https://en.wikipedia.org/wiki/Early\_stopping**](https://en.wikipedia.org/wiki/Early_stopping) **. To summarize, the idea behind early-stopping is to stop the training once the validation loss starts plateauing. Indeed, when this happens it almost always mean you are starting to overfitt your classifier. The training loss value in itself is not something you should trust, beacause it will continue to increase event when you are overfitting your classifier.**

**With** [**cross entropy**](https://www.quora.com/Loss-cross-entropy-is-decreasing-but-accuracy-remains-the-same-while-training-convolutional-neural-networks-How-can-it-happen) **there can be an issue where the accuracy is the same for two cases, one where the loss is decreasing and the other when the loss is not changing much.**

![](https://lh3.googleusercontent.com/f2R8DVu5A9g6LOGbNcmyIfayuVBYnpScO_kNsAcuJ8lsiM-hnYwlqD04qyI1wPYTwmsr2KpFKJa19gMkkJd67y03iJquhRftQdBpfGEdw5OQHficHqgkxudLfgpZsSS7Cc2p9qDS)

[**How to read LOSS graphs \(and accuracy on top\)**](https://github.com/fchollet/keras/issues/3755)

![](https://lh6.googleusercontent.com/blj3natUcvqK-nEmNjv90zAIM74QbA4x7hQ_F_oPGcHxQcdhc0_NrcPZhWDne2EEnUnJKNDOw4Xt_cUkhv3cFTFMcqzzBT4NeOPPnmoTfTXLFrEnVwkrlc5PEsZDNCZXdOr0GRZj)![](https://lh4.googleusercontent.com/o39Jcw1o7JeSsKuD_q-9xGukmT6pWLGs-9sVIumxLRF7dPpf25w8o9e2OBnWbpPc_p6t9e03D46r34N-8CYZa6fvfcWBVp_7N06xE0kbrvIzBC5sGWcMymN_KtPTfRKwHk1-gRcQ)

**This indicates that the model is overfitting. It continues to get better and better at fitting the data that it sees \(training data\) while getting worse and worse at fitting the data that it does not see \(validation data\).**

[**This is a very good example of a train/test loss and an accuracy behavior.**](https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/)

![](https://lh6.googleusercontent.com/GK_rvndJY76-cgBbBetgSZfwTD7RTZW2UsXUtsEZRUvFW1ACpJw9FMhNwj3LBERvmmPvcuTkkwb5HUcXgi7ua42WqJwAZgFP-3NsyF1qEo9GmACXGQGWGSYh3AR7yY765Qm9QfiO)![](https://lh4.googleusercontent.com/Q46fiZLm9mMhuQnOVJjyZWstXj6Aq1Ctev1cvIUsdrOWiOqxfvNlkJjcW08waf8qCERvvt1AkW-HjDrLvjHiVxKTFzxfX0BmVq4hRUERqrGsNLALeJb75Geb06X21Bgb8z2dA6iw)

[**Cross entropy formula with soft labels \(probability\) rather than classes.**](https://stats.stackexchange.com/questions/206925/is-it-okay-to-use-cross-entropy-loss-function-with-soft-labels)

[**Mastery on cross entropy, brier, roc auc, how to ‘game’ them and calibrate them**](https://machinelearningmastery.com/how-to-score-probability-predictions-in-python/)

[**Game changer paper - a general adaptive loss search in nn**](https://www.reddit.com/r/computervision/comments/bsd82j/a_general_and_adaptive_robust_loss_function/?utm_medium=android_app&utm_source=share)

## **LEARNING RATE REDUCTION**

[**Intro to Learning Rate methods**](https://medium.com/@chengweizhang2012/quick-notes-on-how-to-choose-optimizer-in-keras-9d3d12d09039) **- what they are doing and what they are fixing in other algos.**

[**Callbacks**](https://keras.io/callbacks/)**, especially ReduceLROnPlateau - this callback monitors a quantity and if no improvement is seen for a 'patience' number of epochs, the learning rate is reduced.**

[**Cs123**](http://cs231n.github.io/neural-networks-3/) **\(very good\): explains about many things related to CNN, but also about LR and adaptive methods.**

[**An excellent comparison of several learning rate schedule methods and adaptive methods:**](https://medium.com/towards-data-science/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1) **\(**[**same here but not as good**](https://machinelearningmastery.com/using-learning-rate-schedules-deep-learning-models-python-keras/)**\)**

![](https://lh5.googleusercontent.com/UtrDKeqV_UfuPuot937svdmi-fzHp3K_eRS5xFAgQI7CAXPFchkFCQO4YPYOFkWMG6tYDlAeATR0YUwOLKqLlDq17T-Row_iBknUXchk9zT2_0KBzE7BMipHBKPds-sFw_0NDAjF)

**Adaptive gradient descent algorithms such as** [**Adagrad**](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#AdaGrad)**, Adadelta,** [**RMSprop**](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#RMSProp)**,** [**Adam**](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Adam)**, provide an alternative to classical SGD.**

**These per-parameter learning rate methods provide heuristic approach without requiring expensive work in tuning hyperparameters for the learning rate schedule manually.**

1. **Adagrad performs larger updates for more sparse parameters and smaller updates for less sparse parameter. It has good performance with sparse data and training large-scale neural network. However, its monotonic learning rate usually proves too aggressive and stops learning too early when training deep neural networks.**
2. **Adadelta is an extension of Adagrad that seeks to reduce its aggressive, monotonically decreasing learning rate.**
3. **RMSprop adjusts the Adagrad method in a very simple way in an attempt to reduce its aggressive, monotonically decreasing learning rate.** 
4. [**Adam**](https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/) **is an update to the RMSProp optimizer which is like RMSprop with momentum.**

![](https://lh6.googleusercontent.com/ixb189Iy_Z4PuSCZHn48vmBvRDNchESvmANzapkuTNMt5zYp7vl9NLznUzNQYaMuyUQzhLiQgpCPUho9klBdd4W09dcjsdx8D_yIDvOcOK8Jo2_p6nDMmLv3QL5ohm07-pJmIo48)

**adaptive learning rate methods demonstrate better performance than learning rate schedules, and they require much less effort in hyperparamater settings**

![](https://lh3.googleusercontent.com/rYknk8vLbQKYuLSKeItX59a6rdi84U5QaeNJoardmv_jLgXqIMHj1BGbZsMh4l0Pli-mKYg29dNGDMKHS341t94fUScWELjPsIXWy7i1-_zXiCOSR1J46gMODzPQrrX4x64P1ato)

[**Recommended paper**](https://arxiv.org/pdf/1206.5533v2.pdf)**: practical recommendation for gradient based DNN**

**Another great comparison -** [**pdf paper**](https://arxiv.org/abs/1609.04747) **and** [**webpage link**](http://ruder.io/optimizing-gradient-descent/) **-**

* **if your input data is sparse, then you likely achieve the best results using one of the adaptive learning-rate methods.** 
* **An additional benefit is that you will not need to tune the learning rate but will likely achieve the best results with the default value.**
* **In summary, RMSprop is an extension of Adagrad that deals with its radically diminishing learning rates. It is identical to Adadelta, except that Adadelta uses the RMS of parameter updates in the numerator update rule. Adam, finally, adds bias-correction and momentum to RMSprop. Insofar, RMSprop, Adadelta, and Adam are very similar algorithms that do well in similar circumstances. Kingma et al. \[10\] show that its bias-correction helps Adam slightly outperform RMSprop towards the end of optimization as gradients become sparser. Insofar, Adam might be the best overall choice**

## **TRAIN / VAL accuracy in NN**

**The second important quantity to track while training a classifier is the validation/training accuracy. This plot can give you valuable insights into the amount of overfitting in your model:**

![](https://lh5.googleusercontent.com/K8KuSlFCGaOO9qihQGVQf3Cckcy5A2V98Tt_OKbscmv-ZmmemEVJFs2V9eeydc8Aa_dk-TXXjsJhiPCD7UAqKcvaMc4xsP0RIJNl0EiZ7ybQ5HsrINup7AYJjSfayQELeOA3WS_-)

* **The gap between the training and validation accuracy indicates the amount of overfitting.** 
* **Two possible cases are shown in the diagram on the left. The blue validation error curve shows very small validation accuracy compared to the training accuracy, indicating strong overfitting \(note, it's possible for the validation accuracy to even start to go down after some point\).** 
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

**However, i am still not seeing anything empirical that says that glorot surpesses everything else under certain conditions \(**[**except the glorot paper**](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)**\), most importantly, does it really help in LSTM where the vanishing gradient is ~no longer an issue?**

[**He-et-al Initialization**](https://towardsdatascience.com/random-initialization-for-neural-networks-a-thing-of-the-past-bfcdd806bf9e)

**This method of initializing became famous through a paper submitted in 2015 by He et al, and is similar to Xavier initialization, with the factor multiplied by two. In this method, the weights are initialized keeping in mind the size of the previous layer which helps in attaining a global minimum of the cost function faster and more efficiently.**

**w=np.random.randn\(layer\_size\[l\],layer\_size\[l-1\]\)\*np.sqrt\(2/layer\_size\[l-1\]\)**

## **ACTIVATION FUNCTIONS**

**\(**[**a bunch of observations, seems like a personal list**](http://sentiment-mining.blogspot.co.il/2015/08/the-difference-of-activation-function.html)**\) -**

* **Output layer - linear for regression, softmax for classification**
* **Hidden layers - hyperbolic tangent for shallow networks \(less than 3 hidden layers\), and ReLU for deep networks**

**ReLU - The purpose of ReLU is to introduce non-linearity, since most of the real-world data we would want our network to learn would be nonlinear \(e.g. convolution is a linear operation – element wise matrix multiplication and addition, so we account for nonlinearity by introducing a nonlinear function like ReLU, e.g** [**here**](https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/) **- search for ReLU\).**

* **Relu is quite resistant to vanishing gradient & allows for deactivating neurons and for sparsity.**
* **Other nonlinear functions such as tanh or sigmoid can also be used instead of ReLU, but ReLU has been found to perform better in most situations.**
* [**Visual + description of activation functions**](https://medium.com/the-theory-of-everything/understanding-activation-functions-in-neural-networks-9491262884e0)
* [**A very good explanation + figures about activations functions**](https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6)

[**Selu**](https://towardsdatascience.com/selu-make-fnns-great-again-snn-8d61526802a9) **- better than RELU? Possibly.**

[**Mish**](https://github.com/digantamisra98/Mish)**: A Self Regularized Non-Monotonic Neural Activation Function,** [**yam peleg’s code** ](https://gist.github.com/ypeleg/3af35d07d7f659f387952c9843849772?fbclid=IwAR2x_Hzlg79_mo_zQMJGFbQWORbpdydnllnHoA_RmUlCLpqKdGwClBuJy8g)

[**Mish, Medium, Keras Code, with benchmarks, computationally expensive.**](https://towardsdatascience.com/mish-8283934a72df)

## **OPTIMIZERS**

**There are several optimizers, each had his 15 minutes of fame, some optimizers are recommended for CNN, Time Series, etc..**

**There are also what I call ‘experimental’ optimizers, it seems like these pop every now and then, with or without a formal proof. It is recommended to follow the literature and see what are the ‘supposedly’ state of the art optimizers atm.**

[**Adamod**](https://medium.com/@lessw/meet-adamod-a-new-deep-learning-optimizer-with-memory-f01e831b80bd) **deeplearning optimizer with memory**

[**Backstitch**](http://www.danielpovey.com/files/2017_nips_backstitch.pdf) **- September 17 - supposedly an improvement over SGD for speech recognition using DNN. Note: it wasnt tested with other datasets or other network types.**

**\(how does it work?\) take a negative step back, then a positive step forward. I.e., When processing a minibatch, instead of taking a single SGD step, we first take a step with −α times the current learning rate, for α &gt; 0 \(e.g. α = 0.3\), and then a step with 1 + α times the learning rate, with the same minibatch \(and a recomputed gradient\). So we are taking a small negative step, and then a larger positive step. This resulted in quite large improvements – around 10% relative improvement \[37\] – for our best speech recognition DNNs. The recommended hyper parameters are in the paper.**

**Drawbacks: takes twice to train, momentum not implemented or tested, dropout is mandatory for improvement, slow starter.**

[**Documentation about optimizers**](https://keras.io/optimizers/) **in keras**

* **SGD can be fine tuned**
* **For others Leave most parameters as they were**

[**Best description on optimizers with momentum etc, from sgd to nadam, formulas and intuition**](https://towardsdatascience.com/10-gradient-descent-optimisation-algorithms-86989510b5e9)

![](https://lh6.googleusercontent.com/-quQMukoMffONyGh-R-nuGssirsDgFz6YQyZAjQ22FyQFglTbpnN0kA7VNQ3UH_o2DSus3SJs2ThnwMS0rnH3iIZN1cK8OzKb39oBj4c2lU-dE9k3c_MDuiMr51IeghvAHLZh2t9)

## **DROPOUT LAYERS IN KERAS AND GENERAL**

[**A very influential paper about dropout and how beneficial it is - bottom line always use it.**](http://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf)

**OPEN QUESTIONs:**

1. **does a dropout layer improve performance even if an lstm layer has dropout or recurrent dropout.**
2. **What is the diff between a separate layer and inside the lstm layer.**
3. **What is the diff in practice and intuitively between drop and recurrentdrop**

[**Dropout layers in keras, or dropout regularization:** ](https://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/)

* **Dropout is a technique where randomly selected neurons are ignored RANDOMLY during training.** 
* **contribution to the activation of downstream neurons is temporally removed on the forward pass and any weight updates are not applied to the neuron on the backward pass.**
* **As a neural network learns, neuron weights settle into their context within the network.**
* **Weights of neurons are tuned for specific features providing some specialization. Neighboring neurons become to rely on this specialization, which if taken too far can result in a fragile model too specialized to the training data. \(overfitting\)**
* **This reliant on context for a neuron during training is referred to complex co-adaptations.**
* **After dropout, other neurons will have to step in and handle the representation required to make predictions for the missing neurons, which is believed to result in multiple independent internal representations being learned by the network.**
* **Thus, the effect of dropout is that the network becomes less sensitive to the specific weights of neurons.** 
* **This in turn leads to a network with better generalization capability and less likely to overfit the training data.**

[**Another great answer about drop out**](https://www.quora.com/In-Keras-what-is-a-dense-and-a-dropout-layer) **-**

* **as a consequence of the 50% dropout, the neural network will learn different, redundant representations; the network can’t rely on the particular neurons and the combination \(or interaction\) of these to be present.** 
* **Another nice side effect is that training will be faster.**
* **Rules:** 
  * **Dropout is only applied during training,** 
  * **Need to rescale the remaining neuron activations. E.g., if you set 50% of the activations in a given layer to zero, you need to scale up the remaining ones by a factor of 2.** 
  * **if the training has finished, you’d use the complete network for testing \(or in other words, you set the dropout probability to 0\).**

[**Implementation of drop out in keras**](https://datascience.stackexchange.com/questions/18088/convolutional-layer-dropout-layer-in-keras/18098) **is “inverse dropout” - n the Keras implementation, the output values are corrected during training \(by dividing, in addition to randomly dropping out the values\) instead of during testing \(by multiplying\). This is called "inverted dropout".**

**Inverted dropout is functionally equivalent to original dropout \(as per your link to Srivastava's paper\), with a nice feature that the network does not use dropout layers at all during test and prediction. This is explained a little in this** [**Keras issue**](https://github.com/fchollet/keras/issues/3305)**.**

[**Dropout notes and rules of thumb aka “best practice” -** ](http://blog.mrtanke.com/2016/10/09/Keras-Study-Notes-3-Dropout-Regularization-for-Deep-Networks/)

* **dropout value of 20%-50% of neurons with 20% providing a good starting point. \(A probability too low has minimal effect and a value too high results in underlearning by the network.\)**
* **Use a large network for better performance, i.e.,  when dropout is used on a larger network, giving the model more of an opportunity to learn independent representations.**
* **Use dropout on VISIBLE AND HIDDEN. Application of dropout at each layer of the network has shown good results.**
* **Unclear ? Use a large learning rate with decay and a large momentum. Increase your learning rate by a factor of 10 to 100 and use a high momentum value of 0.9 or 0.99.**
* **Unclear ? Constrain the size of network weights. A large learning rate can result in very large network weights. Imposing a constraint on the size of network weights such as max-norm regularization with a size of 4 or 5 has been shown to improve results.**

[**Difference between LSTM ‘dropout’ and ‘recurrent\_dropout’**](https://stackoverflow.com/questions/44924690/keras-the-difference-between-lstm-dropout-and-lstm-recurrent-dropout) **- vertical vs horizontal.**

**I suggest taking a look at \(the first part of\)** [**this paper**](https://arxiv.org/pdf/1512.05287.pdf)**. Regular dropout is applied on the inputs and/or the outputs, meaning the vertical arrows from x\_t and to h\_t. In you add it as an argument to your layer, it will mask the inputs; you can add a Dropout layer after your recurrent layer to mask the outputs as well. Recurrent dropout masks \(or "drops"\) the connections between the recurrent units; that would be the horizontal arrows in your picture.**

**This picture is taken from the paper above. On the left, regular dropout on inputs and outputs. On the right, regular dropout PLUS recurrent dropout:**

![This picture is taken from the paper above. On the left, regular dropout on inputs and outputs. On the right, regular dropout PLUS recurrent dropout.](https://lh3.googleusercontent.com/RF9eawLdYCty8TSrEBsd3NvaxpFbQNG9s551Q-sX1OVlsC3MRZZ1q5s-xYZVv81Z_-3SvK4JwtAwUirZuCE8MPIISw0ebchNTqY3IMEpc76jalJG-0oeRpDGrWMTnYtAELhs0c3-)

## **NEURAL NETWORK OPTIMIZATION TECHNIQUES**

**Basically do these after you have a working network**

1. [**Dont decay the learning rate, increase batchsize - paper**](https://arxiv.org/abs/1711.00489) **\(optimization of a network\)**
2. [**Add one neuron with skip connection, or to every layer in a binary classification network to get global minimum**](https://arxiv.org/abs/1805.08671)**.**

## **Fine tuning**

1. [**3 methods to fine tune, cut softmax layer, smaller learning rate, freeze layers**](https://flyyufelix.github.io/2016/10/03/fine-tuning-in-keras-part1.html)
2. [**Fine tuning on a sunset of data**](https://stats.stackexchange.com/questions/289036/fine-tuning-with-a-subset-of-the-same-data)

## **Deep Learning for NLP**

* **\(did not fully read\)** [**Yoav Goldberg’s course**](https://docs.google.com/document/d/1Xf_dqjf7mWmSoYX0HTKnml2mssP5BjrKUs-4E17CbNo/edit#) **syllabus with lots of relevant topics on DL4NLP, including bidirectional RNNS and tree RNNs.**
* **\(did not fully read\)** [**CS224d**](http://cs224d.stanford.edu/index.html)**: Deep Learning for Natural Language Processing, with** [**slides etc.**](http://cs224d.stanford.edu/syllabus.html)

[**Deep Learning using Linear Support Vector Machines**](http://deeplearning.net/wp-content/uploads/2013/03/dlsvm.pdf) **- 1-3% decrease in error by replacing the softmax layer with a linear support vector machine**

## **MULTI LABEL/OUTPUT**

1. **A machine learning framework for** [**multi-output/multi-label**](https://github.com/scikit-multiflow/scikit-multiflow) **and stream data. Inspired by MOA and MEKA, following scikit-learn's philosophy.** [**https://scikit-multiflow.github.io/**](https://scikit-multiflow.github.io/)
2. [**Medium on MO, sklearn and keras**](https://towardsdatascience.com/what-data-scientists-should-know-about-multi-output-and-multi-label-training-b9d4be620e11)
3. [**MO in keras, see functional API on how.**](https://www.pyimagesearch.com/2018/06/04/keras-multiple-outputs-and-multiple-losses/%5C)

### **FUZZY MULTI LABEL**

1. [**Ie., probabilities or soft values instead of hard labels**](https://datascience.stackexchange.com/questions/48111/multilabel-classifcation-in-sklearn-with-soft-fuzzy-labels)

## **DNN FRAMEWORKS**

### **PYTORCH**

1. **Deep learning with pytorch -** [**The book**](https://pytorch.org/assets/deep-learning/Deep-Learning-with-PyTorch.pdf)
2. [**Pytorch DL course**](https://atcold.github.io/pytorch-Deep-Learning/)**,** [**git**](https://github.com/Atcold/pytorch-Deep-Learning) **- yann lecun** 

### **FAST.AI**

1. [**git**](https://github.com/fastai/fastai)

### **KERAS**

[**A make sense introduction into keras**](https://www.youtube.com/playlist?list=PLFxrZqbLojdKuK7Lm6uamegEFGW2wki6P)**, has several videos on the topic, going through many network types, creating custom activation functions, going through examples.**

**+ Two extra videos from the same author,** [**examples**](https://www.youtube.com/watch?v=6RdflAr66-E) **and** [**examples-2**](https://www.youtube.com/watch?v=fDKdITMBAGk)

**Didn’t read:**

1. [**Keras cheatsheet**](https://www.datacamp.com/community/blog/keras-cheat-sheet)
2. [**Seq2Seq RNN**](https://stackoverflow.com/questions/41933958/how-to-code-a-sequence-to-sequence-rnn-in-keras)
3. [**Stateful LSTM**](https://github.com/fchollet/keras/blob/master/examples/stateful_lstm.py) **- Example script showing how to use stateful RNNs to model long sequences efficiently.**
4. [**CONV LSTM**](https://github.com/fchollet/keras/blob/master/examples/conv_lstm.py) **- this script demonstrate the use of a conv LSTM network, used to predict the next frame of an artificially generated move which contains moving squares.**

[**How to force keras to use tensorflow**](https://github.com/ContinuumIO/anaconda-issues/issues/1735) **and not teano \(set the .bat file\)**

[**Callbacks - how to create an AUC ROC score callback with keras**](https://keunwoochoi.wordpress.com/2016/07/16/keras-callbacks/) **- with code example.**

[**Batch size vs. Iteratio**](https://stats.stackexchange.com/questions/164876/tradeoff-batch-size-vs-number-of-iterations-to-train-a-neural-network)**ns in NN  Keras.**

[**Keras metrics**](https://machinelearningmastery.com/custom-metrics-deep-learning-keras-python/) **- classification regression and custom metrics**

[**Keras Metrics 2**](https://machinelearningmastery.com/metrics-evaluate-machine-learning-algorithms-python/) **- accuracy, ROC, AUC, classification, regression r^2.**

[**Introduction to regression models in Keras,**](https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/) **using MSE, comparing baseline vs wide vs deep networks.**

[**How does Keras calculate accuracy**](https://datascience.stackexchange.com/questions/14415/how-does-keras-calculate-accuracy)**? Formula and explanation**

**Compares label with the rounded predicted float, i.e. bigger than 0.5 = 1, smaller than = 0**

**For categorical we take the argmax for the label and the prediction and compare their location.**

**In both cases, we average the results.**

[**Custom metrics \(precision recall\) in keras**](https://stackoverflow.com/questions/41458859/keras-custom-metric-for-single-class-accuracy)**. Which are taken from** [**here**](https://github.com/autonomio/talos/tree/master/talos/metrics)**, including entropy and f1**

**KERAS MULTI GPU**

1. [**When using SGD only batches between 32-512 are adequate, more can lead to lower performance, less will lead to slow training times.**](https://arxiv.org/pdf/1609.04836.pdf)
2. **Note: probably doesn't reflect on adam, is there a reference?**
3. [**Parallel gpu-code for keras. Its a one liner, but remember to scale batches by the amount of GPU used in order to see a \(non linear\) scaability in training time.**](https://datascience.stackexchange.com/questions/23895/multi-gpu-in-keras)
4. [**Pitfalls in GPU training, this is a very important post, be aware that you can corrupt your weights using the wrong combination of batches-to-input-size**](http://blog.datumbox.com/5-tips-for-multi-gpu-training-with-keras/)**, in keras-tensorflow. When you do multi-GPU training, it is important to feed all the GPUs with data. It can happen that the very last batch of your epoch has less data than defined \(because the size of your dataset can not be divided exactly by the size of your batch\). This might cause some GPUs not to receive any data during the last step. Unfortunately some Keras Layers, most notably the Batch Normalization Layer, can’t cope with that leading to nan values appearing in the weights \(the running mean and variance in the BN layer\).**
5. [**5 things to be aware of for multi gpu using keras, crucial to look at before doing anything** ](http://blog.datumbox.com/5-tips-for-multi-gpu-training-with-keras/)

**KERAS FUNCTIONAL API**

[**What is and how to use?**](https://machinelearningmastery.com/keras-functional-api-deep-learning/) **A flexible way to declare layers in parallel, i.e. parallel ways to deal with input, feature extraction, models and outputs as seen in the following images.**  
![Neural Network Graph With Shared Feature Extraction Layer](https://lh5.googleusercontent.com/tdK7TuCAsYPfx_vLBps4HU2dLQqA2M7prppP5V7xOzuT2SGeV_T3hJ94wvJMC0gBY1XS81bK6uKzOZ2HNazaEBRtD-a1xAtPS8OtcaEtjhqRi-GjH1iFOZM_2WDCWzs73odUzTbd)![Neural Network Graph With Multiple Inputs](https://lh6.googleusercontent.com/ptnE_MAQyTSSYyRCULQRnIx7XRa_7zVLSEbclJuebxvZPotAqJIe2ElY5SuF42UdfrEdIWFII7BwsVUrCkAXp3Ta1GCmrPLsir-duOxF5wkRn62uH0M4etHjBVNQOF7luWc4Qs9K)

![Neural Network Graph With Multiple Outputs](https://lh4.googleusercontent.com/pdU8st0CBS7qGN14dBXm6XbFJCL-hMAPtRjz__la0DN96IwABz-PV0i-xTEEAf5yBMOTBfi6QwAsnuGFnonRbSxdbQWl33bssITuR3zInVupAW0z9RSTCpqc9UwlAi6PZ0elyDLa)

**KERAS EMBEDDING LAYER**

1. [**Injecting glove to keras embedding layer and using it for classification + what is and how to use the embedding layer in keras.**](https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/)
2. [**Keras blog - using GLOVE for pretrained embedding layers.**](https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html)
3. [**Word embedding using keras, continuous BOW - CBOW, SKIPGRAM, word2vec - really good.**](https://towardsdatascience.com/understanding-feature-engineering-part-4-deep-learning-methods-for-text-data-96c44370bbfa)
4. [**Fasttext - comparison of key feature against word2vec**](https://www.quora.com/What-is-the-main-difference-between-word2vec-and-fastText)
5. [**Multiclass classification using word2vec/glove + code**](https://github.com/dennybritz/cnn-text-classification-tf/issues/69)
6. [**word2vec/doc2vec/tfidf code in python for text classification**](https://github.com/davidsbatista/text-classification/blob/master/train_classifiers.py)
7. [**Lda & word2vec**](https://www.kaggle.com/vukglisovic/classification-combining-lda-and-word2vec)
8. [**Text classification with word2vec**](http://nadbordrozd.github.io/blog/2016/05/20/text-classification-with-word2vec/)
9. [**Gensim word2vec**](https://radimrehurek.com/gensim/models/word2vec.html)**, and** [**another one**](http://kavita-ganesan.com/gensim-word2vec-tutorial-starter-code/)
10. [**Fasttext paper**](https://arxiv.org/abs/1607.01759)

**Keras: Predict vs Evaluate**

[**here:**](https://www.quora.com/What-is-the-difference-between-keras-evaluate-and-keras-predict)

**.predict\(\) generates output predictions based on the input you pass it \(for example, the predicted characters in the** [**MNIST example**](https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py)**\)**

**.evaluate\(\) computes the loss based on the input you pass it, along with any other metrics that you requested in the metrics param when you compiled your model \(such as accuracy in the** [**MNIST example**](https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py)**\)**

**Keras metrics**

[**For classification methods - how does keras calculate accuracy, all functions.**](https://www.quora.com/How-does-Keras-calculate-accuracy)

**LOSS IN KERAS**

[**Why is the training loss much higher than the testing loss?**](https://keras.io/getting-started/faq/#why-is-the-training-loss-much-higher-than-the-testing-loss) **A Keras model has two modes: training and testing. Regularization mechanisms, such as Dropout and L1/L2 weight regularization, are turned off at testing time.**

**The training loss is the average of the losses over each batch of training data. Because your model is changing over time, the loss over the first batches of an epoch is generally higher than over the last batches. On the other hand, the testing loss for an epoch is computed using the model as it is at the end of the epoch, resulting in a lower loss.**

## **DNN ALGORITHMS**

### **AUTOENCODERS**

1. [**How to use AE for dimensionality reduction + code**](https://statcompute.wordpress.com/2017/01/15/autoencoder-for-dimensionality-reduction/) **- using keras’ functional API**
2. [**Keras.io blog post about AE’s**](https://blog.keras.io/building-autoencoders-in-keras.html) **- regular, deep, sparse, regularized, cnn, variational**
   1. **A keras.io** [**replicate post**](https://towardsdatascience.com/deep-inside-autoencoders-7e41f319999f) **but explains AE quite nicely.**
3. [**Examples of vanilla, multi layer, CNN and sparse AE’s**](https://wiseodd.github.io/techblog/2016/12/03/autoencoders/)
4. [**Another example of CNN-AE**](https://hackernoon.com/autoencoders-deep-learning-bits-1-11731e200694)
5. [**Another AE tutorial**](https://towardsdatascience.com/how-to-reduce-image-noises-by-autoencoder-65d5e6de543)
6. [**Hinton’s coursera course**](https://www.coursera.org/learn/neural-networks/lecture/JiT1i/from-pca-to-autoencoders-5-mins) **on PCA vs AE, basically some info about what PCA does - maximizing variance and projecting and then what AE does and can do to achieve similar but non-linear dense representations**
7. [**A great tutorial on how does the clusters look like after applying PCA/ICA/AE**](https://www.kaggle.com/den3b81/2d-visualization-pca-ica-vs-autoencoders)
8. [**Another great presentation on PCA vs AE,**](https://web.cs.hacettepe.edu.tr/~aykut/classes/fall2016/bbm406/slides/l25-kernel_pca.pdf) **summarized in the KPCA section of this notebook. +**[**another one**](https://www.cs.toronto.edu/~urtasun/courses/CSC411/14_pca.pdf) **+**[**StackE**](https://stats.stackexchange.com/questions/261265/factor-analysis-vs-autoencoders)**xchange**
9. [**Autoencoder tutorial with python code and how to encode after**](https://ramhiser.com/post/2018-05-14-autoencoders-with-keras/)**,** [**mastery**](https://machinelearningmastery.com/encoder-decoder-attention-sequence-to-sequence-prediction-keras/)
10. [**Git code for low dimensional auto encoder**](https://github.com/Mylittlerapture/Low-Dimensional-Autoencoder)
11. [**Bart denoising AE**](https://arxiv.org/pdf/1910.13461.pdf)**, sequence to sequence pre training for NL generation translation and comprehension.** 
12. [**Attention based seq to seq auto encoder**](https://wanasit.github.io/attention-based-sequence-to-sequence-in-keras.html)**,** [**git**](https://github.com/wanasit/katakana)

[**AE for anomaly detection, fraud detection**](https://medium.com/@curiousily/credit-card-fraud-detection-using-autoencoders-in-keras-tensorflow-for-hackers-part-vii-20e0c85301bd)

### **Variational AE**

1. **Unread -** [**Simple explanation**](https://medium.com/@dmonn/what-are-variational-autoencoders-a-simple-explanation-ea7dccafb0e3)
2. [**Pixel art VAE**](https://mlexplained.wordpress.com/category/generative-models/vae/)
3. [**Unread - another VAE**](https://towardsdatascience.com/teaching-a-variational-autoencoder-vae-to-draw-mnist-characters-978675c95776)
4. [**Pixel GAN VAE**](https://medium.com/@Synced/pixelgan-autoencoders-17496632b755)
5. [**Disentangled VAE**](https://www.youtube.com/watch?v=9zKuYvjFFS8) **- improves VAE**
6. **Optimus -** [**pretrained VAE**](https://github.com/ophiry/Optimus)**,** [**paper**](https://arxiv.org/abs/2004.04092)**,** [**Microsoft blog**](https://www.microsoft.com/en-us/research/blog/a-deep-generative-model-trifecta-three-advances-that-work-towards-harnessing-large-scale-power/)\*\*\*\*

![Optimus](../.gitbook/assets/image%20%281%29.png)

### **SELF ORGANIZING MAPS \(SOM\)**

1. **Git**
   1. [**Sompy**](https://github.com/sevamoo/SOMPY)**,** 
   2. **\*\*\***[**minisom!**](https://github.com/JustGlowing/minisom)
   3. [**Many graph examples**](https://medium.com/@s.ganjoo96/self-organizing-maps-b2cf58b74fdb)**,** [**example**](https://github.com/lightsalsa251/Self-Organizing-Map)
2. [**Step by step with examples, calculations**](https://mc.ai/self-organizing-mapsom/)
3. [**Adds intuition regarding “magnetism”’**](https://towardsdatascience.com/self-organizing-maps-1b7d2a84e065)
4. [**Implementation and faces**](https://medium.com/@navdeepsingh_2336/self-organizing-maps-for-machine-learning-algorithms-ad256a395fc5)**, intuition towards each node and what it represents in a vision. I.e., each face resembles one of K clusters.**
5. [**Medium on kohonen networks, i.e., SOM**](https://towardsdatascience.com/kohonen-self-organizing-maps-a29040d688da)
6. [**Som on iris**](https://towardsdatascience.com/self-organizing-maps-ff5853a118d4)**, explains inference - averaging, and cons of the method.**
7. [**Simple explanation**](https://medium.com/@valentinerutto/selforganizingmaps-in-english-35574f95b0ac)
8. [**Algorithm, formulas**](https://towardsdatascience.com/kohonen-self-organizing-maps-a29040d688da)

### **NEURO EVOLUTION \(GA/GP based\)**

**NEAT**

[**NEAT**](http://www.cs.ucf.edu/~kstanley/neat.html) \*\*stands for NeuroEvolution of Augmenting Topologies. It is a method for evolving artificial neural networks with a genetic algorithm.

NEAT implements the idea that it is most effective to start evolution with small, simple networks and allow them to become increasingly complex over generations.\*\*

\*\*That way, just as organisms in nature increased in complexity since the first cell, so do neural networks in NEAT.

This process of continual elaboration allows finding highly sophisticated and complex neural networks.\*\*

[**A great article about NEAT**](http://hunterheidenreich.com/blog/neuroevolution-of-augmenting-topologies/)

**HYPER-NEAT**

[**HyperNEAT**](http://eplex.cs.ucf.edu/hyperNEATpage/) \*\*computes the connectivity of its neural networks as a function of their geometry.

HyperNEAT is based on a theory of representation that hypothesizes that a good representation for an artificial neural network should be able to describe its pattern of connectivity compactly.\*\*

**The encoding in HyperNEAT, called** [**compositional pattern producing networks**](http://en.wikipedia.org/wiki/Compositional_pattern-producing_network)\*\*, is designed to represent patterns with regularities such as symmetry, repetition, and repetition with variationץ

\(WIKI\) **\[**Compositional pattern-producing networks**\]\(**[https://en.wikipedia.org/wiki/Compositional\_pattern-producing\_network](https://en.wikipedia.org/wiki/Compositional_pattern-producing_network)**\)** \(CPPNs\) are a variation of artificial neural networks \(ANNs\) that have an architecture whose evolution is guided by genetic algorithms\*\*

![](https://lh6.googleusercontent.com/cAbcsLDWcDOMlX4K53ROOLyiAw6EhJ9ZRDuZmURFtBaje8JtwzU_KsOh4aeiC8ukdYgBYEm6zqWd7jZ3tStib3JJGYrmxM4wlrgyBJFhlnMHd_kIcxgO2reEsoE4RPjJLXr3O-R_)

[**A great HyperNeat tutorial on Medium.**](https://towardsdatascience.com/hyperneat-powerful-indirect-neural-network-evolution-fba5c7c43b7b)

### **Radial Basis Function Network \(RBFN\)**

**+** [**RBF layer in Keras.**](https://github.com/PetraVidnerova/rbf_keras/blob/master/test.py)

**The** [**RBFN**](http://mccormickml.com/2013/08/15/radial-basis-function-network-rbfn-tutorial/) **approach is more intuitive than the MLP.**

* **An RBFN performs classification by measuring the input’s similarity to examples from the training set.** 
* **Each RBFN neuron stores a “prototype”, which is just one of the examples from the training set.** 
* **When we want to classify a new input, each neuron computes the Euclidean distance between the input and its prototype.** 
* **Roughly speaking, if the input more closely resembles the class A prototypes than the class B prototypes, it is classified as class A.**

![Architecture\_Simple](https://lh6.googleusercontent.com/5oVVPw02w2Pv1kqAGvQ6drOX6Nh7lA72cBDplTbqgd78u25ceNdjufDe8h4pKWNPKC350_r4V_TPUn1ionjck1IPJiW0Q4rwivL4sH4LJGaj7V7WZBss8eLSuqpZb5Rv525M4sQ1)

### **Bayesian Neural Network \(BNN\)**

[**BNN**](https://eng.uber.com/neural-networks-uncertainty-estimation/) **- \(what is?\)** [**Bayesian neural network \(BNN\)**](http://edwardlib.org/tutorials/bayesian-neural-network) **according to Uber - architecture that more accurately forecasts time series predictions and uncertainty estimations at scale. “how Uber has successfully applied this model to large-scale time series anomaly detection, enabling better accommodate rider demand during high-traffic intervals.”**

**Under the BNN framework, prediction uncertainty can be categorized into three types:**

1. **Model uncertainty captures our ignorance of the model parameters and can be reduced as more samples are collected.** 
2. **model misspecification**
3. **inherent noise captures the uncertainty in the data generation process and is irreducible.** 

**Note: in a series of articles, uber explains about time series and leads to a BNN architecture.**

1. [**Neural networks**](https://eng.uber.com/neural-networks/) **- training on multi-signal raw data, training X and Y are window-based and the window size\(lag\) is determined in advance.**

**Vanilla LSTM did not work properly, therefore an architecture of**

**Regarding point 1: ‘run prediction with dropout 100 times’**

**\*\*\*** [**MEDIUM with code how to do it.**](https://medium.com/hal24k-techblog/how-to-generate-neural-network-confidence-intervals-with-keras-e4c0b78ebbdf)

[**Why do we need a confidence measure when we have a softmax probability layer?**](https://hjweide.github.io/quantifying-uncertainty-in-neural-networks) **The blog post explains, for example, that with a CNN of apples, oranges, cat and dogs, a non related example such as a frog image may influence the network to decide its an apple, therefore we can’t rely on the probability as a confidence measure. The ‘run prediction with dropout 100 times’ should give us a confidence measure because it draws each weight from a bernoulli distribution.**

**“By applying dropout to all the weight layers in a neural network, we are essentially drawing each weight from a** [**Bernoulli distribution**](https://en.wikipedia.org/wiki/Bernoulli_distribution)**. In practice, this mean that we can sample from the distribution by running several forward passes through the network. This is referred to as** [**Monte Carlo dropout**](http://arxiv.org/abs/1506.02158)**.”**

**Taken from Yarin Gal’s** [**blog post**](http://mlg.eng.cam.ac.uk/yarin/blog_3d801aa532c1ce.html) **. In this figure we see how sporadic is the signal from a forward pass \(black line\) compared to a much cleaner signal from 100 dropout passes.**

![](https://lh5.googleusercontent.com/FlcvG689kstX36ya8JNaeIE6C5HeXhL7IKG3wMt5zTacLqJVmb9W6kqpby_e3IMV6iWc7rrIJ8F6IMwKEM6hUiuHnLaJiLp4KBPkTird_AB4GW8i5-5n_DOOm-cZEQYUsM6TWotp)

**Is it applicable for time series? In the figure below he tried to predict the missing signal between each two dotted lines, A is a bad estimation, but with a dropout layer we can see that in most cases the signal is better predicted.**

![](https://lh6.googleusercontent.com/eNr1VJ6ahkfVOvZ0i3HIFqng_hyCYueyZQ5jqb20mB55MtZwpd8EJ6Qhda7Ty0oRwLsNFUN4YSUN2sAUW768lA2PyAqIUiLOMULMXZtBJKlU54Me0p2CeVJIkOubgoNV-hnwD5Ip)

**Going back to uber, they are actually using this idea to predict time series with LSTM, using encoder decoder framework.**

![](https://lh6.googleusercontent.com/OoKHnEH6OcZVOBorLKp-rvUFWueY6qjwLW_v0mHWLGKp1YSZeRscteXA59Ecqp77B-PWv5nB7v6Hyf-emOu6eABkNW6LTAGEVSUgwtPLBKKJZBSRHIy8JbiCqwcc3-RbyiFvtd8z)

**Note: this is probably applicable in other types of networks.**

[**Phd Thesis by Yarin**](http://mlg.eng.cam.ac.uk/yarin/blog_2248.html?fref=gc&dti=999449923520287)**, he talks about uncertainty in Neural networks and using BNNs. he may have proved this thesis, but I did not read it. This blog post links to his full Phd.**

**Old note:** [**The idea behind uncertainty is \(**](http://mlg.eng.cam.ac.uk/yarin/blog_3d801aa532c1ce.html)[**paper here**](https://arxiv.org/pdf/1506.02142.pdf)**\) that in order to trust your network’s classification, you drop some of the neurons during prediction, you do this ~100 times and you average the results. Intuitively this will give you confidence in your classification and increase your classification accuracy, because only a partial part of your network participated in the classification, randomly, 100 times. Please note that Softmax doesn't give you certainty.**

[**Medium post on prediction with drop out** ](https://towardsdatascience.com/is-your-algorithm-confident-enough-1b20dfe2db08)

**The** [**solution for keras**](https://github.com/keras-team/keras/issues/9412) **says to add trainable=true for every dropout layer and add another drop out at the end of the model. Thanks sam.**

**“import keras**

**inputs = keras.Input\(shape=\(10,\)\)**

**x = keras.layers.Dense\(3\)\(inputs\)**

**outputs = keras.layers.Dropout\(0.5\)\(x, training=True\)**

**model = keras.Model\(inputs, outputs\)“**

### **CONVOLUTIONAL NEURAL NET**

![](https://lh5.googleusercontent.com/yw2GIv_A_BJLggUjAcF7K3NFbvf9BsGiMS4PQHgLjl6H5sAziuofhepBZOlsWvJnK296FbGTOGYsOdWCmkpyesvuO9BtqcReXIVQy2xT3SOCNIH4riyTrpjL7M2tOOlG6eH_3SEN)

**\(**[**an excellent and thorough explanation about LeNet**](https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/)**\) -**

* **Convolution Layer primary purpose is to extract features from the input image. Convolution preserves the spatial relationship between pixels by learning image features using small squares of input data.**
* **ReLU \(more in the activation chapter\) - The purpose of ReLU is to introduce non-linearity in our ConvNet**
* **Spatial Pooling \(also called subsampling or downsampling\) reduces the dimensionality of each feature map but retains the most important information. Spatial Pooling can be of different types: Max, Average, Sum etc.**
* **Dense / Fully Connected - a traditional Multi Layer Perceptron that uses a softmax activation function in the output layer to classify. The output from the convolutional and pooling layers represent high-level features of the input image. The purpose of the Fully Connected layer is to use these features for classifying the input image into various classes based on the training dataset.**

**The overall training process of the Convolutional Network may be summarized as below:**

* **Step1: We initialize all filters and parameters / weights with random values**
* **Step2: The network takes a single training image as input, goes through the forward propagation step \(convolution, ReLU and pooling operations along with forward propagation in the Fully Connected layer\) and finds the output probabilities for each class.**
  * **Let's say the output probabilities for the boat image above are \[0.2, 0.4, 0.1, 0.3\]**
  * **Since weights are randomly assigned for the first training example, output probabilities are also random.**
* **Step3: Calculate the total error at the output layer \(summation over all 4 classes\)**
  * **\(L2\) Total Error = ∑  ½ \(target probability – output probability\) ²**
* **Step4: Use Backpropagation to calculate the gradients of the error with respect to all weights in the network and use gradient descent to update all filter values / weights and parameter values to minimize the output error.**
  * **The weights are adjusted in proportion to their contribution to the total error.**
  * **When the same image is input again, output probabilities might now be \[0.1, 0.1, 0.7, 0.1\], which is closer to the target vector \[0, 0, 1, 0\].**
  * **This means that the network has learnt to classify this particular image correctly by adjusting its weights / filters such that the output error is reduced.**
  * **Parameters like number of filters, filter sizes, architecture of the network etc. have all been fixed before Step 1 and do not change during training process – only the values of the filter matrix and connection weights get updated.**
* **Step5: Repeat steps 2-4 with all images in the training set.**

**The above steps train the ConvNet – this essentially means that all the weights and parameters of the ConvNet have now been optimized to correctly classify images from the training set.**

**When a new \(unseen\) image is input into the ConvNet, the network would go through the forward propagation step and output a probability for each class \(for a new image, the output probabilities are calculated using the weights which have been optimized to correctly classify all the previous training examples\). If our training set is large enough, the network will \(hopefully\) generalize well to new images and classify them into correct categories.**

[**Illustrated 10 CNNS architectures**](https://towardsdatascience.com/illustrated-10-cnn-architectures-95d78ace614d)

[**A study that deals with class imbalance in CNN’s**](https://arxiv.org/pdf/1710.05381.pdf) **- we systematically investigate the impact of class imbalance on classification performance of convolutional neural networks \(CNNs\) and compare frequently used methods to address the issue**

1. **Over sampling**
2. **Undersampling**
3. **Thresholding probabilities \(ROC?\)**
4. **Cost sensitive classification -different cost to misclassification**
5. **One class - novelty detection. This is a concept learning technique that recognizes positive instances rather than discriminating between two classes**

**Using several imbalance scenarios, on several known data sets, such as MNIST**![](https://lh5.googleusercontent.com/dsLGbR3YBUjsDjRuOiC5FSrfef4MoK2Y1J-wPzn4NmIJWxg3wP7aY8TvP1EXr8p6a4T5wjcFqv2teT11KlXaMQFh3eWOYRT-5Vn-xlAlacyckL7DDsAx4sJG5lt_tJC4rF2ytfhs)

**The results indication \(loosely\) that oversampling is usually better in most cases, and doesn't cause overfitting in CNNs.**

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

1. [**Making cnn shift invariance**](https://richzhang.github.io/antialiased-cnns/) **- “Small shifts -- even by a single pixel -- can drastically change the output of a deep network \(bars on left\). We identify the cause: aliasing during downsampling. We anti-alias modern deep networks with classic signal processing, stabilizing output classifications \(bars on right\). We even observe accuracy increases \(see plot below\).**

**MAX AVERAGE POOLING**

[**Intuitions to the differences between max and average pooling:**](https://stats.stackexchange.com/questions/291451/feature-extracted-by-max-pooling-vs-mean-pooling)

1. **A max-pool layer compressed by taking the maximum activation in a block. If you have a block with mostly small activation, but a small bit of large activation, you will loose the information on the low activations. I think of this as saying "this type of feature was detected in this general area".** 
2. **A mean-pool layer compresses by taking the mean activation in a block. If large activations are balanced by negative activations, the overall compressed activations will look like no activation at all. On the other hand, you retain some information about low activations in the previous example.**
3. **MAX pooling In other words: Max pooling roughly means that only those features that are most strongly triggering outputs are used in the subsequent layers. You can look at it a little like focusing the network’s attention on what’s most characteristic for the image at hand.**
4. [**GLOBAL MAX pooling**](https://alexisbcook.github.io/2017/global-average-pooling-layers-for-object-localization/)**: In the last few years, experts have turned to global average pooling \(GAP\) layers to minimize overfitting by reducing the total number of parameters in the model. Similar to max pooling layers, GAP layers are used to reduce the spatial dimensions of a three-dimensional tensor. However, GAP layers perform a more extreme type of dimensionality reduction,** 
5. [**Hinton’s controversy thoughts on pooling**](https://mirror2image.wordpress.com/2014/11/11/geoffrey-hinton-on-max-pooling-reddit-ama/)

**Dilated CNN**

1. [**For improved performance**](https://stackoverflow.com/questions/41178576/whats-the-use-of-dilated-convolutions)

   **RESNET, DENSENET UNET**

2. **A** [**https://medium.com/swlh/resnets-densenets-unets-6bbdbcfdf010**](https://medium.com/swlh/resnets-densenets-unets-6bbdbcfdf010)
3. **on the trick behind them, concatenating both f\(x\) = x**

### **Graph Convolutional Networks**

[**Explaination here, with some examples**](https://tkipf.github.io/graph-convolutional-networks/)

### **CAPSULE NEURAL NETS**

1. [**The solution to CNN’s shortcomings**](https://hackernoon.com/capsule-networks-are-shaking-up-ai-heres-how-to-use-them-c233a0971952)**, where features can be identified without relations to each other in an image, i.e. changing the location of body parts will not affect the classification, and changing the orientation of the image will. The promise of capsule nets is that these two issues are solved.**
2. [**Understanding capsule nets - part 2,**](https://medium.com/ai%C2%B3-theory-practice-business/understanding-hintons-capsule-networks-part-ii-how-capsules-work-153b6ade9f66) **there are more parts to the series**

### **Transfer Learning using CNN**

1. **To Add keras book chapter 5 \(i think\)**
2. [**Mastery**](https://machinelearningmastery.com/how-to-use-transfer-learning-when-developing-convolutional-neural-network-models/) **on TL using CNN**
   1. **Classifier: The pre-trained model is used directly to classify new images.**
   2. **Standalone Feature Extractor: The pre-trained model, or some portion of the model, is used to pre-process images and extract relevant features.**
   3. **Integrated Feature Extractor: The pre-trained model, or some portion of the model, is integrated into a new model, but layers of the pre-trained model are frozen during training.**
   4. **Weight Initialization: The pre-trained model, or some portion of the model, is integrated into a new model, and the layers of the pre-trained model are trained in concert with the new model.**

### **VISUALIZE CNN**

1. [**How to**](https://towardsdatascience.com/how-to-visualize-convolutional-features-in-40-lines-of-code-70b7d87b0030)

### **Recurrent Neural Net \(RNN\)**

**RNN - a basic NN node with a loop, previous output is merged with current input \(using tanh?\), for the purpose of remembering history, for time series - to predict the next X based on the previous Y.**

**\(What is RNN?\) by Andrej Karpathy -** [**The Unreasonable Effectiveness of Recurrent Neural Networks**](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)**, basically a lot of information about RNNs and their usage cases 1 to N = frame captioning**

* **N to 1 = classification**
* **N to N = predict frames in a movie**
* **N\2 with time delay to N\2 = predict supply and demand**
* **Vanishing gradient is 100 times worse.**
* **Gate networks like LSTM solves vanishing gradient.**

**\(how to initialize?\)** [**Benchmarking RNN networks for text**](https://danijar.com/benchmarking-recurrent-networks-for-language-modeling) **- don't worry about initialization, use normalization and GRU for big networks.**

**\*\* Experimental improvements:**

[**Ref**](https://arxiv.org/abs/1709.02755) **- ”Simplified RNN, with pytorch implementation” - changing the underlying mechanism in RNNs for the purpose of parallelizing calculation, seems to work nicely in terms of speed, not sure about state of the art results.** [**Controversy regarding said work**](https://www.facebook.com/cho.k.hyun/posts/10208564563785149)**, author claims he already mentioned these ideas \(QRNN\)** [**first**](https://www.reddit.com/r/MachineLearning/comments/6zduh2/r_170902755_training_rnns_as_fast_as_cnns/dmv9gnh/)**, a year before, however it seems like his ideas have also been reviewed as** [**incremental**](https://openreview.net/forum?id=H1zJ-v5xl) **\(PixelRNN\). Its probably best to read all 3 papers in chronological order and use the most optimal solution.**

[**RNNCELLS - recurrent shop**](https://github.com/farizrahman4u/recurrentshop)**, enables you to build complex rnns with keras. Details on their significance are inside the link**

**Masking for RNNs - the ideas is simple, we want to use variable length inputs, although rnns do use that, they require a fixed size input. So masking of 1’s and 0’s will help it understand the real size or where the information is in the input. Motivation: Padded inputs are going to contribute to our loss and we dont want that.**

[**Source 1**](https://www.quora.com/What-is-masking-in-a-recurrent-neural-network-RNN)**,** [**source 2**](https://r2rt.com/recurrent-neural-networks-in-tensorflow-iii-variable-length-sequences.html)**,**

**Visual attention RNNS - Same idea as masking but on a window-based cnn.** [**Paper** ](https://papers.nips.cc/paper/5542-recurrent-models-of-visual-attention.pdf)

**LSTM**

* [**The best, hands down, lstm post out there**](https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21)
* **LSTM -** [**what is?**](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) **the first reference for LSTM on the web, but you should know the background before reading.** 
* ![](https://lh3.googleusercontent.com/7KJz_beT-3kClxvDJHNVZP4gEMtn0oUK08yzh_foRMwqjtrWh8EpC3Yp9oCmH0LOcBzBbA-8E9D-4Dd1TXdWipGjSHXW0GjgMBo4gs-1f8XLpXRjnwN29zhzpJPe2uKIyNXkkqy-)
* [**Hidden state vs cell state**](https://www.quora.com/How-is-the-hidden-state-h-different-from-the-memory-c-in-an-LSTM-cell) **- you have to understand this concept before you dive in. i.e, Hidden state is overall state of what we have seen so far. Cell state is selective memory of the past. The hidden state \(h\) carries the information about what an RNN cell has seen over the time and supply it to the present time such that a loss function is not just dependent upon the data it is seeing in this time instant, but also, data it has seen historically.**
* [**Illustrated rnn lstm gru**](https://towardsdatascience.com/animated-rnn-lstm-and-gru-ef124d06cf45)
* [**Paper**](https://arxiv.org/pdf/1503.04069.pdf) **- a comparison of many LSTMs variants and they are pretty much the same performance wise**
* [**Paper**](https://arxiv.org/pdf/1503.04069.pdf)  **- comparison of lstm variants, vanilla is mostly the best, forget and output gates are the most important in terms of performance. Other conclusions in the paper..**
* **Master on** [**unrolling RNN’s introductory post**](https://machinelearningmastery.com/rnn-unrolling/)
* **Mastery on** [**under/over fitting lstms**](https://machinelearningmastery.com/diagnose-overfitting-underfitting-lstm-models/) **- but makes sense for all types of networks**
* **Mastery on r**[**eturn\_sequence and return\_state in keras LSTM**](https://machinelearningmastery.com/return-sequences-and-return-states-for-lstms-in-keras/)
  * **That return sequences return the hidden state output for each input time step.**
  * **That return state returns the hidden state output and cell state for the last input time step.**
  * **That return sequences and return state can be used at the same time.**
* **Mastery on** [**understanding stateful vs stateless**](https://machinelearningmastery.com/understanding-stateful-lstm-recurrent-neural-networks-python-keras/)**,** [**stateful stateless for time series**](https://machinelearningmastery.com/stateful-stateless-lstm-time-series-forecasting-python/)
* **Mastery on** [**timedistributed layer**](https://machinelearningmastery.com/timedistributed-layer-for-long-short-term-memory-networks-in-python/) **and seq2seq**
  * **TimeDistributed Layer - used to connect 3d inputs from lstms to dense layers, in order to utilize the time element. Otherwise it gets flattened when the connection is direct, nulling the lstm purpose. Note: nice trick that doesn't increase the dense layer structure multiplied by the number of dense neurons. It loops for each time step! I.e., The TimeDistributed achieves this trick by applying the same Dense layer \(same weights\) to the LSTMs outputs for one time step at a time. In this way, the output layer only needs one connection to each LSTM unit \(plus one bias\).**

**For this reason, the number of training epochs needs to be increased to account for the smaller network capacity. I doubled it from 500 to 1000 to match the first one-to-one example**

* **Sequence Learning Problem**
* **One-to-One LSTM for Sequence Prediction**
* **Many-to-One LSTM for Sequence Prediction \(without TimeDistributed\)**
* **Many-to-Many LSTM for Sequence Prediction \(with TimeDistributed\)**
* * **Mastery on** [**wrapping cnn-lstm with time distributed**](https://machinelearningmastery.com/cnn-long-short-term-memory-networks/)**, as a whole model wrap, or on every layer in the model which is equivalent and preferred.**
* **Master on** [**visual examples**](https://machinelearningmastery.com/sequence-prediction/) **for sequence prediction**
* **Unread - sentiment classification of IMDB movies using** [**Keras and LSTM** ](http://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/)
* [**Very important - how to interpret LSTM neurons in keras**](https://yerevann.github.io/2017/06/27/interpreting-neurons-in-an-LSTM-network/)
* [**LSTM for time-series**](http://www.jakob-aungiers.com/articles/a/LSTM-Neural-Network-for-Time-Series-Prediction) **- \(jakob\) single point prediction, sequence prediction and shifted-sequence prediction with code.**

**Stateful vs Stateless: crucial for understanding how to leverage LSTM networks:**

1. [**A good description on what it is and how to use it.**](https://groups.google.com/forum/#!topic/keras-users/l1RV_tthjoY)
2. [**ML mastery**](https://machinelearningmastery.com/stateful-stateless-lstm-time-series-forecasting-python/) _\*\*_
3. [**Philippe remy**](http://philipperemy.github.io/keras-stateful-lstm/) **on stateful vs stateless, intuition mostly with code, but not 100% clear**

**Machine Learning mastery:**

[**A good tutorial on LSTM:**](https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/) **important notes:**

**1. Scale to -1,1, because the internal activation in the lstm cell is tanh.**

**2.**[**stateful**](https://machinelearningmastery.com/understanding-stateful-lstm-recurrent-neural-networks-python-keras/) **- True, needs to reset internal states, False =stateless. Great info & results** [**HERE**](https://machinelearningmastery.com/stateful-stateless-lstm-time-series-forecasting-python/)**, with seeding, with training resets \(and not\) and predicting resets \(and not\) - note: empirically matching the shampoo input, network config, etc.**

[**Another explanation/tutorial about stateful lstm, should be thorough.**](http://philipperemy.github.io/keras-stateful-lstm/)

**3.** [**what is return\_sequence, return\_states**](https://machinelearningmastery.com/return-sequences-and-return-states-for-lstms-in-keras/)**, and how to use each one and both at the same time.**

**Return\_sequence is needed for stacked LSTM layers.**

**4.**[**stacked LSTM**](https://machinelearningmastery.com/stacked-long-short-term-memory-networks/) **- each layer has represents a higher level of abstraction in TIME!**

[**Keras Input shape**](https://stackoverflow.com/questions/44747343/keras-input-explanation-input-shape-units-batch-size-dim-etc) **- a good explanation about differences between input\_shape, dim, and what is. Additionally about layer calculation of inputs and output based on input shape, and sequence model vs API model.**

**A** [**comparison**](https://danijar.com/language-modeling-with-layer-norm-and-gru/) **of LSTM/GRU/MGU with batch normalization and various initializations, GRu/Xavier/Batch are the best and recommended for RNN**

[**Benchmarking LSTM variants**](http://proceedings.mlr.press/v37/jozefowicz15.pdf)**: - it looks like LSTM and GRU are competitive to mutation \(i believe its only in pytorch\) adding a bias to LSTM works \(a bias of 1 as recommended in the** [**paper**](https://pdfs.semanticscholar.org/1154/0131eae85b2e11d53df7f1360eeb6476e7f4.pdf)**\), but generally speaking there is no conclusive empirical evidence that says one type of network is better than the other for all tests, but the mutated networks tend to win over lstm\gru variants.**

[**BIAS 1 in keras**](https://keras.io/layers/recurrent/#lstm) **- unit\_forget\_bias: Boolean. If True, add 1 to the bias of the forget gate at initializationSetting it to true will also force bias\_initializer="zeros". This is recommended in** [**Jozefowicz et al.**](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)

![](https://lh3.googleusercontent.com/fiS0-IpAswRrHvrmnmFA-rrfd1h0rzoxmiZlPHQmBpcOrkbQXxzm9Z-5Q5HPsW26D_qsxzmriQ2tMWCmlG6jP0W5riP-yKjME1vjX-empGjSgycHKyxZZgt916uqiUmuLk4aecb2)

[**Validation\_split arg**](https://www.quora.com/What-is-the-importance-of-the-validation-split-variable-in-Keras) **- The validation split variable in Keras is a value between \[0..1\]. Keras proportionally split your training set by the value of the variable. The first set is used for training and the 2nd set for validation after each epoch.**

**This is a nice helper add-on by Keras, and most other Keras examples you have seen the training and test set was passed into the fit method, after you have manually made the split. The value of having a validation set is significant and is a vital step to understand how well your model is training. Ideally on a curve you want your training accuracy to be close to your validation curve, and the moment your validation curve falls below your training curve the alarm bells should go off and your model is probably busy over-fitting.**

**Keras is a wonderful framework for deep learning, and there are many different ways of doing things with plenty of helpers.**

[**Return\_sequence**](https://stackoverflow.com/questions/42755820/how-to-use-return-sequences-option-and-timedistributed-layer-in-keras)**: unclear.**

[**Sequence.pad\_sequences**](https://stackoverflow.com/questions/42943291/what-does-keras-io-preprocessing-sequence-pad-sequences-do) **- using maxlength it will either pad with zero if smaller than, or truncate it if bigger.**

[**Using batch size for LSTM in Keras**](https://machinelearningmastery.com/use-different-batch-sizes-training-predicting-python-keras/)

**Imbalanced classes? Use** [**class\_weight**](https://stackoverflow.com/questions/43459317/keras-class-weight-vs-sample-weights-in-the-fit-generator)**s, another explanation** [**here**](https://stackoverflow.com/questions/43459317/keras-class-weight-vs-sample-weights-in-the-fit-generator) **about class\_weights and sample\_weights.**

**SKlearn Formula for balanced class weights and why it works,** [**example**](https://stackoverflow.com/questions/50152377/in-sklearn-logistic-regression-class-balanced-helps-run-the-model-with-imbala/50154388)

[**number of units in LSTM**](https://www.quora.com/What-is-the-meaning-of-%E2%80%9CThe-number-of-units-in-the-LSTM-cell)

[**Calculate how many params are in an LSTM layer?** ](https://stackoverflow.com/questions/38080035/how-to-calculate-the-number-of-parameters-of-an-lstm-network)

![](https://lh5.googleusercontent.com/niwCPHMxrR83JzXNLWT8J4dr9S_GJ4_Z4SEDMwPQFv6OghMu9S2X2A5cy9wUwTnaAehXU18IIVM4s--tRnANN8AxnMUOogOt6WjF5azZc0ootq5EIHgj9hfxL253oMCWaAm8ftQj)

[**Understanding timedistributed in Keras**](https://machinelearningmastery.com/timedistributed-layer-for-long-short-term-memory-networks-in-python/)**, but with focus on lstm one to one, one to many and many to many - here the timedistributed is applying a dense layer to each output neuron from the lstm, which returned\_sequence = true for that purpose.**

**This tutorial clearly shows how to manipulate input construction, lstm output neurons and the target layer for the purpose of those three problems \(1:1, 1:m, m:m\).**

**BIDIRECTIONAL LSTM**

**\(what is?\) Wiki - The basic idea of BRNNs is to connect two hidden layers of opposite directions to the same output. By this structure, the output layer can get information from past and future states.**

**BRNN are especially useful when the context of the input is needed. For example, in handwriting recognition, the performance can be enhanced by knowledge of the letters located before and after the current letter.**

[**Another**](https://machinelearningmastery.com/develop-bidirectional-lstm-sequence-classification-python-keras/) **explanation- It involves duplicating the first recurrent layer in the network so that there are now two layers side-by-side, then providing the input sequence as-is as input to the first layer and providing a reversed copy of the input sequence to the second.**

**.. It allows you to specify the merge mode, that is how the forward and backward outputs should be combined before being passed on to the next layer. The options are:**

* **‘sum‘: The outputs are added together.**
* **‘mul‘: The outputs are multiplied together.**
* **‘concat‘: The outputs are concatenated together \(the default\), providing double the number of outputs to the next layer.**
* **‘ave‘: The average of the outputs is taken.**

**The default mode is to concatenate, and this is the method often used in studies of bidirectional LSTMs.**

[**Another simplified example**](https://stackoverflow.com/questions/43035827/whats-the-difference-between-a-bidirectional-lstm-and-an-lstm)

**BACK PROPAGATION**

[**A great Slide about back prop, on a simple 3 neuron network, with very easy to understand calculations.**](https://www.slideshare.net/AhmedGadFCIT/backpropagation-understanding-how-to-update-anns-weights-stepbystep)

**UNSUPERVISED LSTM**

1. [**Paper**](ftp://ftp.idsia.ch/pub/juergen/icann2001unsup.pdf)**,** [**paper2**](https://arxiv.org/pdf/1502.04681.pdf)**,** [**paper3**](https://arxiv.org/abs/1709.02081)
2. [**In keras**](https://www.reddit.com/r/MachineLearning/comments/4adrie/unsupervised_lstm_using_keras/)

**GRU**

[**A tutorial about GRU**](https://towardsdatascience.com/understanding-gru-networks-2ef37df6c9be) **- To solve the vanishing gradient problem of a standard RNN, GRU uses, so called, update gate and reset gate. Basically, these are two vectors which decide what information should be passed to the output. The special thing about them is that they can be trained to keep information from long ago, without washing it through time or remove information which is irrelevant to the prediction.**

1. **update gate helps the model to determine how much of the past information \(from previous time steps\) needs to be passed along to the future.**
2. **Reset gate essentially, this gate is used from the model to decide how much of the past information to forget.** 

**RECURRENT WEIGHTED AVERAGE \(RNN-WA\)**

**What is? \(a type of cell that converges to higher accuracy faster than LSTM.**

**it implements attention into the recurrent neural network:**

**1. the keras implementation is available at** [**https://github.com/keisuke-nakata/rwa**](https://github.com/keisuke-nakata/rwa) _\*\*_

**2. the whitepaper is at** [**https://arxiv.org/pdf/1703.01253.pdf**](https://arxiv.org/pdf/1703.01253.pdf)

![](https://lh6.googleusercontent.com/OgNIg0_EssPKTLuvrFf2cz3R89QeP4FYh7kLrk0J-_AIDjcgaVirW_d668aFDlPXW8mSF2CBtHDgCpiQoFDgc12bChOeePfbyWq1-ybMDdZSga6ezEdr16dKjiFEok8Oajn5XLFm)

**QRNN**

[**Potential competitor to the transformer**](https://towardsdatascience.com/qrnn-a-potential-competitor-to-the-transformer-86b5aef6c137)

### **GRAPH NEURAL NETWORKS \(GNN\)**

1. **\(amazing\)** [**Why i am luke warm about GNN’s**](https://www.singlelunch.com/2020/12/28/why-im-lukewarm-on-graph-neural-networks/) **- really good insight to what they do \(compressing data, vs adjacy graphs, vs graphs, high dim relations, etc.\)**
2. \(amazing\) [Graphical intro to GNNs](https://distill.pub/2021/gnn-intro/) 
3. [**Learning on graphs youtube - uriel singer**](https://www.youtube.com/watch?v=snLsWos_1WU&feature=youtu.be&fbclid=IwAR0JlvF9aPgKMmeh2zGr3l3j_8AebOTjknVGyMsz0Y2EvgcqrS0MmLkBTMU)
4. [**Benchmarking GNN’s, methodology, git, the works.**](https://graphdeeplearning.github.io/post/benchmarking-gnns/)
5. [**Awesome graph classification on github**](https://github.com/benedekrozemberczki/awesome-graph-classification)
6. **Octavian in medium on graphs,** [**A really good intro to graph networks, too long too summarize**](https://medium.com/octavian-ai/deep-learning-with-knowledge-graphs-3df0b469a61a)**, clever, mcgraph, regression, classification, embedding on graphs.** 
7. [**Application of graph networks**](https://towardsdatascience.com/https-medium-com-aishwaryajadhav-applications-of-graph-neural-networks-1420576be574) _\*\*_
8. [**Recommender systems using GNN**](https://towardsdatascience.com/recommender-systems-applying-graph-and-nlp-techniques-619dbedd9ecc)**, w2v, pytorch w2v, networkx, sparse matrices, matrix factorization, dictionary optimization, part 1 here** [**\(how to find product relations, important: creating negative samples\)**](https://eugeneyan.com/2020/01/06/recommender-systems-beyond-the-user-item-matrix)
9. [**Transformers are GNN**](https://towardsdatascience.com/transformers-are-graph-neural-networks-bca9f75412aa)**, original:** [**Transformers are graphs, not the typical embedding on a graph, but a more holistic approach to understanding text as a graph.**](https://thegradient.pub/transformers-are-graph-neural-networks/)
10. [**Cnn for graphs**](https://towardsdatascience.com/how-to-do-deep-learning-on-graphs-with-graph-convolutional-networks-62acf5b143d0)
11. [**Staring with gnn**](https://medium.com/octavian-ai/how-to-get-started-with-machine-learning-on-graphs-7f0795c83763)
12. **Really good -** [**Basics deep walk and graphsage**](https://towardsdatascience.com/a-gentle-introduction-to-graph-neural-network-basics-deepwalk-and-graphsage-db5d540d50b3) _\*\*_
13. [**Application of gnn**](https://towardsdatascience.com/https-medium-com-aishwaryajadhav-applications-of-graph-neural-networks-1420576be574)
14. **Michael Bronstein’s** [**Central page for Graph deep learning articles on Medium**](https://towardsdatascience.com/graph-deep-learning/home) **\(worth reading\)**
15. [**GAT graphi attention networks**](https://petar-v.com/GAT/)**, paper, examples - The graph attentional layer utilised throughout these networks is computationally efficient \(does not require costly matrix operations, and is parallelizable across all nodes in the graph\), allows for \(implicitly\) assigning different importances to different nodes within a neighborhood while dealing with different sized neighborhoods, and does not depend on knowing the entire graph structure upfront—thus addressing many of the theoretical issues with approaches.** 
16. **Medium on** [**Intro, basics, deep walk, graph sage**](https://towardsdatascience.com/a-gentle-introduction-to-graph-neural-network-basics-deepwalk-and-graphsage-db5d540d50b3)

**Deep walk**

1. [**Git**](https://github.com/phanein/deepwalk)
2. [**Paper**](https://arxiv.org/abs/1403.6652)
3. [**Medium**](https://medium.com/@_init_/an-illustrated-explanation-of-using-skipgram-to-encode-the-structure-of-a-graph-deepwalk-6220e304d71b)  **and medium on** [**W2v, deep walk, graph2vec, n2v**](https://towardsdatascience.com/graph-embeddings-the-summary-cc6075aba007)

**Node2vec**

1. [**Git**](https://github.com/eliorc/node2vec)
2. [**Stanford**](https://snap.stanford.edu/node2vec/)
3. [**Elior on medium**](https://towardsdatascience.com/node2vec-embeddings-for-graph-data-32a866340fef)**,** [**youtube**](https://www.youtube.com/watch?v=828rZgV9t1g)
4. [**Paper**](https://cs.stanford.edu/~jure/pubs/node2vec-kdd16.pdf)

**Graphsage**

1. [**medium**](https://towardsdatascience.com/a-gentle-introduction-to-graph-neural-network-basics-deepwalk-and-graphsage-db5d540d50b3)

**SDNE - structural deep network embedding**

1. [**medium**](https://towardsdatascience.com/graph-embeddings-the-summary-cc6075aba007)

**Diff2vec**

1. [**Git**](https://github.com/benedekrozemberczki/diff2vec)
2. ![](https://lh6.googleusercontent.com/otaXffQv-FribLSm922jhO-904l0ZHD4QcWRJ0dgc7u4vW0HMP1cGP-QU63ohhJSLiUxpz5DTB9L6DsK1ettM0S1MRg76sZZhEjzezQpTDDrrXI6pnh5B-2aRrA8FxJrAJK_fufn)

**Splitter**

**,** [**git**](https://github.com/benedekrozemberczki/Splitter)**,** [**paper**](http://epasto.org/papers/www2019splitter.pdf)**, “Is a Single Embedding Enough? Learning Node Representations that Capture Multiple Social Contexts”**

**Recent interest in graph embedding methods has focused on learning a single representation for each node in the graph. But can nodes really be best described by a single vector representation? In this work, we propose a method for learning multiple representations of the nodes in a graph \(e.g., the users of a social network\). Based on a principled decomposition of the ego-network, each representation encodes the role of the node in a different local community in which the nodes participate. These representations allow for improved reconstruction of the nuanced relationships that occur in the graph a phenomenon that we illustrate through state-of-the-art results on link prediction tasks on a variety of graphs, reducing the error by up to 90%. In addition, we show that these embeddings allow for effective visual analysis of the learned community structure.**

![](https://lh3.googleusercontent.com/ZWvxCQ72uAo6J-nr2uojE4KYzqOvgm3dzzXSuKlP0nbry-qFhEbQVZIG4om_SPLZpWZti3--aG1a6dYmOMnot--vFx0dnimMZDLz4LrjJQkRgAZY8ZospzEPKA9MrW__We61ylD9)

![](https://lh5.googleusercontent.com/asBPQZ90fcBXUYlz3tT2uV2LbELCjHVm56nhjbvRFuW7UXFBDX8fy353dF_6_OFGHo7ioBmFOl5wwxsyfSJHhA2LIOS0LkOTIdI23WnTjHFIf-PFdr6tp5RG_GaJF7BACv2RrJcK)

**16.** [**Self clustering graph embeddings**](https://github.com/benedekrozemberczki/GEMSEC)

![](https://lh5.googleusercontent.com/xLcNkor6PpkcSUl1sW9Ws36NxIrNr9kmdoBuhlPYnfCKlrC7zkaJwNIlSlIBDiXvL9OPi62lQ8q3ZA6oLXr_pJfUJvUTmelHnEy7z2hivhQJxQN4Ppz8ZRCErtlLQzROyIoyZaV-)

**17.** [**Walklets**](https://github.com/benedekrozemberczki/walklets?fbclid=IwAR2ymD7lbgP_sUde5UvKGZp7TYYYmACMFJS6UGNjqW29ethONHy7ibmDL0Q)**, similar to deep walk with node skips. - lots of improvements, works in scale due to lower size representations, improves results, etc.**

**Nodevectors**

[**Git**](https://github.com/VHRanger/nodevectors)**, The fastest network node embeddings in the west**![](https://lh3.googleusercontent.com/DwKfPhonL4At5xRePfv77SdSDjSZBYo_Z0Qm1hAFNpLLEYtiGMQhN8QPLO_5tNRr0NYvg3JRyYEECOUhjJkR6sK77k0M-Z1VVYcEwbBLU7cLqjlVN41IV5nGPt1yX8kYP-NlrqO9)

### **SIGNAL PROCESSING NN \(FFT, WAVELETS, SHAPELETS\)**

1. [**Fourier Transform**](https://www.youtube.com/watch?v=spUNpyF58BY) **- decomposing frequencies** 
2. [**WAVELETS On youtube \(4 videos\)**](https://www.youtube.com/watch?v=QX1-xGVFqmw)**:**
   1. [**used for denoising**](https://www.youtube.com/watch?v=veCvP1mYpww)**, compression, detect edges, detect features with various orientation, analyse signal power, detect and localize transients, change points in time series data and detect optimal signal representation \(peaks etc\) of time freq analysis of images and data.**
   2. **Can also be used to** [**reconstruct time and frequencies**](https://www.youtube.com/watch?v=veCvP1mYpww)**, analyse images in space, frequencies, orientation, identifying coherent time oscillation in time series**
   3. **Analyse signal variability and correlation** 
   4. 

### **HIERARCHICAL RNN**

1. [**githubcode**](https://github.com/keras-team/keras/blob/master/examples/mnist_hierarchical_rnn.py)

### **NN-Sequence Analysis**

**\(did not read\)** [**A causal framework for explaining the predictions of black-box sequence-to-sequence models**](http://people.csail.mit.edu/tommi/papers/AlvJaa_EMNLP2017.pdf) **- can this be applied to other time series prediction?**

### **SIAMESE NETWORKS \(one shot\)**

1. [**Siamese CNN, learns a similarity between images, not to classify**](https://medium.com/predict/face-recognition-from-scratch-using-siamese-networks-and-tensorflow-df03e32f8cd0)
2. [**Visual tracking, explains contrastive and triplet loss**](https://medium.com/intel-student-ambassadors/siamese-networks-for-visual-tracking-96262eaaba77)
3. [**One shot learning, very thorough, baseline vs siamese**](https://towardsdatascience.com/one-shot-learning-with-siamese-networks-using-keras-17f34e75bb3d)
4. [**What is triplet loss**](https://towardsdatascience.com/siamese-network-triplet-loss-b4ca82c1aec8)
5. **MULTI NETWORKS**
6. [**Google whitening black boxes using multi nets, segmentation and classification**](https://medium.com/health-ai/google-deepmind-might-have-just-solved-the-black-box-problem-in-medical-ai-3ed8bc21f636)

### **OPTIMIZING NEURAL NETS**

**PRUNING / KNOWLEDGE DISTILLATION / LOTTERY TICKET**

1. [**Awesome Knowledge distillation** ](https://github.com/dkozlov/awesome-knowledge-distillation)
2. **Lottery ticket** 
   1. [**1**](https://towardsdatascience.com/breaking-down-the-lottery-ticket-hypothesis-ca1c053b3e58)**,** [**2**](https://arxiv.org/pdf/1803.03635.pdf)**-paper**
   2. [**Uber on Lottery ticket, masking weights retraining**](https://eng.uber.com/deconstructing-lottery-tickets/?utm_campaign=the_algorithm.unpaid.engagement&utm_source=hs_email&utm_medium=email&utm_content=72562707&_hsenc=p2ANqtz--3mi4IwIFWZsW8UaWeuiv2nCzXDXattjRENzdKT-7J6wc7ftReuDXbn39mxCnX5y18o3z7cXfxPXQgysBMJnVnfeYpHg&_hsmi=72562707)
   3. [**Facebook article and paper**](https://ai.facebook.com/blog/understanding-the-generalization-of-lottery-tickets-in-neural-networks)
3. [**Knowledge distillation 1**](https://medium.com/neuralmachine/knowledge-distillation-dc241d7c2322)**,** [**2**](https://towardsdatascience.com/knowledge-distillation-a-technique-developed-for-compacting-and-accelerating-neural-nets-732098cde690)**,** [**3**](https://medium.com/neuralmachine/knowledge-distillation-dc241d7c2322)
4. [**Pruning 1**](https://towardsdatascience.com/scooping-into-model-pruning-in-deep-learning-da92217b84ac)**,** [**2**](https://towardsdatascience.com/pruning-deep-neural-network-56cae1ec5505)
5. [**Teacher-student knowledge distillation**](https://towardsdatascience.com/model-distillation-and-compression-for-recommender-systems-in-pytorch-5d81c0f2c0ec) **focusing on Knowledge & Ranking distillation**

![](https://lh4.googleusercontent.com/dau-y87nrdDTAGDgPw5H5ETsdU9TIum7G3vdYpdABd44O-iE3Ghp2V2Ymihe3vSowLWU5wzxD27W_N8lExEQ0ISQAKgAnbbj6SiYQ3RDXPONGJFDj-OO-XE5Bjtc-1uPfEEjUDVb)

1. [**Deep network compression using teacher student**](https://github.com/Zhengyu-Li/Deep-Network-Compression-based-on-Student-Teacher-Network-)
2. [**Lottery ticket on BERT**](https://thegradient.pub/when-bert-plays-the-lottery-all-tickets-are-winning/)**, magnitude vs structured pruning on a various metrics, i.e., LT works on bert. The classical Lottery Ticket Hypothesis was mostly tested with unstructured pruning, specifically magnitude pruning \(m-pruning\) where the weights with the lowest magnitude are pruned irrespective of their position in the model. We iteratively prune 10% of the least magnitude weights across the entire fine-tuned model \(except the embeddings\) and evaluate on dev set, for as long as the performance of the pruned subnetwork is above 90% of the full model.**

**We also experiment with structured pruning \(s-pruning\) of entire components of BERT architecture based on their importance scores: specifically, we 'remove' the least important self-attention heads and MLPs by applying a mask. In each iteration, we prune 10% of BERT heads and 1 MLP, for as long as the performance of the pruned subnetwork is above 90% of the full model. To determine which heads/MLPs to prune, we use a loss-based approximation: the importance scores proposed by** [**Michel, Levy and Neubig \(2019\)**](https://thegradient.pub/when-bert-plays-the-lottery-all-tickets-are-winning/#RefMichel) **for self-attention heads, which we extend to MLPs. Please see our paper and the original formulation for more details.**

1. **Troubleshooting Neural Nets**

**\(**[**37 reasons**](https://blog.slavv.com/37-reasons-why-your-neural-network-is-not-working-4020854bd607?fref=gc&dti=543283492502370)**,** [**10 more**](http://theorangeduck.com/page/neural-network-not-working?utm_campaign=Revue%20newsletter&utm_medium=Newsletter&utm_source=The%20Wild%20Week%20in%20AI&fref=gc&dti=543283492502370)**\) - copy pasted and rewritten here for convenience, it's pretty thorough, but long and extensive, you should have some sort of intuition and not go through all of these. The following list is has much more insight and information in the article itself.**

**The author of the original article suggests to turn everything off and then start building your network step by step, i.e., “a divide and conquer ‘debug’ method”.**

**Dataset Issues**

**1. Check your input data - for stupid mistakes**

**2. Try random input - if the error behaves the same on random data, there is a problem in the net. Debug layer by layer**

**3. Check the data loader - input data is possibly broken. Check the input layer.**

**4. Make sure input is connected to output - do samples have correct labels, even after shuffling?**

**5. Is the relationship between input and output too random? - the input are not sufficiently related to the output. Its pretty amorphic, just look at the data.**

**6. Is there too much noise in the dataset? - badly labelled datasets.**

**7. Shuffle the dataset - useful to counteract order in the DS, always shuffle input and labels together.**

**8. Reduce class imbalance - imbalance datasets may add a bias to class prediction. Balance your class, your loss, do something.**

**9. Do you have enough training examples? - training from scratch? ~1000 images per class, ~probably similar numbers for other types of samples.**

**10. Make sure your batches don’t contain a single label - this is probably something you wont notice and will waste a lot of time figuring out! In certain cases shuffle the DS to prevent batches from having the same label.**

**11. Reduce batch size -** [**This paper**](https://arxiv.org/abs/1609.04836) **points out that having a very large batch can reduce the generalization ability of the model. However, please note that I found other references that claim a too small batch will impact performance.**

**12. Test on well known Datasets**

**Data Normalization/Augmentation**

**12. Standardize the features - zero mean and unit variance, sounds like normalization.**

**13. Do you have too much data augmentation?**

**Augmentation has a regularizing effect. Too much of this combined with other forms of regularization \(weight L2, dropout, etc.\) can cause the net to underfit.**

**14. Check the preprocessing of your pretrained model - with a pretrained model make sure your input data is similar in range\[0, 1\], \[-1, 1\] or \[0, 255\]?**

**15. Check the preprocessing for train/validation/test set - CS231n points out a** [**common pitfall**](http://cs231n.github.io/neural-networks-2/#datapre)**:**

**Any preprocessing should be computed ONLY on the training data, then applied to val/test**

**Implementation issues**

**16. Try solving a simpler version of the problem -divide and conquer prediction, i.e., class and box coordinates, just use one.**

**17. Look for correct loss “at chance” - calculat loss for chance level, i.e 10% baseline is -ln\(0.1\) = 2.3 Softmax loss is the negative log probability. Afterwards increase regularization strength which should increase the loss.**

**18. Check your custom loss function.**

**19. Verify loss input - parameter confusion.**

**20. Adjust loss weights -If your loss is composed of several smaller loss functions, make sure their magnitude relative to each is correct. This might involve testing different combinations of loss weights.**

**21. Monitor other metrics -like accuracy.**

**22. Test any custom layers, debugging them.**

**23. Check for “frozen” layers or variables - accidentally frozen?**

**24. Increase network size - more layers, more neurons.**

**25. Check for hidden dimension errors - confusion due to vectors -&gt;\(64, 64, 64\)**

**26. Explore Gradient checking -does your backprop work for custon gradients?** [**1**](http://ufldl.stanford.edu/tutorial/supervised/DebuggingGradientChecking/) **\*\*\[**2**\]\(**[http://cs231n.github.io/neural-networks-3/\#gradcheck](http://cs231n.github.io/neural-networks-3/#gradcheck)**\) \*\***[**3**](https://www.coursera.org/learn/machine-learning/lecture/Y3s6r/gradient-checking)**.**

**Training issues**

**27. Solve for a really small dataset - can you generalize on 2 samples?**

**28. Check weights initialization -** [**Xavier**](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) **or** [**He**](http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf) **or forget about it for networks such as RNN.**

**29. Change your hyperparameters - grid search**

**30. Reduce regularization - too much may underfit, try for dropout, batch norm, weight, bias , L2.**

**31. Give it more training time as long as the loss is decreasing.**

**32. Switch from Train to Test mode - not clear.**

**33. Visualize the training - activations, weights, layer updates, biases.** [**Tensorboard**](https://www.tensorflow.org/get_started/summaries_and_tensorboard) **and** [**Crayon**](https://github.com/torrvision/crayon)**. Tips on** [**Deeplearning4j**](https://deeplearning4j.org/visualization#usingui)**. Expect gaussian distribution for weights, biases start at 0 and end up almost gaussian. Keep an eye out for parameters that are diverging to +/- infinity. Keep an eye out for biases that become very large. This can sometimes occur in the output layer for classification if the distribution of classes is very imbalanced.**

**34. Try a different optimizer, Check this** [**excellent post**](http://ruder.io/optimizing-gradient-descent/) **about gradient descent optimizers.**

**35. Exploding / Vanishing gradients - Gradient clipping may help. Tips on:** [**Deeplearning4j**](https://deeplearning4j.org/visualization#usingui)**: “A good standard deviation for the activations is on the order of 0.5 to 2.0. Significantly outside of this range may indicate vanishing or exploding activations.”**

**36. Increase/Decrease Learning Rate, or use adaptive learning**

**37. Overcoming NaNs, big issue for RNN - decrease LR,** [**how to deal with NaNs**](http://russellsstewart.com/notes/0.html)**. evaluate layer by layer, why does it appear.**

![Neural Network Graph With Shared Inputs](https://lh3.googleusercontent.com/ir9UIqpUmXMNRkrggrIrxHiRj3bOTRKCacXJ6iIaK39u-xEv8LPpAh7aycuMAWObzQl3-hcGZfZO21FzXDDzSPfhwNZh69Zookju_IYOueTB-SDi1VY4NeAYG5ZcT1_BkKhtTdps)

## **EMBEDDINGS**

**\(amazing\)** [**embeddings from the ground up singlelunch**](https://www.singlelunch.com/2020/02/16/embeddings-from-the-ground-up/)

### **VECTOR SIMILARITY SEARCH**

1. [**Faiss**](https://github.com/facebookresearch/faiss) **- a library for efficient similarity search** 
2. [**Benchmarking**](https://github.com/erikbern/ann-benchmarks) **- complete with almost everything imaginable** 
3. [**Singlestore**](https://www.singlestore.com/solutions/predictive-ml-ai/)
4. **Elastic search -** [**dense vector**](https://www.elastic.co/guide/en/elasticsearch/reference/7.6/query-dsl-script-score-query.html#vector-functions)
5. **Google cloud vertex matching engine** [**NN search**](https://cloud.google.com/blog/products/ai-machine-learning/vertex-matching-engine-blazing-fast-and-massively-scalable-nearest-neighbor-search)
   1. **search**
      1. **Recommendation engines**
      2. **Search engines**
      3. **Ad targeting systems**
      4. **Image classification or image search**
      5. **Text classification**
      6. **Question answering**
      7. **Chat bots**
   2. **Features**
      1. **Low latency**
      2. **High recall**
      3. **managed** 
      4. **Filtering**
      5. **scale**
6. **Pinecone - managed**  [**vector similarity search**](https://www.pinecone.io/) **- Pinecone is a fully managed vector database that makes it easy to add vector search to production applications. No more hassles of benchmarking and tuning algorithms or building and maintaining infrastructure for vector search.**
7. [**Nmslib**](https://github.com/nmslib/nmslib) **\(**[**benchmarked**](https://github.com/erikbern/ann-benchmarks) **- Benchmarks of approximate nearest neighbor libraries in Python\) is a Non-Metric Space Library \(NMSLIB\): An efficient similarity search library and a toolkit for evaluation of k-NN methods for generic non-metric spaces.**
8. **scann,** 
9. [**Vespa.ai**](https://vespa.ai/) **- Make AI-driven decisions using your data, in real time. At any scale, with unbeatable performance**
10. [**Weaviate**](https://www.semi.technology/developers/weaviate/current/) **- Weaviate is an** [**open source**](https://github.com/semi-technologies/weaviate) **vector search engine and vector database. Weaviate uses machine learning to vectorize and store data, and to find answers to natural language queries, or any other media type.** 
11. [**Neural Search with BERT and Solr**](https://dmitry-kan.medium.com/list/vector-search-e9b564d14274) **- Indexing BERT vector data in Solr and searching with full traversal**
12. [**Fun With Apache Lucene and BERT Embeddings**](https://medium.com/swlh/fun-with-apache-lucene-and-bert-embeddings-c2c496baa559) **- This post goes much deeper -- to the similarity search algorithm on Apache Lucene level. It upgrades the code from 6.6 to 8.0**
13. [**Speeding up BERT Search in Elasticsearch**](https://towardsdatascience.com/speeding-up-bert-search-in-elasticsearch-750f1f34f455) **- Neural Search in Elasticsearch: from vanilla to KNN to hardware acceleration**
14. [**Ask Me Anything about Vector Search**](https://towardsdatascience.com/ask-me-anything-about-vector-search-4252a01f3889) **- In the Ask Me Anything: Vector Search! session Max Irwin and Dmitry Kan discussed major topics of vector search, ranging from its areas of applicability to comparing it to good ol’ sparse search (TF-IDF/BM25), to its readiness for prime time and what specific engineering elements need further tuning before offering this to users.**
15. [**Search with BERT vectors in Solr and Elasticsearch**](https://github.com/DmitryKey/bert-solr-search) **- GitHub repository used for experiments with Solr and Elasticsearch using DBPedia abstracts comparing Solr, vanilla Elasticsearch, elastiknn enhanced Elasticsearch, OpenSearch, and GSI APU**

### **TOOLS**

**FLAIR**

1. **Name-Entity Recognition \(NER\): It can recognise whether a word represents a person, location or names in the text.**
2. **Parts-of-Speech Tagging \(PoS\): Tags all the words in the given text as to which “part of speech” they belong to.**
3. **Text Classification: Classifying text based on the criteria \(labels\)**
4. **Training Custom Models: Making our own custom models.**
5. **It comprises of popular and state-of-the-art word embeddings, such as GloVe, BERT, ELMo, Character Embeddings, etc. There are very easy to use thanks to the Flair API**
6. **Flair’s interface allows us to combine different word embeddings and use them to embed documents. This in turn leads to a significant uptick in results**
7. **‘Flair Embedding’ is the signature embedding provided within the Flair library. It is powered by contextual string embeddings. We’ll understand this concept in detail in the next section**
8. **Flair supports a number of languages – and is always looking to add new ones**

**HUGGING FACE**

1. [**Git**](https://github.com/huggingface/transformers)
2. 1. [**Hugging face pytorch transformers**](https://github.com/huggingface/pytorch-transformers)
3. [**Hugging face nlp pretrained**](https://huggingface.co/models?search=Helsinki-NLP%2Fopus-mt&fbclid=IwAR0YN7qn9uTlCeBOZw4jzWgq9IXq_9ju1ww_rVL-f1fa9EjlSP50q05QcmU)
4. [**hugging face on emotions**](https://medium.com/huggingface/understanding-emotions-from-keras-to-pytorch-3ccb61d5a983)
   1. **how to make a custom pyTorch LSTM with custom activation functions,**
   2. **how the PackedSequence object works and is built,**
   3. **how to convert an attention layer from Keras to pyTorch,**
   4. **how to load your data in pyTorch: DataSets and smart Batching,**
   5. **how to reproduce Keras weights initialization in pyTorch.**
5. **A** [**thorough tutorial on bert**](http://mccormickml.com/2019/07/22/BERT-fine-tuning/)**, fine tuning using hugging face transformers package.** [**Code**](https://colab.research.google.com/drive/1Y4o3jh3ZH70tl6mCd76vz_IxX23biCPP)

**Youtube** [**ep1**](https://www.youtube.com/watch?v=FKlPCK1uFrc)**,** [**2**](https://www.youtube.com/watch?v=zJW57aCBCTk)**,** [**3**](https://www.youtube.com/watch?v=x66kkDnbzi4)**,** [**3b**](https://www.youtube.com/watch?v=Hnvb9b7a_Ps)**,**

### **LANGUAGE EMBEDDINGS**

![](https://lh6.googleusercontent.com/aibqScGzh66aJK9E5Rho61W_pX8Kw82vJrrUkvRZrRN7vaRBOWnDOz0k29szquWdU3i4cwFFUj6b4-rPZvU2AUIlP5ouxwS7Kq2RwxDwFxtm9fpJZcnVXCMHY3SJ43FEsWj_GTcT)

1. **History:**
   1. [**Google’s intro to transformers and multi-head self attention**](https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html)
   2. [**How self attention and relative positioning work**](https://medium.com/@_init_/how-self-attention-with-relative-position-representations-works-28173b8c245a) **\(great!\)**
      1. **Rnns are sequential, same word in diff position will have diff encoding due to the input from the previous word, which is inherently different.**
      2. **Attention without positional! Will have distinct \(Same\) encoding.**
      3. **Relative look at a window around each word and adds a distance vector in terms of how many words are before and after, which fixes the problem.**
      4. ![](https://lh3.googleusercontent.com/XmFsG2XDB2sLXNkRwmsc90iPfXPBWDgr4AzO-u8lejinMcwb5XzTppAZ5oekBjUjIsJ8u8IBA83Z31bP3rgMjdkvq0qZAteTE2VvxSOa79AUH4KqsRQb0w1Eworanxxm7zFuo494)
      5. ![](https://lh6.googleusercontent.com/JNAgD9NAJQzXCtfZ3ekWddZ1m8nzgMwXoqoQ3rjLsKfHl2NdqVrdrYexDnXCUzik2ZYalllJhm7Hp5Zl1_L5EHumNGN0NAfFHHH0RM6gqBZc4bPkg7Bd4D5ea5gmV1_hXtMXW_9K)
      6. **The authors hypothesized that precise relative position information is not useful beyond a certain distance.**
      7. **Clipping the maximum distance enables the model to generalize to sequence lengths not seen during training.**
      8. 
2. [**From bert to albert**](https://medium.com/@hamdan.hussam/from-bert-to-albert-pre-trained-langaug-models-5865aa5c3762)
3. [**All the latest buzz algos**](https://www.topbots.com/most-important-ai-nlp-research/#ai-nlp-paper-2018-12)
4. **A** [**Summary of them**](https://www.topbots.com/ai-nlp-research-pretrained-language-models/?utm_source=facebook&utm_medium=group_post&utm_campaign=pretrained&fbclid=IwAR0smqf8qanfMayo4fRH2hFuc5LYA8-Bn5oEp-xedKcRR43QsqXIelIAzEE)
5. [**8 pretrained language embeddings**](https://www.analyticsvidhya.com/blog/2019/03/pretrained-models-get-started-nlp/)
6. [**Hugging face pytorch transformers**](https://github.com/huggingface/pytorch-transformers)
7. [**Hugging face nlp pretrained**](https://huggingface.co/models?search=Helsinki-NLP%2Fopus-mt&fbclid=IwAR0YN7qn9uTlCeBOZw4jzWgq9IXq_9ju1ww_rVL-f1fa9EjlSP50q05QcmU)

**Language modelling**

1. [**Ruder on language modelling as the next imagenet**](http://ruder.io/nlp-imagenet/) **- Language modelling, the last approach mentioned, has been shown to capture many facets of language relevant for downstream tasks, such as** [**long-term dependencies**](https://arxiv.org/abs/1611.01368) **,** [**hierarchical relations**](https://arxiv.org/abs/1803.11138) **, and** [**sentiment**](https://arxiv.org/abs/1704.01444) **. Compared to related unsupervised tasks such as skip-thoughts and autoencoding,** [**language modelling performs better on syntactic tasks even with less training data**](https://openreview.net/forum?id=BJeYYeaVJ7)**.**
2. **A** [**tutorial**](https://blog.myyellowroad.com/unsupervised-sentence-representation-with-deep-learning-104b90079a93) **about w2v skipthought - with code!, specifically language modelling here is important - Our second method is training a language model to represent our sentences. A language model describes the probability of a text existing in a language. For example, the sentence “I like eating bananas” would be more probable than “I like eating convolutions.” We train a language model by slicing windows of n words and predicting what the next word will be in the text**
3. [**Unread - universal language model fine tuning for text-classification**](https://arxiv.org/abs/1801.06146)
4. **ELMO -** [**medium**](https://towardsdatascience.com/beyond-word-embeddings-part-2-word-vectors-nlp-modeling-from-bow-to-bert-4ebd4711d0ec)
5. [**Bert**](https://arxiv.org/abs/1810.04805v1) **\*\*\[**python git**\]\(**[https://github.com/CyberZHG/keras-bert](https://github.com/CyberZHG/keras-bert)**\)**- We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT representations can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications. BERT is conceptually simple and empirically powerful. It obtains new state-of-the-art results on eleven natural language processing tasks.\*\* ![](https://lh4.googleusercontent.com/anFY63RxhdYt82bb_XUGDLRUmj2vuR1I0iJye66cOqgC2gQegXVf2ibkC64LRPIfgUj8Brl7VYUFfxw3gG0KBnwTuqJ2NCohd6mi9YzCkZmHGuDz1QxXl7JUtMv5BpiBJXGnC-Zc)
6. [**Open.ai on language modelling**](https://blog.openai.com/language-unsupervised/) **- We’ve obtained state-of-the-art results on a suite of diverse language tasks with a scalable, task-agnostic system, which we’re also releasing. Our approach is a combination of two existing ideas:** [**transformers**](https://arxiv.org/abs/1706.03762) **and** [**unsupervised pre-training**](https://arxiv.org/abs/1511.01432)**.** [**READ PAPER**](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)**,** [**VIEW CODE**](https://github.com/openai/finetune-transformer-lm)**.**
7. **Scikit-learn inspired model finetuning for natural language processing.**

[**finetune**](https://finetune.indico.io/#module-finetune) **ships with a pre-trained language model from** [**“Improving Language Understanding by Generative Pre-Training”**](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) **and builds off the** [**OpenAI/finetune-language-model repository**](https://github.com/openai/finetune-transformer-lm)**.**

1. **Did not read -** [**The annotated Transformer**](http://nlp.seas.harvard.edu/2018/04/03/attention.html?fbclid=IwAR2_ZOfUfXcto70apLdT_StObPwatYHNRPP4OlktcmGfj9uPLhgsZPsAXzE) **- jupyter on transformer with annotation**
2. **Medium on** [**Dissecting Bert**](https://medium.com/dissecting-bert/dissecting-bert-part-1-d3c3d495cdb3)**,** [**appendix**](https://medium.com/dissecting-bert/dissecting-bert-appendix-the-decoder-3b86f66b0e5f)
3. [**Medium on distilling 6 patterns from bert**](https://towardsdatascience.com/deconstructing-bert-distilling-6-patterns-from-100-million-parameters-b49113672f77)

**Embedding spaces**

1. [**A good overview of sentence embedding methods**](http://mlexplained.com/2017/12/28/an-overview-of-sentence-embedding-methods/) **- w2v ft s2v skip, d2v**
2. [**A very good overview of word embeddings**](http://sanjaymeena.io/tech/word-embeddings/)
3. [**Intro to word embeddings - lots of images**](https://www.springboard.com/blog/introduction-word-embeddings/)
4. [**A very long and extensive thesis about embeddings**](http://ad-publications.informatik.uni-freiburg.de/theses/Bachelor_Jon_Ezeiza_2017.pdf)
5. [**Sent2vec by gensim**](https://rare-technologies.com/sent2vec-an-unsupervised-approach-towards-learning-sentence-embeddings/) **- sentence embedding is defined as the average of the source word embeddings of its constituent words. This model is furthermore augmented by also learning source embeddings for not only unigrams but also n-grams of words present in each sentence, and averaging the n-gram embeddings along with the words**
6. [**Sent2vec vs fasttext - with info about s2v parameters**](https://github.com/epfml/sent2vec/issues/19)
7. [**Wordrank vs fasttext vs w2v comparison**](https://en.wikipedia.org/wiki/Automatic_summarization#TextRank_and_LexRank) **- the better word similarity algorithm**
8. [**W2v vs glove vs sppmi vs svd by gensim**](https://rare-technologies.com/making-sense-of-word2vec/)
9. [**Medium on a gentle intro to d2v**](https://medium.com/scaleabout/a-gentle-introduction-to-doc2vec-db3e8c0cce5e)
10. [**Doc2vec tutorial by gensim**](https://rare-technologies.com/doc2vec-tutorial/) **-  Doc2vec \(aka paragraph2vec, aka sentence embeddings\) modifies the word2vec algorithm to unsupervised learning of continuous representations for larger blocks of text, such as sentences, paragraphs or entire documents. - Most importantly this tutorial has crucial information about the implementation parameters that should be read before using it.**
11. [**Git for word embeddings - taken from mastery’s nlp course**](https://github.com/IshayTelavivi/nlp_crash_course)
12. [**Skip-thought -**](http://mlexplained.com/2017/12/28/an-overview-of-sentence-embedding-methods/) **\*\*\[**git**\]\(**[https://github.com/ryankiros/skip-thoughts](https://github.com/ryankiros/skip-thoughts)**\)**- Where word2vec attempts to predict surrounding words from certain words in a sentence, skip-thought vector extends this idea to sentences: it predicts surrounding sentences from a given sentence. NOTE: Unlike the other methods, skip-thought vectors require the sentences to be ordered in a semantically meaningful way. This makes this method difficult to use for domains such as social media text, where each snippet of text exists in isolation.\*\*
13. [**Fastsent**](http://mlexplained.com/2017/12/28/an-overview-of-sentence-embedding-methods/) **- Skip-thought vectors are slow to train. FastSent attempts to remedy this inefficiency while expanding on the core idea of skip-thought: that predicting surrounding sentences is a powerful way to obtain distributed representations. Formally, FastSent represents sentences as the simple sum of its word embeddings, making training efficient. The word embeddings are learned so that the inner product between the sentence embedding and the word embeddings of surrounding sentences is maximized. NOTE: FastSent sacrifices word order for the sake of efficiency, which can be a large disadvantage depending on the use-case.**
14. **Weighted sum of words - In this method, each word vector is weighted by the factor** ![\frac{a}{a + p\(w\)} ](https://lh3.googleusercontent.com/p6He6GoHCb-yA8QgNrn4eIrWTa5i_7lolQyY6EplDa1l7bmf1IF0y-eNuGOPfMfLKMkyw5qOpkwzoejmNB44Fg9fIwt4bIPkYOSWT7r50wdgdhT7qUiDwyNh1toe21CQFolKp5py) **where** ![a ](https://lh5.googleusercontent.com/qeqpAm9JfrNP8TnZzbUsMBKcsv2v-ZpZbmbM01Uf22HVUBcZMwa5nseCQMW_XGYNZQQJ1HvYqOMwGfaL_5NDbrOa_aJTAsA3JdoHEUaB9XMq-sDUKtR348dq6TJuHEr05hetP0-7) **is a hyperparameter and** ![p\(w\) ](https://lh6.googleusercontent.com/cPiXavxPJ8voQb9UE8cmzaNsV_dMWFvG1E5SYJGGm6QrMiA9X_uNUWjb45L96WWhAKLxvLIF4oOXI2q0m5NQRNNzKgBrogEubQDN5bDXPw66sSOyfdx3dzGxjSvwdGYgpAy60B33) **is the \(estimated\) word frequency. This is similar to tf-idf weighting, where more frequent terms are weighted downNOTE: Word order and surrounding sentences are ignored as well, limiting the information that is encoded.**
15. [**Infersent by facebook**](https://github.com/facebookresearch/InferSent) **-** [**paper**](https://arxiv.org/abs/1705.02364)  **InferSent is a sentence embeddings method that provides semantic representations for English sentences. It is trained on natural language inference data and generalizes well to many different tasks. ABSTRACT: we show how universal sentence representations trained using the supervised data of the Stanford Natural Language Inference datasets can consistently outperform unsupervised methods like SkipThought vectors on a wide range of transfer tasks. Much like how computer vision uses ImageNet to obtain features, which can then be transferred to other tasks, our work tends to indicate the suitability of natural language inference for transfer learning to other NLP tasks.** 
16. [**Universal sentence encoder - google**](https://tfhub.dev/google/universal-sentence-encoder/1)  **-** [**notebook**](https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/semantic_similarity_with_tf_hub_universal_encoder.ipynb#scrollTo=8OKy8WhnKRe_)**,** [**git**](https://github.com/tensorflow/hub/blob/master/examples/colab/semantic_similarity_with_tf_hub_universal_encoder.ipynb) **The Universal Sentence Encoder encodes text into high dimensional vectors that can be used for text classification, semantic similarity, clustering and other natural language tasks. The model is trained and optimized for greater-than-word length text, such as sentences, phrases or short paragraphs. It is trained on a variety of data sources and a variety of tasks with the aim of dynamically accommodating a wide variety of natural language understanding tasks. The input is variable length English text and the output is a 512 dimensional vector. We apply this model to the** [**STS benchmark**](http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark) **for semantic similarity, and the results can be seen in the** [**example notebook**](https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/semantic_similarity_with_tf_hub_universal_encoder.ipynb) **made available. The universal-sentence-encoder model is trained with a deep averaging network \(DAN\) encoder.**
17. [**Multi language universal sentence encoder**](https://ai.googleblog.com/2019/07/multilingual-universal-sentence-encoder.html?fbclid=IwAR2fubNOwrxWWxYous7IyQCJ3_bY0UAdAYO_yuWONMv-aV3o8hDckSS3FCE) **- no hebrew**
18. **Pair2vec -** [**paper**](https://arxiv.org/abs/1810.08854) **- paper proposes new methods for learning and using embeddings of word pairs that implicitly represent background knowledge about such relationships. I.e., using p2v information with existing models to increase performance. Experiments show that our pair embeddings can complement individual word embeddings, and that they are perhaps capturing information that eludes the traditional interpretation of the Distributional Hypothesis**
19. [**Fast text python tutorial**](http://ai.intelligentonlinetools.com/ml/fasttext-word-embeddings-text-classification-python-mlp/)

### **Cat2vec**

1. **Part1:** [**Label encoder/ ordinal, One hot, one hot with a rare bucket, hash**](https://blog.myyellowroad.com/using-categorical-data-in-machine-learning-with-python-from-dummy-variables-to-deep-category-66041f734512)
2. [**Part2: cat2vec using w2v**](https://blog.myyellowroad.com/using-categorical-data-in-machine-learning-with-python-from-dummy-variables-to-deep-category-42fd0a43b009)**, and entity embeddings for categorical data**

![](https://lh6.googleusercontent.com/BJjrzp0YPmsy2_OKecufELzNU_AO2I2kSAx9ekSbGmGYNJ27AGkbdhwPv45iMVub_6q0AHF91N6BYdxA4l-eAUspOIat-QMU8xHQrSYYpWmu7TEO8NmRPIcrPItwq1TgkJN-LTd3)

### **ENTITY EMBEDDINGS**

1. **Star -** [**General purpose embedding paper with code somewhere**](https://arxiv.org/pdf/1709.03856.pdf)
2. [**Using embeddings on tabular data, specifically categorical - introduction**](http://www.fast.ai/2018/04/29/categorical-embeddings/)**, using fastai without limiting ourselves to pytorch - the material from this post is covered in much more detail starting around 1:59:45 in** [**the Lesson 3 video**](http://course.fast.ai/lessons/lesson3.html) **and continuing in** [**Lesson 4**](http://course.fast.ai/lessons/lesson4.html) **of our free, online** [**Practical Deep Learning for Coders**](http://course.fast.ai/) **course. To see example code of how this approach can be used in practice, check out our** [**Lesson 3 jupyter notebook**](https://github.com/fastai/fastai/blob/master/courses/dl1/lesson3-rossman.ipynb)**. Perhaps Saturday and Sunday have similar behavior, and maybe Friday behaves like an average of a weekend and a weekday. Similarly, for zip codes, there may be patterns for zip codes that are geographically near each other, and for zip codes that are of similar socio-economic status. The jupyter notebook doesn't seem to have the embedding example they are talking about.**
3. [**Rossman on kaggle**](http://blog.kaggle.com/2016/01/22/rossmann-store-sales-winners-interview-3rd-place-cheng-gui/)**, used entity-embeddings,** [**here**](https://www.kaggle.com/c/rossmann-store-sales/discussion/17974)**,** [**github**](https://github.com/entron/entity-embedding-rossmann)**,** [**paper**](https://arxiv.org/abs/1604.06737)
4. [**Medium on rossman - good**](https://towardsdatascience.com/deep-learning-structured-data-8d6a278f3088)
5. [**Embedder**](https://github.com/dkn22/embedder) **- git code for a simplified entity embedding above.**
6. **Finally what they do is label encode each feature using labelEncoder into an int-based feature, then push each feature into its own embedding layer of size 1 with an embedding size defined by a rule of thumb \(so it seems\), merge all layers, train a synthetic regression/classification and grab the weights of the corresponding embedding layer.**
7. [**Entity2vec**](https://github.com/ot/entity2vec)
8. [**Categorical using keras**](https://medium.com/@satnalikamayank12/on-learning-embeddings-for-categorical-data-using-keras-165ff2773fc9)

### **ALL2VEC EMBEDDINGS**

1. [**ALL ???-2-VEC ideas**](https://github.com/MaxwellRebo/awesome-2vec)
2. **Fast.ai** [**post**](http://www.fast.ai/2018/04/29/categorical-embeddings/) **regarding embedding for tabular data, i.e., cont and categorical data**
3. [**Entity embedding for**](https://github.com/fastai/fastai/blob/master/courses/dl1/lesson3-rossman.ipynb) **categorical data +** [**notebook**](https://github.com/fastai/fastai/blob/master/courses/dl1/lesson3-rossman.ipynb)
4. [**Kaggle taxi competition + code**](http://blog.kaggle.com/2015/07/27/taxi-trajectory-winners-interview-1st-place-team-%F0%9F%9A%95/)
5. [**Ross man competition - entity embeddings, code missing**](http://blog.kaggle.com/2016/01/22/rossmann-store-sales-winners-interview-3rd-place-cheng-gui/) **+**[**alternative code** ](https://github.com/entron/entity-embedding-rossmann)
6. [**CODE TO CREATE EMBEDDINGS straight away, based onthe ideas by cheng guo in keras**](https://github.com/dkn22/embedder)
7. [**PIN2VEC - pinterest embeddings using the same idea**](https://medium.com/the-graph/applying-deep-learning-to-related-pins-a6fee3c92f5e)
8. [**Tweet2Vec**](https://github.com/soroushv/Tweet2Vec) **- code in theano,** [**paper**](https://dl.acm.org/citation.cfm?doid=2911451.2914762)**.**
9. [**Clustering**](https://github.com/svakulenk0/tweet2vec_clustering) **of tweet2vec,** [**paper**](https://arxiv.org/abs/1703.05123)
10. **Paper:** [**Character neural embeddings for tweet clustering**](https://arxiv.org/pdf/1703.05123.pdf)
11. **Diff2vec - might be useful on social network graphs,** [**paper**](http://homepages.inf.ed.ac.uk/s1668259/papers/sequence.pdf)**,** [**code**](https://github.com/benedekrozemberczki/diff2vec)
12. **emoji 2vec \(below\)**
13. [**Char2vec**](https://hackernoon.com/chars2vec-character-based-language-model-for-handling-real-world-texts-with-spelling-errors-and-a3e4053a147d) **\*\*\[**Git**\]\(**[https://github.com/IntuitionEngineeringTeam/chars2vec](https://github.com/IntuitionEngineeringTeam/chars2vec)**\)**, similarity measure for words with types. **\[ \*\***\]\([https://arxiv.org/abs/1708.00524](https://arxiv.org/abs/1708.00524)\)

**EMOJIS**

1. **1.** [**Deepmoji**](http://datadrivenjournalism.net/featured_projects/deepmoji_using_emojis_to_teach_ai_about_emotions)**,** 
2. [**hugging face on emotions**](https://medium.com/huggingface/understanding-emotions-from-keras-to-pytorch-3ccb61d5a983)
   1. **how to make a custom pyTorch LSTM with custom activation functions,**
   2. **how the PackedSequence object works and is built,**
   3. **how to convert an attention layer from Keras to pyTorch,**
   4. **how to load your data in pyTorch: DataSets and smart Batching,**
   5. **how to reproduce Keras weights initialization in pyTorch.**
3. [**Another great emoji paper, how to get vector representations from** ](https://aclweb.org/anthology/S18-1039)
4. [**3. What can we learn from emojis \(deep moji\)**](https://www.media.mit.edu/posts/what-can-we-learn-from-emojis/)
5. [**Learning millions of**](https://arxiv.org/pdf/1708.00524.pdf) **for emoji, sentiment, sarcasm,** [**medium**](https://medium.com/@bjarkefelbo/what-can-we-learn-from-emojis-6beb165a5ea0)
6. [**EMOJI2VEC**](https://tech.instacart.com/deep-learning-with-emojis-not-math-660ba1ad6cdc) **- medium article with keras code, a**[**nother paper on classifying tweets using emojis**](https://arxiv.org/abs/1708.00524)
7. [**Group2vec**](https://github.com/cerlymarco/MEDIUM_NoteBook/tree/master/Group2Vec) **git and** [**medium**](https://towardsdatascience.com/group2vec-for-advance-categorical-encoding-54dfc7a08349)**, which is a multi input embedding network using a-f below. plus two other methods that involve groupby and applying entropy and join/countvec per class. Really interesting**
   1. **Initialize embedding layers for each categorical input;**
   2. **For each category, compute dot-products among other embedding representations. These are our ‘groups’ at the categorical level;**
   3. **Summarize each ‘group’ adopting an average pooling;**
   4. **Concatenate ‘group’ averages;**
   5. **Apply regularization techniques such as BatchNormalization or Dropout;**
   6. **Output probabilities.**

### **WORD EMBEDDINGS**

1. [**Medium on Introduction into word embeddings, sentence embeddings, trends in the field.**](https://towardsdatascience.com/deep-transfer-learning-for-natural-language-processing-text-classification-with-universal-1a2c69e5baa9) **The Indian guy,** [**git**](https://nbviewer.jupyter.org/github/dipanjanS/data_science_for_all/blob/master/tds_deep_transfer_learning_nlp_classification/Deep%20Transfer%20Learning%20for%20NLP%20-%20Text%20Classification%20with%20Universal%20Embeddings.ipynb) **notebook,** [**his git**](https://github.com/dipanjanS)**,** 
   1. **Baseline Averaged Sentence Embeddings**
   2. **Doc2Vec**
   3. **Neural-Net Language Models \(Hands-on Demo!\)**
   4. **Skip-Thought Vectors**
   5. **Quick-Thought Vectors**
   6. **InferSent**
   7. **Universal Sentence Encoder**
2. [**Shay palachy on word embedding covering everything from bow to word/doc/sent/phrase.** ](https://medium.com/@shay.palachy/document-embedding-techniques-fed3e7a6a25d)
3. [**Another intro, not as good as the one above**](https://medium.com/huggingface/universal-word-sentence-embeddings-ce48ddc8fc3a)
4. [**Using sklearn vectorizer to create custom ones, i.e. a vectorizer that does preprocessing and tfidf and other things.**](https://towardsdatascience.com/hacking-scikit-learns-vectorizers-9ef26a7170af)
5. [**TFIDF - n-gram based top weighted tfidf words**](https://stackoverflow.com/questions/25217510/how-to-see-top-n-entries-of-term-document-matrix-after-tfidf-in-scikit-learn)
6. [**Gensim bi-gram phraser/phrases analyser/converter**](https://radimrehurek.com/gensim/models/phrases.html)
7. [**Countvectorizer, stemmer, lemmatization code tutorial**](https://medium.com/@rnbrown/more-nlp-with-sklearns-countvectorizer-add577a0b8c8)
8. [**Current 2018 best universal word and sentence embeddings -&gt; elmo**](https://medium.com/huggingface/universal-word-sentence-embeddings-ce48ddc8fc3a)
9. [**5-part series on word embeddings**](http://ruder.io/word-embeddings-1/)**,** [**part 2**](http://ruder.io/word-embeddings-softmax/index.html)**,** [**3**](http://ruder.io/secret-word2vec/index.html)**,** [**4 - cross lingual review**](http://ruder.io/cross-lingual-embeddings/index.html)**,** [**5-future trends**](http://ruder.io/word-embeddings-2017/index.html)
10. [**Word embedding posts**](https://datawarrior.wordpress.com/2016/05/15/word-embedding-algorithms/)
11. [**Facebook github for embedings called starspace**](https://github.com/facebookresearch/StarSpace)
12. [**Medium on Fast text / elmo etc**](https://medium.com/huggingface/universal-word-sentence-embeddings-ce48ddc8fc3a)

**FastText**

1. [**Fasttext - using fast text and upsampling/oversapmling on twitter data**](https://medium.com/@media_73863/fasttext-sentiment-analysis-for-tweets-a-straightforward-guide-9a8c070449a2)
2. [**A great youtube lecture 9m about ft, rarity, loss, class tree speedup**](https://www.youtube.com/watch?v=4l_At3oalzk) _\*\*_
3. [**A thorough tutorial about what is FT and how to use it, performance, pros and cons.**](https://www.analyticsvidhya.com/blog/2017/07/word-representations-text-classification-using-fasttext-nlp-facebook/)
4. [**Docs**](https://fasttext.cc/blog/2016/08/18/blog-post.html)
5. [**Medium: word embeddings with w2v and fast text in gensim**](https://towardsdatascience.com/word-embedding-with-word2vec-and-fasttext-a209c1d3e12c) **, data cleaning and word similarity**
6. **Gensim -** [**fasttext docs**](https://radimrehurek.com/gensim/models/fasttext.html)**, similarity, analogies**
7. [**Alternative to gensim**](https://github.com/plasticityai/magnitude#benchmarks-and-features) **- promises speed and out of the box support for many embeddings.**
8. [**Comparison of usage w2v fasttext**](http://ai.intelligentonlinetools.com/ml/fasttext-word-embeddings-text-classification-python-mlp/)
9. [**Using gensim fast text - recommendation against using the fb version**](https://blog.manash.me/how-to-use-pre-trained-word-vectors-from-facebooks-fasttext-a71e6d55f27)
10. [**A comparison of w2v vs ft using gensim**](https://rare-technologies.com/fasttext-and-gensim-word-embeddings/) **- “Word2Vec embeddings seem to be slightly better than fastText embeddings at the semantic tasks, while the fastText embeddings do significantly better on the syntactic analogies. Makes sense, since fastText embeddings are trained for understanding morphological nuances, and most of the syntactic analogies are morphology based.**
    1. [**Syntactic**](https://stackoverflow.com/questions/48356421/what-is-the-difference-between-syntactic-analogy-and-semantic-analogy) **means syntax, as in tasks that have to do with the structure of the sentence, these include tree parsing, POS tagging, usually they need less context and a shallower understanding of world knowledge**
    2. [**Semantic**](https://stackoverflow.com/questions/48356421/what-is-the-difference-between-syntactic-analogy-and-semantic-analogy) **tasks mean meaning related, a higher level of the language tree, these also typically involve a higher level understanding of the text and might involve tasks s.a. question answering, sentiment analysis, etc...**
    3. **As for analogies, he is referring to the mathematical operator like properties exhibited by word embedding, in this context a syntactic analogy would be related to plurals, tense or gender, those sort of things, and semantic analogy would be word meaning relationships s.a. man + queen = king, etc... See for instance** [**this article**](http://www.aclweb.org/anthology/W14-1618) **\(and many others\)**
11. [**Skip gram vs CBOW**](https://www.quora.com/What-are-the-continuous-bag-of-words-and-skip-gram-architectures)

![](https://lh5.googleusercontent.com/lnuntHia-uXCNiGbmw0bWYski3uPkeryHj3Rf8si9E9GUCyUi1aXsMv3sKgY_YLjqWbRRWjGLzCZymjWwRlMquDTsQdcd05PcSJ74ZEOmd1QW59SaZlC3XCzTGpyPdPjVDUljOvG)

1. [**Paper**](http://workshop.colips.org/dstc6/papers/track2_paper18_zhuang.pdf) **on fasttext vs glove vs w2v on a single DS, performance comparison. Ft wins by a small margin**
2. [**Medium on w2v/fast text ‘most similar’ words with code**](https://towardsdatascience.com/word-embedding-with-word2vec-and-fasttext-a209c1d3e12c)
3. [**keras/tf code for a fast text implementation**](http://debajyotidatta.github.io/nlp/deep/learning/word-embeddings/2016/09/28/fast-text-and-skip-gram/)
4. [**Medium on fast text and imbalance data**](https://medium.com/@yeyrama/fasttext-and-imbalanced-classification-1f9543f9e0ce)
5. **Medium on universal** [**Sentence encoder, w2v, Fast text for sentiment**](https://medium.com/@jatinmandav3/opinion-mining-sometimes-known-as-sentiment-analysis-or-emotion-ai-refers-to-the-use-of-natural-874f369194c0) **with code.**

**WORD2VEC**

1. **Monitor** [**train loss**](https://stackoverflow.com/questions/52038651/loss-does-not-decrease-during-training-word2vec-gensim) **using callbacks for word2vec**
2. **Cleaning datasets using weighted w2v sentence encoding, then pca and isolation forest to remove outlier sentences.**
3. [**Removing ‘gender bias using pair mean pca**](https://stackoverflow.com/questions/48019843/pca-on-word2vec-embeddings)
4. [**KPCA w2v approach on a very small dataset**](https://medium.com/@vishwanigupta/kpca-skip-gram-model-improving-word-embedding-a6a0cb7aad49)**,** [**similar git**](https://github.com/niitsuma/wordca) **for correspondence analysis,** [**paper**](https://arxiv.org/abs/1605.05087)
5. [**The best w2v/tfidf/bow/ embeddings post ever**](https://www.analyticsvidhya.com/blog/2017/06/word-embeddings-count-word2veec/)
6. [**Chris mccormick ml on w2v,**](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/) **\*\*\[**post \#2**\]\(**[http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/](http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/)**\)** - negative sampling “Negative sampling addresses this by having each training sample only modify a small percentage of the weights, rather than all of them. With negative sampling, we are instead going to randomly select just a small number of “negative” words \(let’s say 5\) to update the weights for. \(In this context, a “negative” word is one for which we want the network to output a 0 for\). We will also still update the weights for our “positive” word \(which is the word “quick” in our current example\). The “negative samples” \(that is, the 5 output words that we’ll train to output 0\) are chosen using a “unigram distribution”. Essentially, the probability for selecting a word as a negative sample is related to its frequency, with more frequent words being more likely to be selected as negative samples.\*\*
7. [**Chris mccormick on negative sampling and hierarchical soft max**](https://www.youtube.com/watch?v=pzyIWCelt_E) **training, i.e., huffman binary tree for the vocabulary, learning internal tree nodes ie.,,  the path as the probability vector instead of having len\(vocabulary\) neurons.**
8. [**Great W2V tutorial**](https://towardsdatascience.com/word2vec-skip-gram-model-part-1-intuition-78614e4d6e0b)
9. **Another** [**gensim-based w2v tutorial**](http://kavita-ganesan.com/gensim-word2vec-tutorial-starter-code/)**, with starter code and some usage examples of similarity**
10. [**Clustering using gensim word2vec**](http://ai.intelligentonlinetools.com/ml/k-means-clustering-example-word2vec/)
11. [**Yet another w2v medium explanation**](https://towardsdatascience.com/word-embeddings-exploration-explanation-and-exploitation-with-code-in-python-5dac99d5d795)
12. **Mean w2v**
13. **Sequential w2v embeddings.**
14. [**Negative sampling, why does it work in w2v - didnt read**](https://www.quora.com/How-does-negative-sampling-work-in-Word2vec-models)
15. [**Semantic contract using w2v/ft - he chose a good food category and selected words that worked best in order to find similar words to good bad etc. lior magen**](https://groups.google.com/forum/#!topic/gensim/wh7B00cc80w)
16. [**Semantic contract, syn-antonym DS, using w2v, a paper that i havent read**](http://anthology.aclweb.org/P16-2074) **yet but looks promising**
17. [**Amazing w2v most similar tutorial, examples for vectors, misspellings, semantic contrast  and relations that may or may not be captured in the network.**](https://quomodocumque.wordpress.com/2016/01/15/messing-around-with-word2vec/)
18. [**Followup tutorial about genderfying words using ‘he’ ‘she’ similarity**](https://quomodocumque.wordpress.com/2016/01/15/gendercycle-a-dynamical-system-on-words/)
19. [**W2v Analogies using predefined anthologies of the**](https://gist.github.com/kylemcdonald/9bedafead69145875b8c) **form x:y::a:b, plus code, plus insights of why it works and doesn't. presence : absence :: happy : unhappy absence : presence :: happy : proud abundant : scarce :: happy : glad refuse : accept :: happy : satisfied accurate : inaccurate :: happy : disappointed admit : deny :: happy : delighted never : always :: happy : Said\_Hirschbeck modern : ancient :: happy : ecstatic**
20. [**Nlpforhackers on bow, w2v embeddings with code on how to use**](https://nlpforhackers.io/word-embeddings/)
21. [**Hebrew word embeddings with w2v, ron shemesh, on wiki/twitter**](https://drive.google.com/drive/folders/1qBgdcXtGjse9Kq7k1wwMzD84HH_Z8aJt?fbclid=IwAR03PeUTGCgluILOQ6EaMR7AgkcRux5rs6Z8HEgWMRvFAwLGqb7-7bznbxM)

**GLOVE**

1. [**W2v vs glove vs fasttext, in terms of overfitting and what is the idea behind**](https://www.kaggle.com/sbongo/do-pretrained-embeddings-give-you-the-extra-edge)
2. [**W2v against glove performance**](http://dsnotes.com/post/glove-enwiki/) **comparison - glove wins in % and time.**
3. [**How glove and w2v work, but the following has a very good description**](https://geekyisawesome.blogspot.com/2017/03/word-embeddings-how-word2vec-and-glove.html) **- “GloVe takes a different approach. Instead of extracting the embeddings from a neural network that is designed to perform a surrogate task \(predicting neighbouring words\), the embeddings are optimized directly so that the dot product of two word vectors equals the log of the number of times the two words will occur near each other \(within 5 words for example\). For example if "dog" and "cat" occur near each other 10 times in a corpus, then vec\(dog\) dot vec\(cat\) = log\(10\). This forces the vectors to somehow encode the frequency distribution of which words occur near them.”**
4. [**Glove vs w2v, concise explanation**](https://www.quora.com/What-is-the-difference-between-fastText-and-GloVe/answer/Ajit-Rajasekharan)

### **SENTENCE EMBEDDING**

**Sense2vec**

1. [**Blog**](https://explosion.ai/blog/sense2vec-with-spacy)**,** [**github**](https://github.com/explosion/sense2vec)**: Using spacy or not, with w2v using POS/ENTITY TAGS to find similarities.based on reddit. “We follow Trask et al in adding part-of-speech tags and named entity labels to the tokens. Additionally, we merge named entities and base noun phrases into single tokens, so that they receive a single vector.”**
2. **&gt;&gt;&gt; model.similarity\('fair\_game\|NOUN', 'game\|NOUN'\) 0.034977455677555599 &gt;&gt;&gt; model.similarity\('multiplayer\_game\|NOUN', 'game\|NOUN'\) 0.54464530644393849**

**SENT2VEC aka “skip-thoughts”**

1. [**Gensim implementation of sent2vec**](https://rare-technologies.com/sent2vec-an-unsupervised-approach-towards-learning-sentence-embeddings/) **- usage examples, parallel training, a detailed comparison against gensim doc2vec**
2. [**Git implementation** ](https://github.com/ryankiros/skip-thoughts)
3. [**Another git - worked**](https://github.com/epfml/sent2vec)

**USE - Universal sentence encoder**

1. [**Git notebook, usage and sentence similarity benchmark / visualization**](https://github.com/tensorflow/hub/blob/master/examples/colab/semantic_similarity_with_tf_hub_universal_encoder.ipynb)

**BERT+W2V**

1. [**Sentence similarity**](https://towardsdatascience.com/how-to-compute-sentence-similarity-using-bert-and-word2vec-ab0663a5d64)

### **PARAGRAPH EMBEDDING**

1. [**Paragraph2VEC by stanford**](https://cs.stanford.edu/~quocle/paragraph_vector.pdf)

### **DOCUMENT EMBEDDING**

**DOC2VEC**

1. [**Shuffle before training each**](https://groups.google.com/forum/#!topic/gensim/IVQBUF5n6aI) **epoch in d2v in order to fight overfitting**

## **ATTENTION**

1. [**Illustrated attention-**](https://towardsdatascience.com/attn-illustrated-attention-5ec4ad276ee3) **AMAZING**
2. [**Illustrated self attention - great**](https://towardsdatascience.com/illustrated-self-attention-2d627e33b20a)
3. [**Jay alamar on attention, the first one is better.**](http://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)
4. [**Attention is all you need \(paper\)**](https://arxiv.org/abs/1706.03762?fbclid=IwAR3-gxVldr_xW0D9m6QvwyIV5vhvl-crVOc2kEI6HZskodJP678ynJKj1-o)
5. [**The annotated transformer - reviewing the paper** ](http://nlp.seas.harvard.edu/2018/04/03/attention.html?fbclid=IwAR2_ZOfUfXcto70apLdT_StObPwatYHNRPP4OlktcmGfj9uPLhgsZPsAXzE)
6. [**Lilian weng on attention**](https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html)**, self, soft vs hard, global vs local, neural turing machines, pointer networks, transformers, snail, self attention GAN.**
7. [**Understanding attention in rnns**](https://medium.com/datadriveninvestor/attention-in-rnns-321fbcd64f05)
8. [**Another good intro with gifs to attention**](https://towardsdatascience.com/attn-illustrated-attention-5ec4ad276ee3)
9. [**Clear insight to what attention is, a must read**](http://webcache.googleusercontent.com/search?q=cache:http://www.wildml.com/2016/01/attention-and-memory-in-deep-learning-and-nlp/)**!**
10. [**Transformer NN by google**](https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html) **- faster, better, more accurate**
11. [**Intuitive explanation to attention**](https://towardsdatascience.com/an-intuitive-explanation-of-self-attention-4f72709638e1)
12. [**Attention by vidhya**](https://www.analyticsvidhya.com/blog/2019/11/comprehensive-guide-attention-mechanism-deep-learning/)
13. [**Augmented rnns**](https://distill.pub/2016/augmented-rnns/) **- including turing / attention / adaptive computation time etc. general overview, not as clear as the one below.** ![](https://lh5.googleusercontent.com/5Cxd-2INMRXvO_TsSWX6cXtx_j4moRLqJAhRMdwYFFTDEkPZ6Ph_NbKbC4dVRAP-ctYMJGQdw5RrBO4eboM6FwA4W_U4Rmwv1_wmrG6SC-2dvdF94AnDnHXcBSqKBWZwByynuFGd)

![](https://lh3.googleusercontent.com/G7aL7maJfczYfXc-Zhg69IHeusTlQxE78b3TGHMd_nrH1f6JXUHosA3K6kg2dZEmOMqWWeF61qhcko260IGUBHUEshL2MW4ZnIh1deTY-OtXnsoluqlOmJsOGHBgsBLIRCKUbFZp)

1. [**A really good REVIEW on attention and its many forms, historical changes, etc**](https://medium.com/@joealato/attention-in-nlp-734c6fa9d983)
2. [**Medium on comparing cnn / rnn / han**](https://medium.com/jatana/report-on-text-classification-using-cnn-rnn-han-f0e887214d5f) **- will change on other data, my impression is that the data is too good in this article**
3. **Mastery on** [**rnn vs attention vs global attention**](https://machinelearningmastery.com/global-attention-for-encoder-decoder-recurrent-neural-networks/) **- a really unclear intro**
4. **Mastery on** [**attention**](https://machinelearningmastery.com/how-does-attention-work-in-encoder-decoder-recurrent-neural-networks/) **- this makes the whole process clear, scoring encoder vs decoder input outputs, normalizing them using softmax \(annotation weights\), multiplying score and the weight summed on all \(i.e., context vector\), and then we decode the context vector.**
   1. **Soft \(above\) and hard crisp attention**
   2. **Dropping the hidden output - HAN or AB BiLSTM**
   3. **Attention concat to input vec**
   4. **Global vs local attention**
5. **Mastery on** [**attention with lstm encoding / decoding**](https://machinelearningmastery.com/implementation-patterns-encoder-decoder-rnn-architecture-attention/) **- a theoretical discussion about many attention architectures. This adds make-sense information to everything above.** 
   1. **Encoder: The encoder is responsible for stepping through the input time steps and encoding the entire sequence into a fixed length vector called a context vector.**
   2. **Decoder: The decoder is responsible for stepping through the output time steps while reading from the context vector.**
   3. **A problem with the architecture is that performance is poor on long input or output sequences. The reason is believed to be because of the fixed-sized internal representation used by the encoder.**
      1. **Enc-decoder**
      2. **Recursive**
      3. **Enc-dev with recursive**![](https://lh6.googleusercontent.com/FcrjF3Fo9W5OeKP6E1YaGLDUBwdiB3AYr_r6-XdIO4g4t58RTe5eRFyIU5Jm3bk2mn1KOSxbPV-CF3mN6M7USCg4q_QYhwAoSoTxtJqvCzJPz0ABVwn3D3nQuXXuIWUvz8mNpMlt)
6. **Code on GIT:**
   1. **HAN -** [**GIT**](https://github.com/richliao/textClassifier)**,** [**paper**](https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf)
   2. [**Non penalized self attention**](https://github.com/uzaymacar/attention-mechanisms/blob/master/examples/sentiment_classification.py)
   3. **LSTM,** [**BiLSTM attention**](https://github.com/gentaiscool/lstm-attention)**,** [**paper**](https://arxiv.org/pdf/1805.12307.pdf)
   4. **Tushv89,** [**Keras layer attention implementation**](https://github.com/thushv89/attention_keras)
   5. **Richliao, hierarchical** [**Attention code for document classification using keras**](https://github.com/richliao/textClassifier/blob/master/textClassifierHATT.py)**,** [**blog**](https://richliao.github.io/supervised/classification/2016/12/26/textclassifier-HATN/)**,** [**group chatter**](https://groups.google.com/forum/#!topic/keras-users/IWK9opMFavQ)

**note: word level then sentence level embeddings.**

**figure= &gt;**

1. [**Self Attention pip for keras**](https://pypi.org/project/keras-self-attention/)**,** [**git**](https://github.com/CyberZHG/keras-self-attention)
2. [**Phillip remy on attention in keras, not a single layer, a few of them to make it.**](https://github.com/philipperemy/keras-attention-mechanism)
3. [**Self attention with relative positiion representations**](https://medium.com/@_init_/how-self-attention-with-relative-position-representations-works-28173b8c245a)
4. [**nMT - jointly learning to align and translate**](https://arxiv.org/abs/1409.0473) _\*\*_
5. [**Medium on attention plus code, comparison keras and pytorch**](https://medium.com/huggingface/understanding-emotions-from-keras-to-pytorch-3ccb61d5a983)

**BERT/ROBERTA**

1. [**Do attention heads in bert roberta track syntactic dependencies?**](https://medium.com/@phu_pmh/do-attention-heads-in-bert-track-syntactic-dependencies-81c8a9be311a) **- tl;dr: The attention weights between tokens in BERT/RoBERTa bear similarity to some syntactic dependency relations, but the results are less conclusive than we’d like as they don’t significantly outperform linguistically uninformed baselines for all types of dependency relations. In the case of MAX, our results indicate that specific heads in the BERT models may correspond to certain dependency relations, whereas for MST, we find much less support “generalist” heads whose attention weights correspond to a full syntactic dependency structure.**

**In both cases, the metrics do not appear to be representative of the extent of linguistic knowledge learned by the BERT models, based on their strong performance on many NLP tasks. Hence, our takeaway is that while we can tease out some structure from the attention weights of BERT models using the above methods, studying the attention weights alone is unlikely to give us the full picture of BERT’s strength processing natural language.**

1. **TRANSFORMERS**
2. [**Jay alammar on transformers**](http://jalammar.github.io/illustrated-transformer/) **\(amazing\)**
3. [**J.A on Bert Elmo**](http://jalammar.github.io/illustrated-bert/) **\(amazing\)** 
4. [**Jay alammar on a visual guide of bert for the first time**](http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/)
5. [**J.A on GPT2**](http://jalammar.github.io/illustrated-bert/)
6. [**Super fast transformers**](http://transformer)
7. [**A survey of long term context in transformers.**](https://www.pragmatic.ml/a-survey-of-methods-for-incorporating-long-term-context/)![](https://lh5.googleusercontent.com/KwcoMe_TwrkQYdxBuSZcd8HROwg3R5jB78OUMFd0Y7AwzL7R-4Wy_Eqfb0IfPyWvbIzCt_4NJjKPcjEjL8crrKcwXIgSxzq2KcCjbtzbJCq541efBKxF9swVTevNo97lJ5uBTIus)
8. [**Lilian Wang on the transformer family**](https://lilianweng.github.io/lil-log/2020/04/07/the-transformer-family.html) **\(seems like it is constantly updated\)**
9. ![](https://lh6.googleusercontent.com/t2dHec2TFYJhdgHx0k9tuxlIRJ1rqpKLzUfJFwrUOxp1ju-yxBzy7Ho1tx04GaZRUk-Op4FmA9wSFUhC9xsRxcbiX3jmV-Is39iXtpqNypOydikXkeZJJW-GfYOSLHhl6LyhW0e3)
10. **Hugging face,** [**encoders decoders in transformers for seq2seq**](https://medium.com/huggingface/encoder-decoders-in-transformers-a-hybrid-pre-trained-architecture-for-seq2seq-af4d7bf14bb8)
11. [**The annotated transformer**](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
12. [**Large memory layers with product keys**](https://arxiv.org/abs/1907.05242) **- This memory layer allows us to tackle very large scale language modeling tasks. In our experiments we consider a dataset with up to 30 billion words, and we plug our memory layer in a state-of-the-art transformer-based architecture. In particular, we found that a memory augmented model with only 12 layers outperforms a baseline transformer model with 24 layers, while being twice faster at inference time.** 
13. [**Adaptive sparse transformers**](https://arxiv.org/abs/1909.00015) **- This sparsity is accomplished by replacing softmax with** 

**α-entmax: a differentiable generalization of softmax that allows low-scoring words to receive precisely zero weight. Moreover, we derive a method to automatically learn the**

**α parameter -- which controls the shape and sparsity of**

**α-entmax -- allowing attention heads to choose between focused or spread-out behavior. Our adaptively sparse Transformer improves interpretability and head diversity when compared to softmax Transformers on machine translation datasets.**

### **ELMO**

1. [**Short tutorial on elmo, pretrained, new data, incremental\(finetune?\)**](https://github.com/PrashantRanjan09/Elmo-Tutorial)**,** [**using elmo  pretrained**](https://github.com/PrashantRanjan09/WordEmbeddings-Elmo-Fasttext-Word2Vec)
2. [**Why you cant use elmo to encode words \(contextualized\)**](https://github.com/allenai/allennlp/issues/1737)
3. [**Vidhya on elmo**](https://www.analyticsvidhya.com/blog/2019/03/learn-to-use-elmo-to-extract-features-from-text/) **- everything you want to know with code**
4. [**Sebastien ruder on language modeling embeddings for the purpose of transfer learning, ELMO, ULMFIT, open AI transformer, BILSTM,**](https://thegradient.pub/nlp-imagenet/)
5. [**Another good tutorial on elmo**](http://www.realworldnlpbook.com/blog/improving-sentiment-analyzer-using-elmo.html)**.**
6. [**ELMO**](https://allennlp.org/elmo)**,** [**tutorial**](https://allennlp.org/tutorials)**,** [**github**](https://allennlp.org/tutorials)
7. [**Elmo on google hub and code**](https://tfhub.dev/google/elmo/2)
8. [**How to use elmo embeddings, advice for word and sentence**](https://github.com/tensorflow/hub/issues/149)
9. [**Using elmo as a lambda embedding layer**](https://towardsdatascience.com/transfer-learning-using-elmo-embedding-c4a7e415103c)
10. [**Elmbo tutorial notebook**](https://github.com/sambit9238/Deep-Learning/blob/master/elmo_embedding_tfhub.ipynb)
11. [**Elmo code on git**](https://github.com/allenai/allennlp/blob/master/tutorials/how_to/elmo.md)
12. [**Elmo on keras using lambda**](https://towardsdatascience.com/elmo-helps-to-further-improve-your-word-embeddings-c6ed2c9df95f)
13. [**Elmo pretrained models for many languages**](https://github.com/HIT-SCIR/ELMoForManyLangs)**, for** [**russian**](http://docs.deeppavlov.ai/en/master/intro/pretrained_vectors.html) **too,** [**mean elmo**](https://stackoverflow.com/questions/53061423/how-to-represent-elmo-embeddings-as-a-1d-array/53088523)
14. [**Ari’s intro on word embeddings part 2, has elmo and some bert**](https://towardsdatascience.com/beyond-word-embeddings-part-2-word-vectors-nlp-modeling-from-bow-to-bert-4ebd4711d0ec)
15. [**Mean elmo**](https://www.analyticsvidhya.com/blog/2019/03/learn-to-use-elmo-to-extract-features-from-text/?utm_source=facebook.com&utm_medium=social&fbclid=IwAR24LwsmhUJshC7gk3P9RIIACCyYYcjlYMa_NbgdzcNBBhD7g38FM2KTA-Q)**, batches, with code and linear regression i**
16. [**Elmo projected using TSNE - grouping are not semantically similar**](https://towardsdatascience.com/elmo-contextual-language-embedding-335de2268604)

### **ULMFIT**

1. [**Tutorial and code by vidhya**](https://www.analyticsvidhya.com/blog/2018/11/tutorial-text-classification-ulmfit-fastai-library/)**,** [**medium**](https://medium.com/analytics-vidhya/tutorial-on-text-classification-nlp-using-ulmfit-and-fastai-library-in-python-2f15a2aac065)
2. [**Paper**](https://arxiv.org/abs/1801.06146)
3. [**Ruder on transfer learning**](http://ruder.io/nlp-imagenet/)
4. [**Medium on how - unclear**](https://blog.frame.ai/learning-more-with-less-1e618a5aa160)
5. [**Fast NLP on how**](http://nlp.fast.ai/classification/2018/05/15/introducting-ulmfit.html)
6. [**Paper: ulmfit**](https://arxiv.org/abs/1801.06146)
7. [**Fast.ai on ulmfit**](http://nlp.fast.ai/category/classification.html)**,** [**this too**](https://github.com/fastai/fastai/blob/c502f12fa0c766dda6c2740b2d3823e2deb363f9/nbs/examples/ulmfit.ipynb)
8. [**Vidhya on ulmfit using fastai**](https://www.analyticsvidhya.com/blog/2018/11/tutorial-text-classification-ulmfit-fastai-library/?utm_source=facebook.com&fbclid=IwAR0ghBUHEphXrSRZZfkbEOklY1RtveC7XG3I48eH_LNAfCnRQzgraw-AZWs)
9. [**Medium on ulmfit**](https://towardsdatascience.com/explainable-data-efficient-text-classification-888cc7a1af05)
10. [**Building blocks of ulm fit**](https://medium.com/mlreview/understanding-building-blocks-of-ulmfit-818d3775325b)
11. [**Applying ulmfit on entity level sentiment analysis using business news artcles**](https://github.com/jannenev/ulmfit-language-model)
12. [**Understanding language modelling using Ulmfit, fine tuning etc**](https://towardsdatascience.com/understanding-language-modelling-nlp-part-1-ulmfit-b557a63a672b)
13. [**Vidhaya on ulmfit  + colab**](https://www.analyticsvidhya.com/blog/2018/11/tutorial-text-classification-ulmfit-fastai-library/) **“The one cycle policy provides some form of regularisation”,  if you wish to know more about one cycle policy, then feel free to refer to this excellent paper by Leslie Smith – “**[**A disciplined approach to neural network hyper-parameters: Part 1 — learning rate, batch size, momentum, and weight decay**](https://arxiv.org/abs/1803.09820)**”.**

### **BERT**

1. [**The BERT PAPER**](https://arxiv.org/pdf/1810.04805.pdf)
   1. [**Prerequisite about transformers and attention - this is not enough**](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
   2. [**Embeddings using bert in python**](https://hackerstreak.com/word-embeddings-using-bert-in-python/?fbclid=IwAR1sQDbxgCekqsFZBjZ6VAHYDUk41ijgvwNu_oAXJpgAdWG0KrMAPhePEF4) **- using bert as a service to encode 1024 vectors and do cosine similarity**
   3. [**Identifying the right meaning with bert**](https://towardsdatascience.com/identifying-the-right-meaning-of-the-words-using-bert-817eef2ac1f0) **- the idea is to classify the word duck into one of three meanings using bert embeddings, which promise contextualized embeddings. I.e., to duck, the Duck, etc**![](https://lh5.googleusercontent.com/WnEaYRk3za14yoiPr0dxf7f3D4iPdmNoLPnQaFi9V94oBd38mTsLvAbqLHeNYsobJmy415hWgGSoMBPrcoIXIJkwK2xHF9QHWO5vKQGI2BEA_7aQQAppHQeYePFUewj4EQRjlpaF)
   4. [**Google neural machine translation \(attention\) - too long**](https://arxiv.org/pdf/1609.08144.pdf)
2. [**What is bert**](https://towardsdatascience.com/breaking-bert-down-430461f60efb)
3. **\(amazing\) Deconstructing bert** 
   1. **I found some fairly distinctive and surprisingly intuitive attention patterns. Below I identify six key patterns and for each one I show visualizations for a particular layer / head that exhibited the pattern.**
   2. [**part 1**](https://towardsdatascience.com/deconstructing-bert-distilling-6-patterns-from-100-million-parameters-b49113672f77) **- attention to the next/previous/ identical/related \(same and other sentences\), other words predictive of a word, delimeters tokens** 
   3. **\(good\)** [**Deconstructing bert part 2**](https://towardsdatascience.com/deconstructing-bert-part-2-visualizing-the-inner-workings-of-attention-60a16d86b5c1) **- looking at the visualization and attention heads, focusing on Delimiter attention, bag of words attention, next word attention - patterns.**
4. [**Bert demystified**](https://medium.com/@_init_/why-bert-has-3-embedding-layers-and-their-implementation-details-9c261108e28a) **\(read this first!\)**
5. [**Read this after**](https://towardsdatascience.com/understanding-bert-is-it-a-game-changer-in-nlp-7cca943cf3ad)**, the most coherent explanation on bert, 15% masked word prediction and next sentence prediction. Roberta, xlm bert, albert, distilibert.**
6. **A** [**thorough tutorial on bert**](http://mccormickml.com/2019/07/22/BERT-fine-tuning/)**, fine tuning using hugging face transformers package.** [**Code**](https://colab.research.google.com/drive/1Y4o3jh3ZH70tl6mCd76vz_IxX23biCPP)

**Youtube** [**ep1**](https://www.youtube.com/watch?v=FKlPCK1uFrc)**,** [**2**](https://www.youtube.com/watch?v=zJW57aCBCTk)**,** [**3**](https://www.youtube.com/watch?v=x66kkDnbzi4)**,** [**3b**](https://www.youtube.com/watch?v=Hnvb9b7a_Ps)**,**

1. [**How to train bert**](https://medium.com/@vineet.mundhra/loading-bert-with-tensorflow-hub-7f5a1c722565) **from scratch using TF, with \[CLS\] \[SEP\] etc**
2. [**Extending a vocabulary for bert, another kind of transfer learning.**](https://towardsdatascience.com/3-ways-to-make-new-language-models-f3642e3a4816)
3. [**Bert tutorial**](http://mccormickml.com/2019/07/22/BERT-fine-tuning/?fbclid=IwAR3TBQSjq3lcWa2gH3gn2mpBcn3vLKCD-pvpHGue33Cs59RQAz34dPHaXys)**, on fine tuning, some talk on from scratch and probably not discussed about using embeddings as input**
4. [**Bert for summarization thread**](https://github.com/google-research/bert/issues/352)
5. [**Bert on logs**](https://medium.com/rapids-ai/cybert-28b35a4c81c4)**, feature names as labels, finetune bert, predict.**
6. [**Bert scikit wrapper for pipelines**](https://towardsdatascience.com/build-a-bert-sci-kit-transformer-59d60ddd54a5)
7. [**What is bert not good at, also refer to the cited paper**](https://towardsdatascience.com/bert-is-not-good-at-7b1ca64818c5) **\(is/is not\)**
8. [**Jay Alamar on Bert**](http://jalammar.github.io/illustrated-bert/)
9. [**Jay Alamar on using distilliBert** ](http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/)
10. [**sparse bert**](https://github.com/huggingface/transformers/tree/master/examples/movement-pruning)**,** [**paper**](https://arxiv.org/abs/2005.07683) **- When combined with distillation, the approach achieves minimal accuracy loss with down to only 3% of the model parameters.**
11. **Bert with keras,** [**blog post**](https://www.ctolib.com/Separius-BERT-keras.html)**,** [**colaboratory**](https://colab.research.google.com/gist/HighCWu/3a02dc497593f8bbe4785e63be99c0c3/bert-keras-tutorial.ipynb)
12. [**Bert with t-hub**](https://github.com/google-research/bert/blob/master/run_classifier_with_tfhub.py)
13. [**Bert on medium with code**](https://medium.com/huggingface/multi-label-text-classification-using-bert-the-mighty-transformer-69714fa3fb3d)
14. [**Bert on git**](https://github.com/SkullFang/BERT_NLP_Classification)
15. **Finetuning -** [**Better sentiment analysis with bert**](https://medium.com/southpigalle/how-to-perform-better-sentiment-analysis-with-bert-ba127081eda)**, claims 94% on IMDB. official code** [**here**](https://github.com/google-research/bert/blob/master/predicting_movie_reviews_with_bert_on_tf_hub.ipynb) **“ it creates a single new layer that will be trained to adapt BERT to our sentiment task \(i.e. classifying whether a movie review is positive or negative\). This strategy of using a mostly trained model is called** [**fine-tuning**](http://wiki.fast.ai/index.php/Fine_tuning)**.”**
16. [**Explain bert**](http://exbert.net/) **- bert visualization tool.**
17. **sentenceBERT** [**paper**](https://arxiv.org/pdf/1908.10084.pdf)
18. [**Bert question answering**](https://towardsdatascience.com/testing-bert-based-question-answering-on-coronavirus-articles-13623637a4ff?source=email-4dde5994e6c1-1586483206529-newsletter.v2-7f60cf5620c9-----0-------------------b506d4ba_2902_4718_9c95_a36e33d638e6---48577de843eb----20200410) **on covid19**
19. [**Codebert**](https://arxiv.org/pdf/2002.08155.pdf?fbclid=IwAR3XXrpuILgnqTHCI1-0LHPT39IJVVaBl9uGXTVAjUwb1xM8NGrKUHrEyac)
20. [**Bert multilabel classification**](http://towardsdatascience)
21. [**Tabert**](https://ai.facebook.com/blog/tabert-a-new-model-for-understanding-queries-over-tabular-data/) **-** [**TaBERT**](https://ai.facebook.com/research/publications/tabert-pretraining-for-joint-understanding-of-textual-and-tabular-data/) **is the first model that has been pretrained to learn representations for both natural language sentences and tabular data.** 
22. [**All the ways that you can compress BERT**](http://mitchgordon.me/machine/learning/2019/11/18/all-the-ways-to-compress-BERT.html?fbclid=IwAR0X2g4VQDpN4otb7YPzn88r5XMg8gRd3NWfm3dd6P0aFZEEtOGKY9QU5ec)

**Pruning - Removes unnecessary parts of the network after training. This includes weight magnitude pruning, attention head pruning, layers, and others. Some methods also impose regularization during training to increase prunability \(layer dropout\).**

**Weight Factorization - Approximates parameter matrices by factorizing them into a multiplication of two smaller matrices. This imposes a low-rank constraint on the matrix. Weight factorization can be applied to both token embeddings \(which saves a lot of memory on disk\) or parameters in feed-forward / self-attention layers \(for some speed improvements\).**

**Knowledge Distillation - Aka “Student Teacher.” Trains a much smaller Transformer from scratch on the pre-training / downstream-data. Normally this would fail, but utilizing soft labels from a fully-sized model improves optimization for unknown reasons. Some methods also distill BERT into different architectures \(LSTMS, etc.\) which have faster inference times. Others dig deeper into the teacher, looking not just at the output but at weight matrices and hidden activations.**

**Weight Sharing - Some weights in the model share the same value as other parameters in the model. For example, ALBERT uses the same weight matrices for every single layer of self-attention in BERT.**

**Quantization - Truncates floating point numbers to only use a few bits \(which causes round-off error\). The quantization values can also be learned either during or after training.**

**Pre-train vs. Downstream - Some methods only compress BERT w.r.t. certain downstream tasks. Others compress BERT in a way that is task-agnostic.**

1. [**Bert and nlp in 2019**](https://towardsdatascience.com/2019-year-of-bert-and-transformer-f200b53d05b9)
2. [**HeBert - bert for hebrwe sentiment and emotions**](https://github.com/avichaychriqui/HeBERT)
3. [**Kdbuggets on visualizing bert**](https://www.kdnuggets.com/2019/03/deconstructing-bert-part-2-visualizing-inner-workings-attention.html)
4. [**What does bert look at, analysis of attention**](https://www-nlp.stanford.edu/pubs/clark2019what.pdf) **-  We further show that certain attention heads correspond well to linguistic notions of syntax and coreference. For example, we find heads that attend to the direct objects of verbs, determiners of nouns, objects of prepositions, and coreferent mentions with remarkably high accuracy. Lastly, we propose an attention-based probing classifier and use it to further demonstrate that substantial syntactic information is captured in BERT’s attention**
5. [**Bertviz**](https://github.com/jessevig/bertviz) **BertViz is a tool for visualizing attention in the Transformer model, supporting all models from the** [**transformers**](https://github.com/huggingface/transformers) **library \(BERT, GPT-2, XLNet, RoBERTa, XLM, CTRL, etc.\). It extends the** [**Tensor2Tensor visualization tool**](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/visualization) **by** [**Llion Jones**](https://medium.com/@llionj) **and the** [**transformers**](https://github.com/huggingface/transformers) **library from** [**HuggingFace**](https://github.com/huggingface)**.**
6. **PMI-masking** [**paper**](https://openreview.net/forum?id=3Aoft6NWFej)**,** [**post**](https://www.ai21.com/pmi-masking) **- Joint masking of correlated tokens significantly speeds up and improves BERT's pretraining**
7. **\(really good/\)** [**Examining bert raw embeddings**](https://towardsdatascience.com/examining-berts-raw-embeddings-fd905cb22df7) **- TL;DR BERT’s raw word embeddings capture useful and separable information \(distinct histogram tails\) about a word in terms of other words in BERT’s vocabulary. This information can be harvested from both raw embeddings and their transformed versions after they pass through BERT with a Masked language model \(MLM\) head**

![](https://lh6.googleusercontent.com/nIgQQPipHF7dhRxdOw79cMhogIBvcdNjftMtQckXAKuZWkZgpgXiaBgyijRI1IB5x7oTLSRF0yL9XKv64hsSAhdnsPiRWMiIR8vQyZOpzpPdD-Qe9YTzvMgRVcEdOMQf9bCTdjVb)

![](https://lh6.googleusercontent.com/gma8aGDKP8chI7HuhKdl2Gu6tFUT_iHghfYZ8YyvfQta3-6DFw5YSZK2v-at3XneSjo0QnVtXfcs9wNL8CdCY4D8aZXxNlduUjwXxqjao6WoiAN17R5qH46Cx1SDGjU-yu5O9W13)

![](https://lh5.googleusercontent.com/4_FW_BymDsKMdFzKVNZ2Dmm_3pI6UrNlPWK7YsBgIznbAi551G0QkCUrRVK0sW6_sMsZ_WFJ0GwHdlu0X3YNjZ0k947iQ27PVG6ZSp7jOWjhRNr5d7FbMe1lauiresaYn9u1nXIY)

![](https://lh5.googleusercontent.com/Hp7oLFDNtANqlV5RQzKWF-TsuURUlxQZS_sjQFXD48H3PnTtwthIGfN1zxKU14uf8y4746oXRzc4KvfyW4zBcKOdwL92LKYb9cwfDsD14-y_Lv6pmBdnwrpDyqzP0LjLEpEqWk5b)

### **GPT2**

1. [**the GPT-2**](https://medium.com/dair-ai/experimenting-with-openais-improved-language-model-abf73bc123b9) **small algorithm was trained on the task of language modeling — which tests a program’s ability to predict the next word in a given sentence — by ingesting huge numbers of articles, blogs, and websites. By using just this data it achieved state-of-the-art scores on a number of unseen language tests, an achievement known as zero-shot learning. It can also perform other writing-related tasks, such as translating text from one language to another, summarizing long articles, and answering trivia questions.**
2. [**Medium code**](https://medium.com/dair-ai/explore-pretrained-language-models-with-pytorch-1b1e06b7510c) **for GPT=2 - big algo**

### **GPT3**

1. [**GPT3**](https://medium.com/swlh/all-hail-gpt-3-389c7f1fcb3b) **on medium - language models can be used to produce good results on zero-shot, one-shot, or few-shot learning.**
2. [**Fit More and Train Faster With ZeRO via DeepSpeed and FairScale**](https://huggingface.co/blog/zero-deepspeed-fairscale)

### **XLNET**

1. [**Xlnet is transformer and bert combined**](https://medium.com/logits/xlnet-sota-pre-training-method-that-outperforms-bert-26d4e9978983) **- Actually its quite good explaining it**
2. [**git**](https://github.com/zihangdai/xlnet)
3. **CLIP**
4. **\(keras\)** [**Implementation of a dual encoder**](https://keras.io/examples/nlp/nl_image_search/) **model for retrieving images that match natural language queries. - The example demonstrates how to build a dual encoder \(also known as two-tower\) neural network model to search for images using natural language. The model is inspired by the** [**CLIP**](https://openai.com/blog/clip/) **approach, introduced by Alec Radford et al. The idea is to train a vision encoder and a text encoder jointly to project the representation of images and their captions into the same embedding space, such that the caption embeddings are located near the embeddings of the images they describe.**
5. 1. **Adversarial methodologies**
6. **What is label** [**flipping and smoothing**](https://datascience.stackexchange.com/questions/55359/how-label-smoothing-and-label-flipping-increases-the-performance-of-a-machine-le/56662) **and usage for making a model more robust against adversarial methodologies - 0**

**Label flipping is a training technique where one selectively manipulates the labels in order to make the model more robust against label noise and associated attacks - the specifics depend a lot on the nature of the noise. Label flipping bears no benefit only under the assumption that all labels are \(and will always be\) correct and that no adversaries exist. In cases where noise tolerance is desirable, training with label flipping is beneficial.**

**Label smoothing is a regularization technique \(and then some\) aimed at improving model performance. Its effect takes place irrespective of label correctness.**

1. [ **Paper: when does label smoothing helps?**](https://arxiv.org/abs/1906.02629) **Smoothing the labels in this way prevents the network from becoming overconfident and label smoothing has been used in many state-of-the-art models, including image classification, language translation and speech recognition...Here we show empirically that in addition to improving generalization, label smoothing improves model calibration which can significantly improve beam-search. However, we also observe that if a teacher network is trained with label smoothing, knowledge distillation into a student network is much less effective.**
2. [**Label smoothing, python code, multi class examples**](https://rickwierenga.com/blog/fast.ai/FastAI2019-12.html)

![](https://lh4.googleusercontent.com/pScpTAmy9S8uTobVoSLAjSlASouxyA2iBDNxH8VEjBg4indhs57dHWYXoqEZSTfp6Hhwh9i0LboD65o1LXfxv61dMJwnz1dDbm1lhcvVYtvVbW8H6Rhia-lk0bLfDomS3z6kKNlZ)

1. [**Label sanitazation against label flipping poisoning attacks**](https://arxiv.org/abs/1803.00992) **- In this paper we propose an efficient algorithm to perform optimal label flipping poisoning attacks and a mechanism to detect and relabel suspicious data points, mitigating the effect of such poisoning attacks.**
2. [**Adversarial label flips attacks on svm**](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.398.7446&rep=rep1&type=pdf) **- To develop a robust classification algorithm in the adversarial setting, it is important to understand the adversary’s strategy. We address the problem of label flips attack where an adversary contaminates the training set through flipping labels. By analyzing the objective of the adversary, we formulate an optimization framework for finding the label flips that maximize the classification error. An algorithm for attacking support vector machines is derived. Experiments demonstrate that the accuracy of classifiers is significantly degraded under the attack.**
3. **GAN**
4. [**Great advice for training gans**](https://medium.com/@utk.is.here/keep-calm-and-train-a-gan-pitfalls-and-tips-on-training-generative-adversarial-networks-edd529764aa9)**, such as label flipping batch norm, etc read!**
5. [**Intro to Gans**](https://medium.com/sigmoid/a-brief-introduction-to-gans-and-how-to-code-them-2620ee465c30)
6. [**A fantastic series about gans, the following two what are gans and applications are there**](https://medium.com/@jonathan_hui/gan-gan-series-2d279f906e7b)
   1. [**What are a GANs?**](https://medium.com/@jonathan_hui/gan-whats-generative-adversarial-networks-and-its-application-f39ed278ef09)**, and cool** [**applications**](https://medium.com/@jonathan_hui/gan-some-cool-applications-of-gans-4c9ecca35900)
   2. [**Comprehensive overview**](https://medium.com/@jonathan_hui/gan-a-comprehensive-review-into-the-gangsters-of-gans-part-1-95ff52455672)
   3. [**Cycle gan**](https://medium.com/@jonathan_hui/gan-cyclegan-6a50e7600d7) **- transferring styles**
   4. [**Super gan resolution**](https://medium.com/@jonathan_hui/gan-super-resolution-gan-srgan-b471da7270ec) **- super res images**
   5. [**Why gan so hard to train**](https://medium.com/@jonathan_hui/gan-why-it-is-so-hard-to-train-generative-advisory-networks-819a86b3750b) **- good for critique**
   6. [**And how to improve gans performance**](https://towardsdatascience.com/gan-ways-to-improve-gan-performance-acf37f9f59b)
   7. [**Dcgan good as a starting point in new projects**](https://medium.com/@jonathan_hui/gan-dcgan-deep-convolutional-generative-adversarial-networks-df855c438f)
   8. [**Labels to improve gans, cgan, infogan**](https://medium.com/@jonathan_hui/gan-cgan-infogan-using-labels-to-improve-gan-8ba4de5f9c3d)
   9. [**Stacked - labels, gan adversarial loss, entropy loss, conditional loss**](https://medium.com/@jonathan_hui/gan-stacked-generative-adversarial-networks-sgan-d9449ac63db8) **- divide and conquer**
   10. [**Progressive gans**](https://medium.com/@jonathan_hui/gan-progressive-growing-of-gans-f9e4f91edf33) **- mini batch discrimination**
   11. [**Using attention to improve gan**](https://medium.com/@jonathan_hui/gan-self-attention-generative-adversarial-networks-sagan-923fccde790c)
   12. [**Least square gan - lsgan**](https://medium.com/@jonathan_hui/gan-lsgan-how-to-be-a-good-helper-62ff52dd3578)
   13. **Unread:**
       1. [**Wasserstein gan, wgan gp**](https://medium.com/@jonathan_hui/gan-wasserstein-gan-wgan-gp-6a1a2aa1b490)
       2. [**Faster training for gans, lower training count rsgan ragan**](https://medium.com/@jonathan_hui/gan-rsgan-ragan-a-new-generation-of-cost-function-84c5374d3c6e)
       3. [**Addressing gan stability, ebgan began**](https://medium.com/@jonathan_hui/gan-energy-based-gan-ebgan-boundary-equilibrium-gan-began-4662cceb7824)
       4. [**What is wrong with gan cost functions**](https://medium.com/@jonathan_hui/gan-what-is-wrong-with-the-gan-cost-function-6f594162ce01)
       5. [**Using cost functions for gans inspite of the google brain paper**](https://medium.com/@jonathan_hui/gan-does-lsgan-wgan-wgan-gp-or-began-matter-e19337773233)
       6. [**Proving gan is js-convergence**](https://medium.com/@jonathan_hui/proof-gan-optimal-point-658116a236fb)
       7. [**Dragan on minimizing local equilibria, how to stabilize gans**](https://medium.com/@jonathan_hui/gan-dragan-5ba50eafcdf2)**, reducing mode collapse**
       8. [**Unrolled gan for reducing mode collapse**](https://medium.com/@jonathan_hui/gan-unrolled-gan-how-to-reduce-mode-collapse-af5f2f7b51cd)
       9. [**Measuring gans**](https://medium.com/@jonathan_hui/gan-how-to-measure-gan-performance-64b988c47732)
       10. [**Ways to improve gans performance**](https://towardsdatascience.com/gan-ways-to-improve-gan-performance-acf37f9f59b)
       11. [**Introduction to gans**](https://medium.freecodecamp.org/an-intuitive-introduction-to-generative-adversarial-networks-gans-7a2264a81394) **with tf code**
       12. [**Intro to gans**](https://medium.com/datadriveninvestor/deep-learning-generative-adversarial-network-gan-34abb43c0644)
       13. [**Intro to gan in KERAS**](https://towardsdatascience.com/demystifying-generative-adversarial-networks-c076d8db8f44)
7. **“GAN”** [**using xgboost and gmm for density sampling**](https://edge.skyline.ai/data-synthesizers-on-aws-sagemaker)
8. [**Reverse engineering** ](https://ai.facebook.com/blog/reverse-engineering-generative-model-from-a-single-deepfake-image/)

## **SIAMESE NETWORKS**

1. [**Siamese for conveyor belt fault prediction**](https://towardsdatascience.com/predictive-maintenance-with-lstm-siamese-network-51ee7df29767)
2. [**Burlow**](https://arxiv.org/abs/2103.03230)**,** [**fb post**](https://www.facebook.com/yann.lecun/posts/10157682573642143) **- Self-supervised learning \(SSL\) is rapidly closing the gap with supervised methods on large computer vision benchmarks. A successful approach to SSL is to learn representations which are invariant to distortions of the input sample. However, a recurring issue with this approach is the existence of trivial constant solutions. Most current methods avoid such solutions by careful implementation details. We propose an objective function that naturally avoids such collapse by measuring the cross-correlation matrix between the outputs of two identical networks fed with distorted versions of a sample, and making it as close to the identity matrix as possible. This causes the representation vectors of distorted versions of a sample to be similar, while minimizing the redundancy between the components of these vectors.** 

## **Gated Multi-Layer Perceptron \(GMLP\)**

1. \*\*\*\*[**paper**](https://arxiv.org/abs/2105.08050)**,** [**git1**](https://github.com/jaketae/g-mlp)**,** [**git2**](https://github.com/lucidrains/g-mlp-pytorch) **- "**a simple network architecture, gMLP, based on MLPs with gating, and show that it can perform as well as Transformers in key language and vision applications. Our comparisons show that self-attention is not critical for Vision Transformers, as gMLP can achieve the same accuracy."

![](../.gitbook/assets/image%20%285%29.png)

