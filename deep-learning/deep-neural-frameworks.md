# Deep Neural Frameworks

## **PYTORCH**

1. **Deep learning with pytorch -** [**The book**](https://pytorch.org/assets/deep-learning/Deep-Learning-with-PyTorch.pdf)
2. [**Pytorch DL course**](https://atcold.github.io/pytorch-Deep-Learning/)**,** [**git**](https://github.com/Atcold/pytorch-Deep-Learning) **- yann lecun**
3. **Pytorch Official**
   1. [**Tutorials**](https://pytorch.org/tutorials/)\
      ![](<../.gitbook/assets/image (44).png>)
   2. [**Learning with examples**](https://pytorch.org/tutorials/beginner/pytorch\_with\_examples.html)
   3. [Learn the Basics](https://pytorch.org/tutorials/beginner/basics/intro.html) || [Quickstart](https://pytorch.org/tutorials/beginner/basics/quickstart\_tutorial.html) || [Tensors](https://pytorch.org/tutorials/beginner/basics/tensorqs\_tutorial.html) || [Datasets & DataLoaders](https://pytorch.org/tutorials/beginner/basics/data\_tutorial.html) || [Transforms](https://pytorch.org/tutorials/beginner/basics/transforms\_tutorial.html) || [Build Model](https://pytorch.org/tutorials/beginner/basics/buildmodel\_tutorial.html) || [Autograd](https://pytorch.org/tutorials/beginner/basics/autogradqs\_tutorial.html) || [Optimization](https://pytorch.org/tutorials/beginner/basics/optimization\_tutorial.html) || [Save & Load Model](https://pytorch.org/tutorials/beginner/basics/saveloadrun\_tutorial.html)
   4. [60 minute blitz](https://pytorch.org/tutorials/beginner/deep\_learning\_60min\_blitz.html)
   5. (good) - [youtube series](https://pytorch.org/tutorials/beginner/introyt.html)

## **FAST.AI**

1. [**git**](https://github.com/fastai/fastai)

## **KERAS**

[**A make sense introduction into keras**](https://www.youtube.com/playlist?list=PLFxrZqbLojdKuK7Lm6uamegEFGW2wki6P)**, has several videos on the topic, going through many network types, creating custom activation functions, going through examples.**

**+ Two extra videos from the same author,** [**examples**](https://www.youtube.com/watch?v=6RdflAr66-E) **and** [**examples-2**](https://www.youtube.com/watch?v=fDKdITMBAGk)

**Didn’t read:**

1. [**Keras cheatsheet**](https://www.datacamp.com/community/blog/keras-cheat-sheet)
2. [**Seq2Seq RNN**](https://stackoverflow.com/questions/41933958/how-to-code-a-sequence-to-sequence-rnn-in-keras)
3. [**Stateful LSTM**](https://github.com/fchollet/keras/blob/master/examples/stateful\_lstm.py) **- Example script showing how to use stateful RNNs to model long sequences efficiently.**
4. [**CONV LSTM**](https://github.com/fchollet/keras/blob/master/examples/conv\_lstm.py) **- this script demonstrate the use of a conv LSTM network, used to predict the next frame of an artificially generated move which contains moving squares.**

[**How to force keras to use tensorflow**](https://github.com/ContinuumIO/anaconda-issues/issues/1735) **and not teano (set the .bat file)**

[**Callbacks - how to create an AUC ROC score callback with keras**](https://keunwoochoi.wordpress.com/2016/07/16/keras-callbacks/) **- with code example.**

[**Batch size vs. Iteratio**](https://stats.stackexchange.com/questions/164876/tradeoff-batch-size-vs-number-of-iterations-to-train-a-neural-network)**ns in NN Keras.**

[**Keras metrics**](https://machinelearningmastery.com/custom-metrics-deep-learning-keras-python/) **- classification regression and custom metrics**

[**Keras Metrics 2**](https://machinelearningmastery.com/metrics-evaluate-machine-learning-algorithms-python/) **- accuracy, ROC, AUC, classification, regression r^2.**

[**Introduction to regression models in Keras,**](https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/) **using MSE, comparing baseline vs wide vs deep networks.**

[**How does Keras calculate accuracy**](https://datascience.stackexchange.com/questions/14415/how-does-keras-calculate-accuracy)**? Formula and explanation**

**Compares label with the rounded predicted float, i.e. bigger than 0.5 = 1, smaller than = 0**

**For categorical we take the argmax for the label and the prediction and compare their location.**

**In both cases, we average the results.**

[**Custom metrics (precision recall) in keras**](https://stackoverflow.com/questions/41458859/keras-custom-metric-for-single-class-accuracy)**. Which are taken from** [**here**](https://github.com/autonomio/talos/tree/master/talos/metrics)**, including entropy and f1**

### **KERAS MULTI GPU**

1. [**When using SGD only batches between 32-512 are adequate, more can lead to lower performance, less will lead to slow training times.**](https://arxiv.org/pdf/1609.04836.pdf)
2. **Note: probably doesn't reflect on adam, is there a reference?**
3. [**Parallel gpu-code for keras. Its a one liner, but remember to scale batches by the amount of GPU used in order to see a (non linear) scaability in training time.**](https://datascience.stackexchange.com/questions/23895/multi-gpu-in-keras)
4. [**Pitfalls in GPU training, this is a very important post, be aware that you can corrupt your weights using the wrong combination of batches-to-input-size**](http://blog.datumbox.com/5-tips-for-multi-gpu-training-with-keras/)**, in keras-tensorflow. When you do multi-GPU training, it is important to feed all the GPUs with data. It can happen that the very last batch of your epoch has less data than defined (because the size of your dataset can not be divided exactly by the size of your batch). This might cause some GPUs not to receive any data during the last step. Unfortunately some Keras Layers, most notably the Batch Normalization Layer, can’t cope with that leading to nan values appearing in the weights (the running mean and variance in the BN layer).**
5. [**5 things to be aware of for multi gpu using keras, crucial to look at before doing anything**](http://blog.datumbox.com/5-tips-for-multi-gpu-training-with-keras/)

**KERAS FUNCTIONAL API**

[**What is and how to use?**](https://machinelearningmastery.com/keras-functional-api-deep-learning/) **A flexible way to declare layers in parallel, i.e. parallel ways to deal with input, feature extraction, models and outputs as seen in the following images.**\
![Neural Network Graph With Shared Feature Extraction Layer](https://lh5.googleusercontent.com/tdK7TuCAsYPfx\_vLBps4HU2dLQqA2M7prppP5V7xOzuT2SGeV\_T3hJ94wvJMC0gBY1XS81bK6uKzOZ2HNazaEBRtD-a1xAtPS8OtcaEtjhqRi-GjH1iFOZM\_2WDCWzs73odUzTbd)![Neural Network Graph With Multiple Inputs](https://lh6.googleusercontent.com/ptnE\_MAQyTSSYyRCULQRnIx7XRa\_7zVLSEbclJuebxvZPotAqJIe2ElY5SuF42UdfrEdIWFII7BwsVUrCkAXp3Ta1GCmrPLsir-duOxF5wkRn62uH0M4etHjBVNQOF7luWc4Qs9K)

![Neural Network Graph With Multiple Outputs](https://lh4.googleusercontent.com/pdU8st0CBS7qGN14dBXm6XbFJCL-hMAPtRjz\_\_la0DN96IwABz-PV0i-xTEEAf5yBMOTBfi6QwAsnuGFnonRbSxdbQWl33bssITuR3zInVupAW0z9RSTCpqc9UwlAi6PZ0elyDLa)

### **KERAS EMBEDDING LAYER**

1. [**Injecting glove to keras embedding layer and using it for classification + what is and how to use the embedding layer in keras.**](https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/)
2. [**Keras blog - using GLOVE for pretrained embedding layers.**](https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html)
3. [**Word embedding using keras, continuous BOW - CBOW, SKIPGRAM, word2vec - really good.**](https://towardsdatascience.com/understanding-feature-engineering-part-4-deep-learning-methods-for-text-data-96c44370bbfa)
4. [**Fasttext - comparison of key feature against word2vec**](https://www.quora.com/What-is-the-main-difference-between-word2vec-and-fastText)
5. [**Multiclass classification using word2vec/glove + code**](https://github.com/dennybritz/cnn-text-classification-tf/issues/69)
6. [**word2vec/doc2vec/tfidf code in python for text classification**](https://github.com/davidsbatista/text-classification/blob/master/train\_classifiers.py)
7. [**Lda & word2vec**](https://www.kaggle.com/vukglisovic/classification-combining-lda-and-word2vec)
8. [**Text classification with word2vec**](http://nadbordrozd.github.io/blog/2016/05/20/text-classification-with-word2vec/)
9. [**Gensim word2vec**](https://radimrehurek.com/gensim/models/word2vec.html)**, and** [**another one**](http://kavita-ganesan.com/gensim-word2vec-tutorial-starter-code/)
10. [**Fasttext paper**](https://arxiv.org/abs/1607.01759)

### **Keras: Predict vs Evaluate**

[**here:**](https://www.quora.com/What-is-the-difference-between-keras-evaluate-and-keras-predict)

**.predict() generates output predictions based on the input you pass it (for example, the predicted characters in the** [**MNIST example**](https://github.com/fchollet/keras/blob/master/examples/mnist\_mlp.py)**)**

**.evaluate() computes the loss based on the input you pass it, along with any other metrics that you requested in the metrics param when you compiled your model (such as accuracy in the** [**MNIST example**](https://github.com/fchollet/keras/blob/master/examples/mnist\_mlp.py)**)**

**Keras metrics**

[**For classification methods - how does keras calculate accuracy, all functions.**](https://www.quora.com/How-does-Keras-calculate-accuracy)

### **LOSS IN KERAS**

[**Why is the training loss much higher than the testing loss?**](https://keras.io/getting-started/faq/#why-is-the-training-loss-much-higher-than-the-testing-loss) **A Keras model has two modes: training and testing. Regularization mechanisms, such as Dropout and L1/L2 weight regularization, are turned off at testing time.**

**The training loss is the average of the losses over each batch of training data. Because your model is changing over time, the loss over the first batches of an epoch is generally higher than over the last batches. On the other hand, the testing loss for an epoch is computed using the model as it is at the end of the epoch, resulting in a lower loss.**
