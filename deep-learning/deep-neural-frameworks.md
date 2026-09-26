# Deep Neural Frameworks

A deep learning idea only becomes a trained model once it is written in a library that runs it on real hardware. This page walks the libraries used in practice in that order: PyTorch first, then fast.ai on top of it, then Keras with its metrics, multi-GPU training, functional API, embeddings, predict versus evaluate, and training loss, and finally the NVIDIA CUDA and cuDNN stack that TensorFlow and the other GPU tooling need underneath.

## PYTORCH

The first framework stack is PyTorch, and the path into it runs from a book to a course to the official tutorials. It starts with Deep learning with pytorch, the book, whose address is kept at the end of the page. The course that follows it is Yann LeCun's NYU Deep Learning Spring 2020, a Pytorch DL course whose repository is the [git](https://github.com/Atcold/pytorch-Deep-Learning) link.

After the book and the course comes Pytorch Official. [Tutorials](https://pytorch.org/tutorials/) is the welcome page of the PyTorch tutorials documentation, the index for everything below.

 <figure><img src="../.gitbook/assets/image (44).png" alt=""><figcaption><p>Tutorials</p></figcaption></figure>

From that index, [Learning with examples](https://pytorch.org/tutorials/beginner/pytorch_with_examples.html) is Learning PyTorch with Examples. The beginner sequence then goes step by step, and here the order is the point:

1. [Learn the Basics](https://pytorch.org/tutorials/beginner/basics/intro.html) is the entry page of the sequence.
2. [Quickstart](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html) is the short end-to-end run.
3. [Tensors](https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html) covers the basic data structure.
4. [Datasets & DataLoaders](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html) covers loading data.
5. [Transforms](https://pytorch.org/tutorials/beginner/basics/transforms_tutorial.html) covers preparing it.
6. [Build Model](https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html) is Build the Neural Network.
7. [Autograd](https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html) is Automatic Differentiation with torch.autograd.
8. [Optimization](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html) is Optimizing Model Parameters.
9. [Save & Load Model](https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html) is Save and Load the Model.

If that sequence is too slow, the [60 minute blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html) is Deep Learning with PyTorch: A 60 Minute Blitz. For video, the Introduction to PyTorch [youtube series](https://pytorch.org/tutorials/beginner/introyt.html) is marked (good).

## FAST.AI

Once PyTorch is familiar, fast.ai is the library and course series built on it. The same notes are in [FAST.AI](deep-neural-frameworks.md#fastai).

The library itself is the [git](https://github.com/fastai/fastai) repository, the fastai deep learning library. The courses that teach it are summarized on [Medium](https://medium.com/@hiromi_suenaga/deep-learning-2-part-1-lesson-1-602f73869197) on all fast.ai courses, 14 posts.

## KERAS

Keras is the other high-level route, and it carries most of the practical notes on this page: an introduction, metrics, the functional API, embeddings, predict versus evaluate, and training loss.

The way in is [A make sense introduction into keras](https://www.youtube.com/playlist?list=PLFxrZqbLojdKuK7Lm6uamegEFGW2wki6P), which has several videos on the topic, going through many network types, creating custom activation functions, going through examples. Two extra videos from the same author, Danielle Van Boxel, walk Keras code line by line: [examples](https://www.youtube.com/watch?v=6RdflAr66-E) and [examples-2](https://www.youtube.com/watch?v=fDKdITMBAGk).

Some sources are kept under "Didn't read:". The [Keras cheatsheet](https://www.datacamp.com/community/blog/keras-cheat-sheet) is DataCamp's sheet for Keras as a high-level neural networks API over Theano and TensorFlow. [Seq2Seq RNN](https://stackoverflow.com/questions/41933958/how-to-code-a-sequence-to-sequence-rnn-in-keras) is a question on how to code a sequence to sequence RNN in keras, with tokenized, padded text and a target shifted left. Stateful LSTM was an example script showing how to use stateful RNNs to model long sequences efficiently, and CONV LSTM was a script to demonstrate the use of a conv LSTM network, used to predict the next frame of an artificially generated move which contains moving squares; both scripts are kept at the end of the page.

The first practical snag is the backend. [How to force keras to use tensorflow](https://github.com/ContinuumIO/anaconda-issues/issues/1735) and not teano (set the .bat file) is the Anaconda issue on switching keras backends between theano and tensorflow under Windows.

With the backend set, training needs hooks and a batch size. [Callbacks - how to create an AUC ROC score callback with keras](https://keunwoochoi.wordpress.com/2016/07/16/keras-callbacks/) is a Keras callbacks guide, with code example. [Batch size vs. Iterations in NN Keras.](https://stats.stackexchange.com/questions/164876/tradeoff-batch-size-vs-number-of-iterations-to-train-a-neural-network) asks what the trade-off is between batch size and number of iterations when the total number of training examples stays the same.

Then comes measuring the model. [Keras metrics](https://machinelearningmastery.com/custom-metrics-deep-learning-keras-python/) is Jason Brownlee's guide to the standard metrics Keras reports during training and how to define your own - classification regression and custom metrics. [Keras Metrics 2](https://machinelearningmastery.com/metrics-evaluate-machine-learning-algorithms-python/) is his note on why the choice of metric shapes how algorithms are compared - accuracy, ROC, AUC, classification, regression r^2. [Introduction to regression models in Keras,](https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/) using MSE, comparing baseline vs wide vs deep networks.

The most common metric hides a threshold. [How does Keras calculate accuracy](https://datascience.stackexchange.com/questions/14415/how-does-keras-calculate-accuracy)? Formula and explanation: the question is what threshold Keras uses to turn classwise probabilities into a class. For binary labels it compares label with the rounded predicted float, i.e. bigger than 0.5 = 1, smaller than = 0. For categorical we take the argmax for the label and the prediction and compare their location. In both cases, we average the results.

When accuracy over all classes is not enough, [Custom metrics (precision recall) in keras](https://stackoverflow.com/questions/41458859/keras-custom-metric-for-single-class-accuracy) shows a custom metric for the accuracy of one class in a multi-class dataset. Which are taken from [here](https://github.com/autonomio/talos/tree/master/talos/metrics), the metrics folder of talos, a tool for hyperparameter experiments with TensorFlow and Keras, including entropy and f1.

### KERAS MULTI GPU

Batch size matters even more once training spreads across several GPUs, and this part is multi-GPU training in Keras, including batch size and the last-batch pitfall.

The starting constraint comes from the paper on SGD in the small-batch regime, where a fraction of the training data, say 32 to 512 points, is sampled to approximate the gradient: [When using SGD only batches between 32-512 are adequate, more can lead to lower performance, less will lead to slow training times.](https://arxiv.org/pdf/1609.04836.pdf) Note: probably doesn't reflect on adam, is there a reference?

The code itself is short. [Parallel gpu-code for keras. Its a one liner, but remember to scale batches by the amount of GPU used in order to see a (non linear) scaability in training time.](https://datascience.stackexchange.com/questions/23895/multi-gpu-in-keras) The question behind it is how to partition training across, say, the 8 GPUs of an Amazon ec2 instance.

The danger is in the last batch. [Pitfalls in GPU training, this is a very important post, be aware that you can corrupt your weights using the wrong combination of batches-to-input-size](http://blog.datumbox.com/5-tips-for-multi-gpu-training-with-keras/), in keras-tensorflow. When you do multi-GPU training, it is important to feed all the GPUs with data. It can happen that the very last batch of your epoch has less data than defined (because the size of your dataset can not be divided exactly by the size of your batch). This might cause some GPUs not to receive any data during the last step. Unfortunately some Keras Layers, most notably the Batch Normalization Layer, can't cope with that leading to nan values appearing in the weights (the running mean and variance in the BN layer). The same post, 5 tips for multi-GPU training with Keras, is linked again as [5 things to be aware of for multi gpu using keras, crucial to look at before doing anything](http://blog.datumbox.com/5-tips-for-multi-gpu-training-with-keras/).

### KERAS FUNCTIONAL API

A sequential stack of layers is not the only shape a model can take, and this part is the functional API: layers in parallel, and graphs with shared features, multiple inputs, or multiple outputs.

[What is and how to use?](https://machinelearningmastery.com/keras-functional-api-deep-learning/) A flexible way to declare layers in parallel, i.e. parallel ways to deal with input, feature extraction, models and outputs as seen in the following images.

<figure><img src="../.gitbook/assets/gimg-1735947f35a7.png" alt=""><figcaption><p>Neural Network Graph With Shared Feature Extraction Layer</p><p>Credit: <a href="https://lh5.googleusercontent.com/tdK7TuCAsYPfx_vLBps4HU2dLQqA2M7prppP5V7xOzuT2SGeV_T3hJ94wvJMC0gBY1XS81bK6uKzOZ2HNazaEBRtD-a1xAtPS8OtcaEtjhqRi-GjH1iFOZM_2WDCWzs73odUzTbd">copied from the original hosted image</a>.</p></figcaption></figure>

<figure><img src="../.gitbook/assets/gimg-475a3967478b.png" alt=""><figcaption><p>Neural Network Graph With Multiple Inputs</p><p>Credit: <a href="https://lh6.googleusercontent.com/ptnE_MAQyTSSYyRCULQRnIx7XRa_7zVLSEbclJuebxvZPotAqJIe2ElY5SuF42UdfrEdIWFII7BwsVUrCkAXp3Ta1GCmrPLsir-duOxF5wkRn62uH0M4etHjBVNQOF7luWc4Qs9K">copied from the original hosted image</a>.</p></figcaption></figure>

<figure><img src="../.gitbook/assets/gimg-68296f68b1ca.png" alt=""><figcaption><p>Neural Network Graph With Multiple Outputs</p><p>Credit: <a href="https://lh4.googleusercontent.com/pdU8st0CBS7qGN14dBXm6XbFJCL-hMAPtRjz__la0DN96IwABz-PV0i-xTEEAf5yBMOTBfi6QwAsnuGFnonRbSxdbQWl33bssITuR3zInVupAW0z9RSTCpqc9UwlAi6PZ0elyDLa">copied from the original hosted image</a>.</p></figcaption></figure>

The three graphs show the shared feature extraction layer, multiple inputs, and multiple outputs, each copied from the original hosted image.

### KERAS EMBEDDING LAYER

Text models usually start with one particular layer, and this part is the Keras embedding layer, GloVe, word2vec, and fastText. The same notes are in [Embedding](representations.md).

Jason Brownlee's guide to word embedding layers explains why dense word representations beat sparse bag-of-words ones and how they can be learned as part of fitting the network: [Injecting glove to keras embedding layer and using it for classification + what is and how to use the embedding layer in keras.](https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/) The official route to pretrained vectors is [Keras blog - using GLOVE for pretrained embedding layers.](https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html) Word embedding using keras, continuous BOW - CBOW, SKIPGRAM, word2vec - really good; the link for that one sits at the end of the loss section below.

To compare the two families of vectors, [Fasttext - comparison of key feature against word2vec](https://www.quora.com/What-is-the-main-difference-between-word2vec-and-fastText) asks what the main difference between word2vec and fastText is, and the [Fasttext paper](https://arxiv.org/abs/1607.01759) is Bag of Tricks for Efficient Text Classification.

The classification examples follow. [Multiclass classification using word2vec/glove + code](https://github.com/dennybritz/cnn-text-classification-tf/issues/69) is an issue on Denny Britz's CNN text classification repo that extends the code to multiclass classification with pre-trained word2vec and GloVe. [word2vec/doc2vec/tfidf code in python for text classification](https://github.com/davidsbatista/text-classification/blob/master/train_classifiers.py) is an example of training supervised classifiers for multi-label text classification using sklearn pipelines. [Lda & word2vec](https://www.kaggle.com/vukglisovic/classification-combining-lda-and-word2vec) is a Kaggle notebook combining LDA and Word2Vec on the Spooky Author Identification data. [Text classification with word2vec](http://nadbordrozd.github.io/blog/2016/05/20/text-classification-with-word2vec/) is the DS lore post on the same idea.

For training the vectors yourself, [Gensim word2vec](https://radimrehurek.com/gensim/models/word2vec.html) is the model page of Gensim, topic modelling for humans, and [another one](http://kavita-ganesan.com/gensim-word2vec-tutorial-starter-code/) is Kavita Ganesan's end-to-end Gensim Word2Vec tutorial, with working code and a dataset.

### Keras: Predict vs Evaluate

After the model is trained, Keras offers two calls that look alike, and this part is the difference between predict and evaluate. The answer comes from [here:](https://www.quora.com/What-is-the-difference-between-keras-evaluate-and-keras-predict), which separates inference outputs from evaluation metrics and loss.

.predict() generates output predictions based on the input you pass it (for example, the predicted characters in the MNIST example)

.evaluate() computes the loss based on the input you pass it, along with any other metrics that you requested in the metrics param when you compiled your model (such as accuracy in the MNIST example)

The MNIST example script is kept at the end of the page. Keras metrics come back here too: [For classification methods - how does keras calculate accuracy, all functions.](https://www.quora.com/How-does-Keras-calculate-accuracy)

### LOSS IN KERAS

The loss that evaluate reports can look odd next to the training loss, and this part is why training loss is higher than testing loss.

The same notes are in [Cross entropy, relative ent, KL-D, JS-D, soft max](../data/information-theory.md#cross-entropy-relative-ent-kl-d-js-d-soft-max) and [Perplexity](../evals/evaluation-metrics.md#perplexity).

[Why is the training loss much higher than the testing loss?](https://keras.io/getting-started/faq/#why-is-the-training-loss-much-higher-than-the-testing-loss) A Keras model has two modes: training and testing. Regularization mechanisms, such as Dropout and L1/L2 weight regularization, are turned off at testing time.

The training loss is the average of the losses over each batch of training data. Because your model is changing over time, the loss over the first batches of an epoch is generally higher than over the last batches. On the other hand, the testing loss for an epoch is computed using the model as it is at the end of the epoch, resulting in a lower loss.

The link for the embedding note above, Word embedding using keras, continuous BOW - CBOW, SKIPGRAM, word2vec - really good, is [https://towardsdatascience.com/understanding-feature-engineering-part-4-deep-learning-methods-for-text-data-96c44370bbfa](https://towardsdatascience.com/understanding-feature-engineering-part-4-deep-learning-methods-for-text-data-96c44370bbfa).


## NVIDIA TF CUDA CUDNN

None of the Keras or TensorFlow code above uses a GPU until the driver stack is in place, so this section is install notes for TensorFlow, CUDA, and cuDNN.

The starting point is [Install TF](https://www.tensorflow.org/install/install_linux#NVIDIARequirements), TensorFlow's page to install TensorFlow with pip. Programs often demand one exact CUDA version as a dependency, which is the problem in [Install cuda on ubuntu](https://devtalk.nvidia.com/default/topic/1030495/cuda-setup-and-installation/install-a-specific-cuda-version-for-ubuntu-16-04/), an NVIDIA forum thread about installing a specific cuda version for ubuntu 16.04. The [official linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/) guide has the installation instructions for the CUDA Toolkit on Linux. If the wrong version is already on the machine, [Replace cuda version](https://askubuntu.com/questions/959835/how-to-remove-cuda-9-0-and-install-cuda-8-0-instead) is how to remove cuda-9.0 and install cuda-8.0 instead, after cuda-9.0 turned out not to be compatible with TensorFlow yet. [Cuda 9 download](https://developer.nvidia.com/cuda-90-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1704&target_type=runfilelocal) is the archive to get CUDA Toolkit 9.0 for Windows, Linux, and Mac OSX.

cuDNN sits on top of CUDA. [Install cudnn](https://askubuntu.com/questions/1033489/the-easy-way-install-nvidia-drivers-cuda-cudnn-and-tensorflow-gpu-on-ubuntu-1) and [Installing everything easily](https://askubuntu.com/questions/1033489/the-easy-way-install-nvidia-drivers-cuda-cudnn-and-tensorflow-gpu-on-ubuntu-1) are the same answer, the easy way to install Nvidia drivers, CUDA, CUDNN and Tensorflow GPU on Ubuntu 18.04. When nvidia-smi then [Failed](https://stackoverflow.com/questions/43022843/nvidia-nvml-driver-library-version-mismatch) to initialize NVML: Driver/library version mismatch, that thread is where the fix is discussed.

## Deprecated links

{% hint style="warning" %}
These links and images no longer work. The original wording is kept here. A same-resource copy, when one was checked, is used above.
{% endhint %}

- The book. This address no longer opens: https://pytorch.org/assets/deep-learning/Deep-Learning-with-PyTorch.pdf
- Pytorch DL course. This address no longer opens: https://atcold.github.io/pytorch-Deep-Learning/
- Stateful LSTM. This address no longer opens: https://github.com/fchollet/keras/blob/master/examples/stateful_lstm.py
- CONV LSTM. This address no longer opens: https://github.com/fchollet/keras/blob/master/examples/conv_lstm.py
- MNIST example. This address no longer opens: https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py
