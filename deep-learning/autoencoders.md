# AUTOENCODERS

An autoencoder learns to squeeze data into a small code and rebuild it, which makes it a nonlinear answer to the dimensionality reduction that PCA and ICA already do. This page starts with the basic autoencoder, its comparison to PCA, and its reconstruction uses such as denoising and anomaly detection, then moves to variational autoencoders.

The reconstruction error is also what makes autoencoders useful for spotting unusual rows. The same notes are in [Anomaly Detection](../predictive-ml/anomaly-detection.md) and [Fraud Detection](../ai-product/fraud-detection.md).

We often use ICA or PCA to extract features from the high-dimensional data, and the autoencoder is another way to achieve the same purpose in deep learning. [How to use AE for dimensionality reduction + code](https://statcompute.wordpress.com/2017/01/15/autoencoder-for-dimensionality-reduction/) shows it using keras’ functional API. The broad tour is the [Keras.io blog post about AE’s](https://blog.keras.io/building-autoencoders-in-keras.html), Building Autoencoders in Keras - regular, deep, sparse, regularized, cnn, variational. A keras.io replicate post but explains AE quite nicely, and Examples of vanilla, multi layer, CNN and sparse AE’s used to follow it; that address is kept at the end of the page.

Convolutional autoencoders get their own walkthrough. [Another example of CNN-AE](https://hackernoon.com/autoencoders-deep-learning-bits-1-11731e200694) is Autoencoders, Deep Learning bits #1, featuring data compression, image reconstruction and segmentation (with examples!). Another AE tutorial sat beside it.

The comparison with PCA is where the idea becomes clear. Hinton’s coursera course on PCA vs AE, basically some info about what PCA does - maximizing variance and projecting and then what AE does and can do to achieve similar but non-linear dense representations; the lecture address is kept at the end. To see the difference, [A great tutorial on how does the clusters look like after applying PCA/ICA/AE](https://www.kaggle.com/den3b81/2d-visualization-pca-ica-vs-autoencoders) is a Kaggle notebook doing 2D visualization with PCA and ICA against autoencoders on the Mercedes-Benz Greener Manufacturing data. Another great presentation on PCA vs AE, summarized in the KPCA section of this notebook, is kept at the end too. +[another one](https://www.cs.toronto.edu/~urtasun/courses/CSC411/14_pca.pdf) is the Urtasun and Zemel CSC 411 lecture on PCA and autoencoders, and +[StackE](https://stats.stackexchange.com/questions/261265/factor-analysis-vs-autoencoders)xchange asks what the difference between factor analysis and autoencoders is, and when to use one and not the other.

From comparison back to code: autoencoders have become an intriguing tool for data compression, and implementing them in Keras is surprisingly straightforward, which is Trujillo Herman's point in the [Autoencoder tutorial with python code and how to encode after](https://ramhiser.com/post/2018-05-14-autoencoders-with-keras/). The same encoder-decoder shape carries over to sequences, and Jason Brownlee's [mastery](https://machinelearningmastery.com/encoder-decoder-attention-sequence-to-sequence-prediction-keras/) post shows how to develop an encoder-decoder model with attention in Keras, where attention addresses the limits of the architecture on long sequences. When the code has to be very small, [Git code for low dimensional auto encoder](https://github.com/Mylittlerapture/Low-Dimensional-Autoencoder) is a Keras model training method that allows to get good results for heavily dimensionality reduction.

Denoising is the same idea at language scale. [Bart denoising AE](https://arxiv.org/pdf/1910.13461.pdf), sequence to sequence pre training for NL generation translation and comprehension. An attention based seq to seq auto encoder has its code in the [git](https://github.com/wanasit/katakana) repo that trains a machine to write Katakana using the sequence-to-sequence technique; its write-up is kept at the end.

The reconstruction error then becomes a detector: [AE for anomaly detection, fraud detection](https://medium.com/@curiousily/credit-card-fraud-detection-using-autoencoders-in-keras-tensorflow-for-hackers-part-vii-20e0c85301bd) is Venelin Valkov's credit card fraud detection using autoencoders in Keras, part VII of TensorFlow for Hackers.

## Variational AE

A plain autoencoder compresses, but it cannot be asked to generate something specific, and the variational autoencoder builds on the material above to fix that.

The [Simple explanation](https://medium.com/@dmonn/what-are-variational-autoencoders-a-simple-explanation-ea7dccafb0e3) is still marked Unread; it starts from GANs that generate random faces and asks how to generate a specific one. [Pixel art VAE](https://mlexplained.wordpress.com/category/generative-models/vae/) is the VAE category of Machine Learning Insights. Unread - another VAE sat next to it. [Pixel GAN VAE](https://medium.com/@Synced/pixelgan-autoencoders-17496632b755) is Synced on the PixelGAN Autoencoder, whose generative path is a convolutional autoregressive network on pixels conditioned on a latent code, with a GAN imposing a prior on that code. [Disentangled VAE](https://www.youtube.com/watch?v=9zKuYvjFFS8) - improves VAE, and the video is Arxiv Insights on variational autoencoders.

At language scale, Optimus - [pretrained VAE](https://github.com/ophiry/Optimus) is the first large-scale pre-trained VAE language model. Its [paper](https://arxiv.org/abs/2004.04092) is Optimus: Organizing Sentences via Pre-trained Modeling of a Latent Space, and the [Microsoft blog](https://www.microsoft.com/en-us/research/blog/a-deep-generative-model-trifecta-three-advances-that-work-towards-harnessing-large-scale-power/) places Optimus next to FQ-GAN and Prevalent as advances in large-scale deep generative models.

![](<../.gitbook/assets/image).png>)

See also [Cleaning Data With AI Denoisers](https://pub.towardsai.net/cleaning-data-with-ai-denoisers-be1bdea0fe20) (October 2024), on reducing noise in images, audio, and video beyond traditional filters and statistical methods.

For a longer look inside the model, the Towards Data Science post deep-inside-autoencoders-7e41f319999f is at [https://towardsdatascience.com/deep-inside-autoencoders-7e41f319999f](https://towardsdatascience.com/deep-inside-autoencoders-7e41f319999f).

## Deprecated links

{% hint style="warning" %}
These links and images no longer work. The original wording is kept here. A same-resource copy, when one was checked, is used above.
{% endhint %}
- Towards Data Science: how-to-reduce-image-noises-by-autoencoder-65d5e6de543. This address no longer opens: https://towardsdatascience.com/how-to-reduce-image-noises-by-autoencoder-65d5e6de543
- Towards Data Science: teaching-a-variational-autoencoder-vae-to-draw-mnist-characters-978675c95776. This address no longer opens: https://towardsdatascience.com/teaching-a-variational-autoencoder-vae-to-draw-mnist-characters-978675c95776
- Examples of vanilla, multi layer, CNN and sparse AE’s. This address no longer opens: https://wiseodd.github.io/techblog/2016/12/03/autoencoders/
- Hinton’s coursera course. This address no longer opens: https://www.coursera.org/learn/neural-networks/lecture/JiT1i/from-pca-to-autoencoders-5-mins
- Another great presentation on PCA vs AE,. This address no longer opens: https://web.cs.hacettepe.edu.tr/~aykut/classes/fall2016/bbm406/slides/l25-kernel_pca.pdf
- Attention based seq to seq auto encoder. This address no longer opens: https://wanasit.github.io/attention-based-sequence-to-sequence-in-keras.html
