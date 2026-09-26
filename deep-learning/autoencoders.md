# AUTOENCODERS

This page collects notes on autoencoders for dimensionality reduction and related reconstruction setups.
Variational autoencoders follow after the basic autoencoder material below.

The same notes are in [Anomaly Detection](../predictive-ml/anomaly-detection.md) and [Fraud Detection](../ai-product/fraud-detection.md).

- We often use ICA or PCA to extract features from the high-dimensional data. - using keras’ functional API [How to use AE for dimensionality reduction + code](https://statcompute.wordpress.com/2017/01/15/autoencoder-for-dimensionality-reduction/)
2. [Keras.io blog post about AE’s](https://blog.keras.io/building-autoencoders-in-keras.html) - regular, deep, sparse, regularized, cnn, variational
 1. A keras.io replicate post but explains AE quite nicely.
3. Examples of vanilla, multi layer, CNN and sparse AE’s
- <em>Featured</em>: data compression, image reconstruction and segmentation (with examples!). [Another example of CNN-AE](https://hackernoon.com/autoencoders-deep-learning-bits-1-11731e200694)
5. Another AE tutorial
6. Hinton’s coursera course on PCA vs AE, basically some info about what PCA does - maximizing variance and projecting and then what AE does and can do to achieve similar but non-linear dense representations
- Explore and run AI code with Kaggle Notebooks | Using data from Mercedes-Benz Greener Manufacturing. [A great tutorial on how does the clusters look like after applying PCA/ICA/AE](https://www.kaggle.com/den3b81/2d-visualization-pca-ica-vs-autoencoders)
8. Another great presentation on PCA vs AE, summarized in the KPCA section of this notebook. +[another one](https://www.cs.toronto.edu/~urtasun/courses/CSC411/14_pca.pdf) +[StackE](https://stats.stackexchange.com/questions/261265/factor-analysis-vs-autoencoders)xchange
- Autoencoders have become an intriguing tool for data compression, and implementing them in Keras is surprisingly straightforward, by Trujillo Herman. [Autoencoder tutorial with python code and how to encode after](https://ramhiser.com/post/2018-05-14-autoencoders-with-keras/)
- How to Develop an Encoder-Decoder Model with Attention in Keras - MachineLearningMastery.com, by Jason Brownlee. [mastery](https://machinelearningmastery.com/encoder-decoder-attention-sequence-to-sequence-prediction-keras/)
- Keras model training method that allows to get good results for heavily dimensionality reduction - volotat/Low-Dimensional-Autoencoder. [Git code for low dimensional auto encoder](https://github.com/Mylittlerapture/Low-Dimensional-Autoencoder)
11. [Bart denoising AE](https://arxiv.org/pdf/1910.13461.pdf), sequence to sequence pre training for NL generation translation and comprehension.
- Attention based seq to seq auto encoder. [git](https://github.com/wanasit/katakana)

[AE for anomaly detection, fraud detection](https://medium.com/@curiousily/credit-card-fraud-detection-using-autoencoders-in-keras-tensorflow-for-hackers-part-vii-20e0c85301bd)

## Variational AE

This section collects variational autoencoder notes that build on the autoencoder material above.

- Unread - [Simple explanation](https://medium.com/@dmonn/what-are-variational-autoencoders-a-simple-explanation-ea7dccafb0e3)
- VAE – Machine Learning Insights. [Pixel art VAE](https://mlexplained.wordpress.com/category/generative-models/vae/)
3. Unread - another VAE
4. [Pixel GAN VAE](https://medium.com/@Synced/pixelgan-autoencoders-17496632b755)
5. [Disentangled VAE](https://www.youtube.com/watch?v=9zKuYvjFFS8) - improves VAE
- Optimus: the first large-scale pre-trained VAE language model - ophiry/Optimus. Optimus - [pretrained VAE](https://github.com/ophiry/Optimus)
- Abstract page for arXiv paper 2004.04092: Optimus: Organizing Sentences via Pre-trained Modeling of a Latent Space. [paper](https://arxiv.org/abs/2004.04092)
- Microsoft. [Microsoft blog](https://www.microsoft.com/en-us/research/blog/a-deep-generative-model-trifecta-three-advances-that-work-towards-harnessing-large-scale-power/)

![](<../.gitbook/assets/image).png>)

See also [Cleaning Data With AI Denoisers](https://pub.towardsai.net/cleaning-data-with-ai-denoisers-be1bdea0fe20) (October 2024).

- Towards Data Science: deep-inside-autoencoders-7e41f319999f. [https://towardsdatascience.com/deep-inside-autoencoders-7e41f319999f](https://towardsdatascience.com/deep-inside-autoencoders-7e41f319999f)

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
