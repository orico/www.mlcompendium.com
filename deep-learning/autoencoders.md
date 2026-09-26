## AUTOENCODERS

This section collects notes on autoencoders.

The same notes are in [Anomaly Detection](../predictive-ml/anomaly-detection.md) and [Fraud Detection](../ai-product/fraud-detection.md).


1. [How to use AE for dimensionality reduction + code](https://statcompute.wordpress.com/2017/01/15/autoencoder-for-dimensionality-reduction/) - using keras’ functional API
2. [Keras.io blog post about AE’s](https://blog.keras.io/building-autoencoders-in-keras.html) - regular, deep, sparse, regularized, cnn, variational
   1. A keras.io replicate post but explains AE quite nicely.
3. Examples of vanilla, multi layer, CNN and sparse AE’s
4. [Another example of CNN-AE](https://hackernoon.com/autoencoders-deep-learning-bits-1-11731e200694)
5. Another AE tutorial
6. Hinton’s coursera course on PCA vs AE, basically some info about what PCA does - maximizing variance and projecting and then what AE does and can do to achieve similar but non-linear dense representations
7. [A great tutorial on how does the clusters look like after applying PCA/ICA/AE](https://www.kaggle.com/den3b81/2d-visualization-pca-ica-vs-autoencoders)
8. Another great presentation on PCA vs AE, summarized in the KPCA section of this notebook. +[another one](https://www.cs.toronto.edu/~urtasun/courses/CSC411/14_pca.pdf) +[StackE](https://stats.stackexchange.com/questions/261265/factor-analysis-vs-autoencoders)xchange
9. [Autoencoder tutorial with python code and how to encode after](https://ramhiser.com/post/2018-05-14-autoencoders-with-keras/), [mastery](https://machinelearningmastery.com/encoder-decoder-attention-sequence-to-sequence-prediction-keras/)
10. [Git code for low dimensional auto encoder](https://github.com/Mylittlerapture/Low-Dimensional-Autoencoder)
11. [Bart denoising AE](https://arxiv.org/pdf/1910.13461.pdf), sequence to sequence pre training for NL generation translation and comprehension.
12. Attention based seq to seq auto encoder, [git](https://github.com/wanasit/katakana)

[AE for anomaly detection, fraud detection](https://medium.com/@curiousily/credit-card-fraud-detection-using-autoencoders-in-keras-tensorflow-for-hackers-part-vii-20e0c85301bd)

## Variational AE

This section collects notes on variational ae.


1. Unread - [Simple explanation](https://medium.com/@dmonn/what-are-variational-autoencoders-a-simple-explanation-ea7dccafb0e3)
2. [Pixel art VAE](https://mlexplained.wordpress.com/category/generative-models/vae/)
3. Unread - another VAE
4. [Pixel GAN VAE](https://medium.com/@Synced/pixelgan-autoencoders-17496632b755)
5. [Disentangled VAE](https://www.youtube.com/watch?v=9zKuYvjFFS8) - improves VAE
6. Optimus - [pretrained VAE](https://github.com/ophiry/Optimus), [paper](https://arxiv.org/abs/2004.04092), [Microsoft blog](https://www.microsoft.com/en-us/research/blog/a-deep-generative-model-trifecta-three-advances-that-work-towards-harnessing-large-scale-power/)

![](<../.gitbook/assets/image).png>)


See also [Cleaning Data With AI Denoisers](https://pub.towardsai.net/cleaning-data-with-ai-denoisers-be1bdea0fe20) (October 2024).
