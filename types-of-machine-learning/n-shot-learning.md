# N-Shot Learning

### N-SHOT LEARNING

1. [Zero shot, one shot, few shot](https://blog.floydhub.com/n-shot-learning/) (siamese is one shot)

### ZERO SHOT LEARNING

[Instead of using class labels](https://www.youtube.com/watch?v=jBnCcr-3bXc), we use some kind of vector representation for the classes, taken from a co-occurrence-after-svd or word2vec. - quite clever. This enables us to figure out if a new unseen class is near one of the known supervised classes. KNN can be used or some other distance-based classifier. Can we use word2vec for similarity measurements of new classes?\
![](https://lh3.googleusercontent.com/Rim9\_QVRRSj7eJTYeCcs1FfXzf-k7Qp2Wdmgcd1H-N\_ZZ6-krl1O3pH8GLkZMAVk2eQ5Ye\_Os2nUMqqsKzq92iP2rtlt1lix\_KnsMQsSrpMDPYcqI02TU0RrcZZMBmqfiLQj7xeN)

Image by [Dr. Timothy Hospedales, Yandex](https://www.youtube.com/watch?v=jBnCcr-3bXc)

for classification, we can use nearest neighbour or manifold-based labeling propagation.\
![](https://lh4.googleusercontent.com/nwZTsm4rfemR9-hNsyVpn1sFc4jJ9b2RAf\_gZKds51ki81crI9\_C6L5xI5M1F7OMK6a2Et7vS4JKWwtFMODKj\_RfQ6jTmCtrSPfQb4jMoZrZ5ZEoIm4uxublmBTgkJLkSvsMqYYF)

Image by [Dr. Timothy Hospedales, Yandex](https://www.youtube.com/watch?v=jBnCcr-3bXc)

Multiple category vectors? Multilabel zero-shot also in the video

#### GPT3 is ZERO, ONE, FEW

1. [Prompt Engineering Tips & Tricks](https://blog.andrewcantino.com/blog/2021/04/21/prompt-engineering-tips-and-tricks/)
2. [Open GPT3 prompt engineering](https://medium.com/swlh/openai-gpt-3-and-prompt-engineering-dcdc2c5fcd29)
