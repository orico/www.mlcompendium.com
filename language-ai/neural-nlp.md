# Neural NLP

This page collects CNN-for-text and sequence-to-sequence resources.
The sections below group CNN-for-text material and sequence-to-sequence resources.

The same notes are in [Deep Learning for NLP](../deep-learning/deep-neural-nets.md#deep-learning-for-nlp).

## CONVOLUTION NEURAL NETS (CNN)

This section links CNN approaches for text and 1D sequences.

The same notes are in [CONVOLUTIONAL NEURAL NET](../deep-learning/convolutional-nets.md#convolutional-neural-net).

- [Cnn for text](https://medium.com/@TalPerry/convolutional-methods-for-text-d5260fd5675f) tal perry
2. 1D CNN using KERAS

## SEQ2SEQ SEQUENCE TO SEQUENCE

This section is about encoder–decoder seq2seq, teacher forcing, and related tutorials.

The same notes are in [Attention](../deep-learning/attention.md), [Decoding Algorithms For NLP](decoding-algorithms-for-nlp.md), and [Recurrent Neural Net (RNN)](../deep-learning/recurrent-nets.md#recurrent-neural-net-rnn).

1. [Keras blog](https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html) — char-level, token-using embedding layer, teacher forcing
- [Teacher forcing explained](https://medium.com/data-science/what-is-teacher-forcing-3da6217fed1c)
- [Same as keras but with token-level](https://medium.com/data-science/machine-translation-with-the-seq2seq-model-different-approaches-f078081aaa37)
- [Medium on char, word, byte-level](https://medium.com/@petepeeradejtanruangporn/experimenting-with-neural-machine-translation-for-thai-1681fd2b375a)
- How to Develop an Encoder-Decoder Model for Sequence-to-Sequence Prediction in Keras - MachineLearningMastery.com. [Mastery on enc-dec using the keras method](https://machinelearningmastery.com/develop-encoder-decoder-model-sequence-sequence-prediction-keras/)
- How to Develop a Seq2Seq Model for Neural Machine Translation in Keras - MachineLearningMastery.com. and on [neural translation](https://machinelearningmastery.com/define-encoder-decoder-sequence-sequence-model-neural-machine-translation-keras/)
- Contribute to samurainote/seq2seq_translate_slackbot development by creating an account on GitHub. [Machine translation git from eng to jap](https://github.com/samurainote/seq2seq_translate_slackbot/blob/master/seq2seq_translate.py)
- Contribute to samurainote/seq2seq_translate_slackbot development by creating an account on GitHub. [another](https://github.com/samurainote/seq2seq_translate_slackbot)
- and its [medium](https://medium.com/data-science/how-to-implement-seq2seq-lstm-model-in-keras-shortcutnlp-6f355f3e5639)

<figure><img src="../.gitbook/assets/gimg-e7f594cff87a.png" alt=""><figcaption><p>Sequence-to-sequence machine translation.</p><p>Credit: <a href="https://lh6.googleusercontent.com/bcrIRzPLlcnQBl1zWR2s0_tB-NNEQxd8ZNQK8oK2NJsc29Fv6RdfKynfjHeNsSvl5d0SqK55k8xN1NAIrvEcnFEtpfZCfOHZzCSFLKmxeBWXn903VOJKiKTMV4Ynm_HL6Sgls2BN">copied from the original hosted image</a>.</p></figcaption></figure>

1. [Incorporating Copying Mechanism in Sequence-to-Sequence Learning](https://arxiv.org/abs/1603.06393) — In this paper, we incorporate copying into neural network-based Seq2Seq learning and propose a new model called CopyNet with encoder-decoder structure. CopyNet can nicely integrate the regular way of word generation in the decoder with the new copying mechanism which can choose sub-sequences in the input sequence and put them at proper places in the output sequence.

## Deprecated links

{% hint style="warning" %}
These links and images no longer work. The original wording is kept here. A same-resource copy, when one was checked, is used above.
{% endhint %}

- Teacher forcing explained This address no longer opens: https://towardsdatascience.com/what-is-teacher-forcing-3da6217fed1c
- Same as keras but with token-level This address no longer opens: https://towardsdatascience.com/machine-translation-with-the-seq2seq-model-different-approaches-f078081aaa37
- its medium This address no longer opens: https://towardsdatascience.com/how-to-implement-seq2seq-lstm-model-in-keras-shortcutnlp-6f355f3e5639
- 1D CNN using KERAS This address no longer opens: https://blog.goodaudience.com/introduction-to-1d-convolutional-neural-networks-in-keras-for-time-sequences-3a7ff801a2cf
