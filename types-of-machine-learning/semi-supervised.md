# Semi Supervised

This page lists semi-supervised learning surveys, self-training and tri-training methods, and recent consistency-based approaches.

1. [Paper review](https://pdfs.semanticscholar.org/3adc/fd254b271bcc2fb7e2a62d750db17e6c2c08.pdf)
2. [Ruder an overview of proxy labeled for semi supervised (AMAZING)](https://ruder.io/semi-supervised/)
3. Self training
   1. [Self training and tri training](https://github.com/zidik/Self-labeled-techniques-for-semi-supervised-learning)
   2. [Confidence regularized self training](https://github.com/yzou2/CRST)
   3. [Domain adaptation for semantic segmentation using class balanced self-training](https://github.com/yzou2/CBST)
   4. [Self labeled techniques for semi supervised learning](https://github.com/zidik/Self-labeled-techniques-for-semi-supervised-learning)
4. Tri training
   1. [Trinet for semi supervised Deep learning](https://www.ijcai.org/Proceedings/2018/0278.pdf)
   2. [Tri training exploiting unlabeled data using 3 classes](https://www.researchgate.net/publication/3297469_Tri-training_Exploiting_unlabeled_data_using_three_classifiers), paper
   3. [Improving tri training with unlabeled data](https://link.springer.com/chapter/10.1007/978-3-642-25349-2_19)
   4. [Tri training using NN ensemble](https://link.springer.com/chapter/10.1007/978-3-642-31919-8_6)
   5. [Asymmetric try training for unsupervised domain adaptation](https://github.com/corenel/pytorch-atda), [another implementation](https://github.com/vtddggg/ATDA), [another](https://github.com/ksaito-ut/atda), [paper](https://arxiv.org/abs/1702.08400)
   6. [Tri training git](https://github.com/LiangjunFeng/Tri-training)
5. [Fast ai forums](https://forums.fast.ai/t/semi-supervised-learning-ssl-uda-mixmatch-s4l/56826)
6. [UDA GIT](https://github.com/google-research/uda), [paper](https://arxiv.org/abs/1904.12848), [medium\*](https://medium.com/syncedreview/google-brain-cmu-advance-unsupervised-data-augmentation-for-ssl-c0a6157505ce), medium 2 ([has data augmentation articles)](https://medium.com/towards-artificial-intelligence/unsupervised-data-augmentation-6760456db143)
7. [s4l](https://arxiv.org/abs/1905.03670)
8. [Google’s UDM and MixMatch dissected](https://mlexplained.com/2019/06/02/papers-dissected-mixmatch-a-holistic-approach-to-semi-supervised-learning-and-unsupervised-data-augmentation-explained/)- For text classification, the authors used a combination of back translation and a new method called TF-IDF based word replacing.

Back translation consists of translating a sentence into some other intermediate language (e.g. French) and then translating it back to the original language (English in this case). The authors trained an English-to-French and French-to-English system on the WMT 14 corpus.

TF-IDF word replacement replaces words in a sentence at random based on the TF-IDF scores of each word (words with a lower TF-IDF have a higher probability of being replaced).

9. [MixMatch](https://arxiv.org/abs/1905.02249), medium, [2](https://medium.com/@sanjeev.vadiraj/eureka-mixmatch-a-holistic-approach-to-semi-supervised-learning-125b14e82d2f), [3](https://medium.com/@sshleifer/mixmatch-paper-summary-1995f3d11cf), [4](https://medium.com/@literallywords/tl-dr-papers-mixmatch-9dc4cd217121), that works by guessing low-entropy labels for data-augmented unlabeled examples and mixing labeled and unlabeled data using MixUp. We show that MixMatch obtains state-of-the-art results by a large margin across many datasets and labeled data amounts
10. ReMixMatch - [paper](https://arxiv.org/pdf/1911.09785.pdf) is really good. “We improve the recently-proposed “MixMatch” semi-supervised learning algorithm by introducing two new techniques: distribution alignment and augmentation anchoring”
11. [FixMatch](https://amitness.com/2020/03/fixmatch-semi-supervised/) - FixMatch is a recent semi-supervised approach by Sohn et al. from Google Brain that improved the state of the art in semi-supervised learning(SSL). It is a simpler combination of previous methods such as UDA and ReMixMatch.

<figure><img src="../.gitbook/assets/gimg-8fa10d719d15.png" alt=""><figcaption><p>FixMatch semi-supervised learning.</p><p><em>Image via</em> <a href="https://amitness.com/">Amit Chaudhary</a> <em>wrong credit?</em> <a href="mailto:ori@oricohen.com"><em>let me know</em></a></p><p>Credit: <a href="https://lh6.googleusercontent.com/9gNryK4qk-1VHSlpbSFThr0rTnKe6EDiwSDxqDaW4EEx-rIm9LGqs5uGFYHfMsQtJWd9Ls_NAnap_wHHAe_qOBGcZgMJ7ruGkuxv2nIY8AP1mq82PgDxtgmsVO59G_rDOnoNvUDk">copied from the original hosted image</a>.</p></figcaption></figure>

12. [Curriculum Labeling: Self-paced Pseudo-Labeling for Semi-Supervised Learning](https://arxiv.org/pdf/2001.06001.pdf)
13. [FAIR](https://ai.facebook.com/blog/billion-scale-semi-supervised-learning/) [2](https://ai.facebook.com/blog/mapping-the-world-to-help-aid-workers-with-weakly-semi-supervised-learning/) original, [Summarization of FAIR’s student teacher weak/ semi supervision](https://analyticsindiamag.com/how-to-do-machine-learning-when-data-is-unlabelled/)
14. [Leveraging Just a Few Keywords for Fine-Grained Aspect Detection Through Weakly Supervised Co-Training](https://www.aclweb.org/anthology/D19-1468.pdf)
15. [Fidelity-Weighted](https://openreview.net/forum?id=B1X0mzZCW) Learning - “fidelity-weighted learning” (FWL), a semi-supervised student- teacher approach for training deep neural networks using weakly-labeled data. FWL modulates the parameter updates to a student network (trained on the task we care about) on a per-sample basis according to the posterior confidence of its label-quality estimated by a teacher (who has access to the high-quality labels). Both student and teacher are learned from the data."
16. [Unproven student teacher git](https://github.com/EricHe98/Teacher-Student-Training)
17. [A really nice student teacher git with examples](https://github.com/yuanli2333/Teacher-free-Knowledge-Distillation).

<figure><img src="../.gitbook/assets/gimg-533b3ac2c7bc.png" alt=""><figcaption><p>Image by yuanli2333. wrong credit? let me know</p><p>Credit: <a href="https://lh6.googleusercontent.com/tlo5HqMjycySNl9Pbmr-uW-azozTC5cc7if-7r6-0LCeRJO2snTm-hsEf7mUpr1hp6wSnIVy6GnqFG6pEbxTPgu9fjjHP6gtn1dKQCwEI-x12UxYzWBWfidqMwVxZetA10VznMhs">copied from the original hosted image</a>.</p></figcaption></figure>

18. [Teacher student for tri training for unlabeled data exploitation](https://arxiv.org/abs/1909.11233)

<figure><img src="../.gitbook/assets/gimg-ebd316fe87ec.png" alt=""><figcaption><p>Image by the late Dr. Hui Li, @ SAS. wrong credit? let me know</p><p>Credit: <a href="https://lh6.googleusercontent.com/J648WfIzGrbgjfSCK4S4lkCFbPWrSq6vwN1KERJ-yk5E21Jl3ZIeX7V98LS6rNIuY1Yc631oKIX-8H-dUyoqBHSoQEerZG_KnKpwKWbhk5IHK3G0nTpCZ4ddGYGP-beBydYVOkKx">copied from the original hosted image</a>.</p></figcaption></figure>

## Deprecated links

{% hint style="warning" %}
These links and images no longer work. The original wording is kept here. A same-resource copy, when one was checked, is used above.
{% endhint %}


- medium. This address no longer opens: https://towardsdatascience.com/a-fastai-pytorch-implementation-of-mixmatch-314bb30d0f99

- paper. This address no longer opens: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.487.2431&rep=rep1&type=pdf
