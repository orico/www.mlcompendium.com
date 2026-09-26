# Semi Supervised

This page lists semi-supervised learning surveys, self-training and tri-training methods, and recent consistency-based approaches. It is for training with lots of unlabeled data and only a little labeled data.

The same notes are in [COP-CLUSTERING](../predictive-ml/clustering-algorithms.md#cop-clustering), [Label Propagation / Spreading](label-algorithms.md#label-propagation--spreading), and [Weakly Supervised](weakly-supervised.md).

- # Zhi-Hua Zhou National Key Laboratory for Novel Software Technology Nanjing University, Nanjing 210023, China zhouzh@nju.edu.cn. [Paper review](https://pdfs.semanticscholar.org/3adc/fd254b271bcc2fb7e2a62d750db17e6c2c08.pdf)
- While unsupervised learning is still elusive, researchers have made a lot of progress in semi-supervised learning. [Ruder an overview of proxy labeled for semi supervised (AMAZING)](https://ruder.io/semi-supervised/)
3. Self training
 - Contribute to zidik/Self-labeled-techniques-for-semi-supervised-learning development by creating an account on GitHub. [Self training and tri training](https://github.com/zidik/Self-labeled-techniques-for-semi-supervised-learning)
 - Code for <Confidence Regularized Self-Training> in ICCV19 (Oral) - yzou2/CRST. [Confidence regularized self training](https://github.com/yzou2/CRST)
 - Code for <Domain Adaptation for Semantic Segmentation via Class-Balanced Self-Training> in ECCV18 - yzou2/CBST. [Domain adaptation for semantic segmentation using class balanced self-training](https://github.com/yzou2/CBST)
 - Contribute to zidik/Self-labeled-techniques-for-semi-supervised-learning development by creating an account on GitHub. [Self labeled techniques for semi supervised learning](https://github.com/zidik/Self-labeled-techniques-for-semi-supervised-learning)
4. Tri training
 - ## National Key Laboratory for Novel Software Technology Nanjing University, Nanjing 210023, China. [Trinet for semi supervised Deep learning](https://www.ijcai.org/Proceedings/2018/0278.pdf)
 - ResearchGate - Temporarily Unavailable. ResearchGate - Temporarily Unavailable. [Tri training exploiting unlabeled data using 3 classes](https://www.researchgate.net/publication/3297469_Tri-training_Exploiting_unlabeled_data_using_three_classifiers)
 - Client Challenge. Client Challenge. [Improving tri training with unlabeled data](https://link.springer.com/chapter/10.1007/978-3-642-25349-2_19)
 - Client Challenge. Client Challenge. [Tri training using NN ensemble](https://link.springer.com/chapter/10.1007/978-3-642-31919-8_6)
 - A PyTorch implementation for Asymmetric Tri-training for Unsupervised Domain Adaptation - corenel/pytorch-atda. [Asymmetric try training for unsupervised domain adaptation](https://github.com/corenel/pytorch-atda)
 - Unofficial Implement of Asymmetric Tri-training for Unsupervised Domain Adaptation - vtddggg/ATDA. [another implementation](https://github.com/vtddggg/ATDA)
 - Implemenation of Asymmetric-TriTraining by Tensorflow - ksaito-ut/atda. [another](https://github.com/ksaito-ut/atda)
 - Abstract page for arXiv paper 1702.08400: Asymmetric Tri-training for Unsupervised Domain Adaptation. [paper](https://arxiv.org/abs/1702.08400)
 - The python implementation of tri-triaing. [Tri training git](https://github.com/LiangjunFeng/Tri-training)
- If you have access to lots of unlabeled data, but a relatively small amount of labeled data, Semi-Supervised Learning (SSL) might be really useful. [Fast ai forums](https://forums.fast.ai/t/semi-supervised-learning-ssl-uda-mixmatch-s4l/56826)
- GitHub - google-research/uda: Unsupervised Data Augmentation (UDA). [UDA GIT](https://github.com/google-research/uda)
- Abstract page for arXiv paper 1904.12848: Unsupervised Data Augmentation for Consistency Training. [paper](https://arxiv.org/abs/1904.12848)
- [medium\*](https://medium.com/syncedreview/google-brain-cmu-advance-unsupervised-data-augmentation-for-ssl-c0a6157505ce)
- Unsupervised Data Augmentation. medium 2 ( [has data augmentation articles)](https://medium.com/towards-artificial-intelligence/unsupervised-data-augmentation-6760456db143)
- Abstract page for arXiv paper 1905.03670: S4L: Self-Supervised Semi-Supervised Learning. [s4l](https://arxiv.org/abs/1905.03670)
8. [Google’s UDM and MixMatch dissected](https://mlexplained.com/2019/06/02/papers-dissected-mixmatch-a-holistic-approach-to-semi-supervised-learning-and-unsupervised-data-augmentation-explained/)- For text classification, the authors used a combination of back translation and a new method called TF-IDF based word replacing.

 Back translation consists of translating a sentence into some other intermediate language (e.g. French) and then translating it back to the original language (English in this case). The authors trained an English-to-French and French-to-English system on the WMT 14 corpus.

 TF-IDF word replacement replaces words in a sentence at random based on the TF-IDF scores of each word (words with a lower TF-IDF have a higher probability of being replaced).

9. [MixMatch](https://arxiv.org/abs/1905.02249), medium, [2](https://medium.com/@sanjeev.vadiraj/eureka-mixmatch-a-holistic-approach-to-semi-supervised-learning-125b14e82d2f), [3](https://medium.com/@sshleifer/mixmatch-paper-summary-1995f3d11cf), [4](https://medium.com/@literallywords/tl-dr-papers-mixmatch-9dc4cd217121), that works by guessing low-entropy labels for data-augmented unlabeled examples and mixing labeled and unlabeled data using MixUp. We show that MixMatch obtains state-of-the-art results by a large margin across many datasets and labeled data amounts
10. ReMixMatch - [paper](https://arxiv.org/pdf/1911.09785.pdf) is really good. “We improve the recently-proposed “MixMatch” semi-supervised learning algorithm by introducing two new techniques: distribution alignment and augmentation anchoring”
11. [FixMatch](https://amitness.com/2020/03/fixmatch-semi-supervised/) - FixMatch is a recent semi-supervised approach by Sohn et al. from Google Brain that improved the state of the art in semi-supervised learning(SSL). It is a simpler combination of previous methods such as UDA and ReMixMatch.

<figure><img src="../.gitbook/assets/gimg-8fa10d719d15.png" alt=""><figcaption><p>FixMatch semi-supervised learning.</p><p><em>Image via</em> <a href="https://amitness.com/">Amit Chaudhary</a> <em>wrong credit?</em> <a href="mailto:ori@oricohen.com"><em>let me know</em></a></p><p>Credit: <a href="https://lh6.googleusercontent.com/9gNryK4qk-1VHSlpbSFThr0rTnKe6EDiwSDxqDaW4EEx-rIm9LGqs5uGFYHfMsQtJWd9Ls_NAnap_wHHAe_qOBGcZgMJ7ruGkuxv2nIY8AP1mq82PgDxtgmsVO59G_rDOnoNvUDk">copied from the original hosted image</a>.</p></figcaption></figure>

- 2001.06001v2.pdf. [Curriculum Labeling: Self-paced Pseudo-Labeling for Semi-Supervised Learning](https://arxiv.org/pdf/2001.06001.pdf)
- Facebook AI is developing alternative ways to train our AI systems so that we can do more with less labeled training data overall. [FAIR](https://ai.facebook.com/blog/billion-scale-semi-supervised-learning/)
- To help humanitarian organizations, Facebook AI researchers have created the world’s most detailed population density maps of Africa. [2](https://ai.facebook.com/blog/mapping-the-world-to-help-aid-workers-with-weakly-semi-supervised-learning/)
- AIM — India. AIM — India. original [Summarization of FAIR’s student teacher weak/ semi supervision](https://analyticsindiamag.com/how-to-do-machine-learning-when-data-is-unlabelled/)
- Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing , pages 4611–4621, Hong Kong, China, November 3–7, 2019. [Leveraging Just a Few Keywords for Fine-Grained Aspect Detection Through Weakly Supervised Co-Training](https://www.aclweb.org/anthology/D19-1468.pdf)
15. [Fidelity-Weighted](https://openreview.net/forum?id=B1X0mzZCW) Learning - “fidelity-weighted learning” (FWL), a semi-supervised student- teacher approach for training deep neural networks using weakly-labeled data. FWL modulates the parameter updates to a student network (trained on the task we care about) on a per-sample basis according to the posterior confidence of its label-quality estimated by a teacher (who has access to the high-quality labels). Both student and teacher are learned from the data."
- This repository stores the files used for my summer internship's work on "teacher-student learning", an experimental method for training deep neural networks using a trained teacher model. [Unproven student teacher git](https://github.com/EricHe98/Teacher-Student-Training)
- Knowledge Distillation: CVPR2020 Oral, Revisiting Knowledge Distillation via Label Smoothing Regularization - yuanli2333/Teacher-free-Knowledge-Distillation. [A really nice student teacher git with examples](https://github.com/yuanli2333/Teacher-free-Knowledge-Distillation)

<figure><img src="../.gitbook/assets/gimg-533b3ac2c7bc.png" alt=""><figcaption><p>Image by yuanli2333. wrong credit? let me know</p><p>Credit: <a href="https://lh6.googleusercontent.com/tlo5HqMjycySNl9Pbmr-uW-azozTC5cc7if-7r6-0LCeRJO2snTm-hsEf7mUpr1hp6wSnIVy6GnqFG6pEbxTPgu9fjjHP6gtn1dKQCwEI-x12UxYzWBWfidqMwVxZetA10VznMhs">copied from the original hosted image</a>.</p></figcaption></figure>

- Abstract page for arXiv paper 1909.11233: Teacher-Student Learning Paradigm for Tri-training: An Efficient Method for Unlabeled Data Exploitation. [Teacher student for tri training for unlabeled data exploitation](https://arxiv.org/abs/1909.11233)

<figure><img src="../.gitbook/assets/gimg-ebd316fe87ec.png" alt=""><figcaption><p>Image by the late Dr. Hui Li, @ SAS. wrong credit? let me know</p><p>Credit: <a href="https://lh6.googleusercontent.com/J648WfIzGrbgjfSCK4S4lkCFbPWrSq6vwN1KERJ-yk5E21Jl3ZIeX7V98LS6rNIuY1Yc631oKIX-8H-dUyoqBHSoQEerZG_KnKpwKWbhk5IHK3G0nTpCZ4ddGYGP-beBydYVOkKx">copied from the original hosted image</a>.</p></figcaption></figure>

- Abstract — In many practical data mining applications such as web page classification, unlabeled training examples are readily available but labeled ones are fairly expensive to obtain. paper. [http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.487.2431&rep=rep1&type=pdf](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.487.2431&rep=rep1&type=pdf)

## Deprecated links

{% hint style="warning" %}
These links and images no longer work. The original wording is kept here. A same-resource copy, when one was checked, is used above.
{% endhint %}


- medium. This address no longer opens: https://towardsdatascience.com/a-fastai-pytorch-implementation-of-mixmatch-314bb30d0f99
