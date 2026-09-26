BERT/ROBERTA

1. [Do attention heads in bert roberta track syntactic dependencies?](https://medium.com/@phu_pmh/do-attention-heads-in-bert-track-syntactic-dependencies-81c8a9be311a) - tl;dr: The attention weights between tokens in BERT/RoBERTa bear similarity to some syntactic dependency relations, but the results are less conclusive than we’d like as they don’t significantly outperform linguistically uninformed baselines for all types of dependency relations. In the case of MAX, our results indicate that specific heads in the BERT models may correspond to certain dependency relations, whereas for MST, we find much less support “generalist” heads whose attention weights correspond to a full syntactic dependency structure.

In both cases, the metrics do not appear to be representative of the extent of linguistic knowledge learned by the BERT models, based on their strong performance on many NLP tasks. Hence, our takeaway is that while we can tease out some structure from the attention weights of BERT models using the above methods, studying the attention weights alone is unlikely to give us the full picture of BERT’s strength processing natural language.

### ELMO
This section lists ELMo tutorials, AllenNLP resources, and transfer-learning write-ups.

The same notes are in [Language modeling](language-embeddings.md#language-modeling).

1. [Short tutorial on elmo, pretrained, new data, incremental(finetune?)](https://github.com/PrashantRanjan09/Elmo-Tutorial), [using elmo pretrained](https://github.com/PrashantRanjan09/WordEmbeddings-Elmo-Fasttext-Word2Vec)
2. [Why you cant use elmo to encode words (contextualized)](https://github.com/allenai/allennlp/issues/1737)
3. [Vidhya on elmo](https://www.analyticsvidhya.com/blog/2019/03/learn-to-use-elmo-to-extract-features-from-text/) - everything you want to know with code
4. [Sebastien ruder on language modeling embeddings for the purpose of transfer learning, ELMO, ULMFIT, open AI transformer, BILSTM,](https://thegradient.pub/nlp-imagenet/)
5. [Another good tutorial on elmo](http://www.realworldnlpbook.com/blog/improving-sentiment-analyzer-using-elmo.html).
6. [ELMO](https://allennlp.org/elmo), tutorial, github
7. [Elmo on google hub and code](https://tfhub.dev/google/elmo/2)
8. [How to use elmo embeddings, advice for word and sentence](https://github.com/tensorflow/hub/issues/149)
9. Using elmo as a lambda embedding layer
10. [Elmbo tutorial notebook](https://github.com/sambit9238/Deep-Learning/blob/master/elmo_embedding_tfhub.ipynb)
11. Elmo code on git
12. Elmo on keras using lambda
13. [Elmo pretrained models for many languages](https://github.com/HIT-SCIR/ELMoForManyLangs), for [russian](http://docs.deeppavlov.ai/en/master/intro/pretrained_vectors.html) too, [mean elmo](https://stackoverflow.com/questions/53061423/how-to-represent-elmo-embeddings-as-a-1d-array/53088523)
14. Ari’s intro on word embeddings part 2, has elmo and some bert
15. [Mean elmo](https://www.analyticsvidhya.com/blog/2019/03/learn-to-use-elmo-to-extract-features-from-text/?utm_source=facebook.com&utm_medium=social), batches, with code and linear regression i
16. Elmo projected using TSNE - grouping are not semantically similar

### ULMFIT
This section covers ULMFiT papers, Fast.ai tutorials, and text-classification walkthroughs.

The same notes are in [Language modeling](language-embeddings.md#language-modeling).

1. [Tutorial and code by vidhya](https://www.analyticsvidhya.com/blog/2018/11/tutorial-text-classification-ulmfit-fastai-library/), [medium](https://medium.com/analytics-vidhya/tutorial-on-text-classification-nlp-using-ulmfit-and-fastai-library-in-python-2f15a2aac065)
2. [Paper](https://arxiv.org/abs/1801.06146)
3. [Ruder on transfer learning](http://ruder.io/nlp-imagenet/)
4. Medium on how - unclear
5. Fast NLP on how
6. [Paper: ulmfit](https://arxiv.org/abs/1801.06146)
7. [Fast.ai on ulmfit](http://nlp.fast.ai/category/classification.html), [this too](https://github.com/fastai/fastai/blob/c502f12fa0c766dda6c2740b2d3823e2deb363f9/nbs/examples/ulmfit.ipynb)
8. [Vidhya on ulmfit using fastai](https://www.analyticsvidhya.com/blog/2018/11/tutorial-text-classification-ulmfit-fastai-library/?utm_source=facebook.com)
9. Medium on ulmfit
10. [Building blocks of ulm fit](https://medium.com/mlreview/understanding-building-blocks-of-ulmfit-818d3775325b)
11. [Applying ulmfit on entity level sentiment analysis using business news artcles](https://github.com/jannenev/ulmfit-language-model)
12. Understanding language modelling using Ulmfit, fine tuning etc
13. [Vidhaya on ulmfit + colab](https://www.analyticsvidhya.com/blog/2018/11/tutorial-text-classification-ulmfit-fastai-library/) “The one cycle policy provides some form of regularisation”, if you wish to know more about one cycle policy, then feel free to refer to this excellent paper by Leslie Smith – “[A disciplined approach to neural network hyper-parameters: Part 1 — learning rate, batch size, momentum, and weight decay](https://arxiv.org/abs/1803.09820)”.

### BERT
This section gathers BERT papers, fine-tuning guides, compression notes, and analysis tools.

The same notes are in [GLUE](../evals/benchmarking.md#glue), [Language modeling](language-embeddings.md#language-modeling), [Named Entity Recognition (NER)](named-entity-recognition-ner.md), [PRUNING / KNOWLEDGE DISTILLATION / LOTTERY TICKET](../deep-learning/deep-network-optimization.md#pruning--knowledge-distillation--lottery-ticket), [Tokenization](tokenization.md), and [TOP2VEC](topics-modeling.md#top2vec).

1. [The BERT PAPER](https://arxiv.org/pdf/1810.04805.pdf)
   1. [Prerequisite about transformers and attention - this is not enough](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
   2. [Embeddings using bert in python](https://hackerstreak.com/word-embeddings-using-bert-in-python/) - using bert as a service to encode 1024 vectors and do cosine similarity
   3. Identifying the right meaning with bert - the idea is to classify the word duck into one of three meanings using bert embeddings, which promise contextualized embeddings. I.e., to duck, the Duck, etc<figure><img src="../.gitbook/assets/gimg-3654cc85635e.png" alt=""><figcaption><p>Identifying the right meaning of words using BERT.</p><p>Credit: <a href="https://lh5.googleusercontent.com/WnEaYRk3za14yoiPr0dxf7f3D4iPdmNoLPnQaFi9V94oBd38mTsLvAbqLHeNYsobJmy415hWgGSoMBPrcoIXIJkwK2xHF9QHWO5vKQGI2BEA_7aQQAppHQeYePFUewj4EQRjlpaF">copied from the original hosted image</a>.</p></figcaption></figure>
   4. [Google neural machine translation (attention) - too long](https://arxiv.org/pdf/1609.08144.pdf)
2. What is bert
3. (amazing) Deconstructing bert
   1. I found some fairly distinctive and surprisingly intuitive attention patterns. Below I identify six key patterns and for each one I show visualizations for a particular layer / head that exhibited the pattern.
   2. part 1 - attention to the next/previous/ identical/related (same and other sentences), other words predictive of a word, delimeters tokens
   3. (good) Deconstructing bert part 2 - looking at the visualization and attention heads, focusing on Delimiter attention, bag of words attention, next word attention - patterns.
4. [Bert demystified](https://medium.com/@_init_/why-bert-has-3-embedding-layers-and-their-implementation-details-9c261108e28a) (read this first!)
5. Read this after, the most coherent explanation on bert, 15% masked word prediction and next sentence prediction. Roberta, xlm bert, albert, distilibert.
6. A [thorough tutorial on bert](http://mccormickml.com/2019/07/22/BERT-fine-tuning/), fine tuning using hugging face transformers package. [Code](https://colab.research.google.com/drive/1Y4o3jh3ZH70tl6mCd76vz_IxX23biCPP)

Youtube [ep1](https://www.youtube.com/watch?v=FKlPCK1uFrc), [2](https://www.youtube.com/watch?v=zJW57aCBCTk), [3](https://www.youtube.com/watch?v=x66kkDnbzi4), [3b](https://www.youtube.com/watch?v=Hnvb9b7a_Ps),

1. [How to train bert](https://medium.com/@vineet.mundhra/loading-bert-with-tensorflow-hub-7f5a1c722565) from scratch using TF, with \[CLS] \[SEP] etc
2. Extending a vocabulary for bert, another kind of transfer learning.
3. [Bert tutorial](http://mccormickml.com/2019/07/22/BERT-fine-tuning/), on fine tuning, some talk on from scratch and probably not discussed about using embeddings as input
4. [Bert for summarization thread](https://github.com/google-research/bert/issues/352)
5. [Bert on logs](https://medium.com/rapids-ai/cybert-28b35a4c81c4), feature names as labels, finetune bert, predict.
6. Bert scikit wrapper for pipelines
7. What is bert not good at, also refer to the cited paper (is/is not)
8. [Jay Alamar on Bert](http://jalammar.github.io/illustrated-bert/)
9. [Jay Alamar on using distilliBert](http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/)
10. sparse bert, [paper](https://arxiv.org/abs/2005.07683) - When combined with distillation, the approach achieves minimal accuracy loss with down to only 3% of the model parameters.
11. Bert with keras, [blog post](https://www.ctolib.com/Separius-BERT-keras.html), [colaboratory](https://colab.research.google.com/gist/HighCWu/3a02dc497593f8bbe4785e63be99c0c3/bert-keras-tutorial.ipynb)
12. [Bert with t-hub](https://github.com/google-research/bert/blob/master/run_classifier_with_tfhub.py)
13. [Bert on medium with code](https://medium.com/huggingface/multi-label-text-classification-using-bert-the-mighty-transformer-69714fa3fb3d)
14. [Bert on git](https://github.com/SkullFang/BERT_NLP_Classification)
15. Finetuning - [Better sentiment analysis with bert](https://medium.com/southpigalle/how-to-perform-better-sentiment-analysis-with-bert-ba127081eda), claims 94% on IMDB. official code [here](https://github.com/google-research/bert/blob/master/predicting_movie_reviews_with_bert_on_tf_hub.ipynb) “ it creates a single new layer that will be trained to adapt BERT to our sentiment task (i.e. classifying whether a movie review is positive or negative). This strategy of using a mostly trained model is called fine-tuning.”
16. [Explain bert](https://web.archive.org/web/2020/http://exbert.net/) - bert visualization tool.
17. sentenceBERT [paper](https://arxiv.org/pdf/1908.10084.pdf)
18. Bert question answering on covid19
19. [Codebert](https://arxiv.org/pdf/2002.08155.pdf)
20. Bert multilabel classification
21. [Tabert](https://ai.facebook.com/blog/tabert-a-new-model-for-understanding-queries-over-tabular-data/) - [TaBERT](https://ai.facebook.com/research/publications/tabert-pretraining-for-joint-understanding-of-textual-and-tabular-data/) is the first model that has been pretrained to learn representations for both natural language sentences and tabular data.
22. [All the ways that you can compress BERT](http://mitchgordon.me/machine/learning/2019/11/18/all-the-ways-to-compress-BERT.html)

Pruning - Removes unnecessary parts of the network after training. This includes weight magnitude pruning, attention head pruning, layers, and others. Some methods also impose regularization during training to increase prunability (layer dropout).

Weight Factorization - Approximates parameter matrices by factorizing them into a multiplication of two smaller matrices. This imposes a low-rank constraint on the matrix. Weight factorization can be applied to both token embeddings (which saves a lot of memory on disk) or parameters in feed-forward / self-attention layers (for some speed improvements).

Knowledge Distillation - Aka “Student Teacher.” Trains a much smaller Transformer from scratch on the pre-training / downstream-data. Normally this would fail, but utilizing soft labels from a fully-sized model improves optimization for unknown reasons. Some methods also distill BERT into different architectures (LSTMS, etc.) which have faster inference times. Others dig deeper into the teacher, looking not just at the output but at weight matrices and hidden activations.

Weight Sharing - Some weights in the model share the same value as other parameters in the model. For example, ALBERT uses the same weight matrices for every single layer of self-attention in BERT.

Quantization - Truncates floating point numbers to only use a few bits (which causes round-off error). The quantization values can also be learned either during or after training.

Pre-train vs. Downstream - Some methods only compress BERT w.r.t. certain downstream tasks. Others compress BERT in a way that is task-agnostic.

1. Bert and nlp in 2019
2. [HeBert - bert for hebrwe sentiment and emotions](https://github.com/avichaychriqui/HeBERT)
3. [Kdbuggets on visualizing bert](https://www.kdnuggets.com/2019/03/deconstructing-bert-part-2-visualizing-inner-workings-attention.html)
4. [What does bert look at, analysis of attention](https://www-nlp.stanford.edu/pubs/clark2019what.pdf) - We further show that certain attention heads correspond well to linguistic notions of syntax and coreference. For example, we find heads that attend to the direct objects of verbs, determiners of nouns, objects of prepositions, and coreferent mentions with remarkably high accuracy. Lastly, we propose an attention-based probing classifier and use it to further demonstrate that substantial syntactic information is captured in BERT’s attention
5. [Bertviz](https://github.com/jessevig/bertviz) BertViz is a tool for visualizing attention in the Transformer model, supporting all models from the [transformers](https://github.com/huggingface/transformers) library (BERT, GPT-2, XLNet, RoBERTa, XLM, CTRL, etc.). It extends the [Tensor2Tensor visualization tool](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/visualization) by [Llion Jones](https://medium.com/@llionj) and the [transformers](https://github.com/huggingface/transformers) library from [HuggingFace](https://github.com/huggingface).
6. PMI-masking [paper](https://openreview.net/forum?id=3Aoft6NWFej), post - Joint masking of correlated tokens significantly speeds up and improves BERT's pretraining
7. (really good/) Examining bert raw embeddings - TL;DR BERT’s raw word embeddings capture useful and separable information (distinct histogram tails) about a word in terms of other words in BERT’s vocabulary. This information can be harvested from both raw embeddings and their transformed versions after they pass through BERT with a Masked language model (MLM) head

<figure><img src="../.gitbook/assets/gimg-e6ccb16efa98.png" alt=""><figcaption><p>Examining BERT raw embeddings.</p><p>Credit: <a href="https://lh6.googleusercontent.com/nIgQQPipHF7dhRxdOw79cMhogIBvcdNjftMtQckXAKuZWkZgpgXiaBgyijRI1IB5x7oTLSRF0yL9XKv64hsSAhdnsPiRWMiIR8vQyZOpzpPdD-Qe9YTzvMgRVcEdOMQf9bCTdjVb">copied from the original hosted image</a>.</p></figcaption></figure>

<figure><img src="../.gitbook/assets/gimg-8b2bf945bb70.png" alt=""><figcaption><p>Examining BERT raw embeddings.</p><p>Credit: <a href="https://lh6.googleusercontent.com/gma8aGDKP8chI7HuhKdl2Gu6tFUT_iHghfYZ8YyvfQta3-6DFw5YSZK2v-at3XneSjo0QnVtXfcs9wNL8CdCY4D8aZXxNlduUjwXxqjao6WoiAN17R5qH46Cx1SDGjU-yu5O9W13">copied from the original hosted image</a>.</p></figcaption></figure>

<figure><img src="../.gitbook/assets/gimg-e6cb3ce8172f.png" alt=""><figcaption><p>Examining BERT raw embeddings.</p><p>Credit: <a href="https://lh5.googleusercontent.com/4_FW_BymDsKMdFzKVNZ2Dmm_3pI6UrNlPWK7YsBgIznbAi551G0QkCUrRVK0sW6_sMsZ_WFJ0GwHdlu0X3YNjZ0k947iQ27PVG6ZSp7jOWjhRNr5d7FbMe1lauiresaYn9u1nXIY">copied from the original hosted image</a>.</p></figcaption></figure>

<figure><img src="../.gitbook/assets/gimg-65e5d533ccde.png" alt=""><figcaption><p>Examining BERT raw embeddings.</p><p>Credit: <a href="https://lh5.googleusercontent.com/Hp7oLFDNtANqlV5RQzKWF-TsuURUlxQZS_sjQFXD48H3PnTtwthIGfN1zxKU14uf8y4746oXRzc4KvfyW4zBcKOdwL92LKYb9cwfDsD14-y_Lv6pmBdnwrpDyqzP0LjLEpEqWk5b">copied from the original hosted image</a>.</p></figcaption></figure>

### GPT2
This section notes GPT-2 language modeling and PyTorch exploration posts.

The same notes are in [GPT](../generative-ai/large-language-models-llms.md) and [Language modeling](language-embeddings.md#language-modeling).

1. [the GPT-2](https://medium.com/dair-ai/experimenting-with-openais-improved-language-model-abf73bc123b9) small algorithm was trained on the task of language modeling — which tests a program’s ability to predict the next word in a given sentence — by ingesting huge numbers of articles, blogs, and websites. By using just this data it achieved state-of-the-art scores on a number of unseen language tests, an achievement known as zero-shot learning. It can also perform other writing-related tasks, such as translating text from one language to another, summarizing long articles, and answering trivia questions.
2. [Medium code](https://medium.com/dair-ai/explore-pretrained-language-models-with-pytorch-1b1e06b7510c) for GPT=2 - big algo

### GPT3
This section points at GPT-3 zero-shot commentary and large-model training infrastructure.

The same notes are in [GPT](../generative-ai/large-language-models-llms.md), [GPT3 is ZERO, ONE, FEW](../problem-framing/n-shot-learning.md#gpt3-is-zero-one-few), and [Language modeling](language-embeddings.md#language-modeling).

1. [GPT3](https://medium.com/swlh/all-hail-gpt-3-389c7f1fcb3b) on medium - language models can be used to produce good results on zero-shot, one-shot, or few-shot learning.
2. [Fit More and Train Faster With ZeRO via DeepSpeed and FairScale](https://huggingface.co/blog/zero-deepspeed-fairscale)

### XLNET
This section covers XLNet explanations and code.

1. [Xlnet is transformer and bert combined](https://medium.com/logits/xlnet-sota-pre-training-method-that-outperforms-bert-26d4e9978983) - Actually its quite good explaining it
2. [git](https://github.com/zihangdai/xlnet)
3. CLIP
4. (keras) [Implementation of a dual encoder](https://keras.io/examples/nlp/nl_image_search/) model for retrieving images that match natural language queries. - The example demonstrates how to build a dual encoder (also known as two-tower) neural network model to search for images using natural language. The model is inspired by the [CLIP](https://openai.com/blog/clip/) approach, introduced by Alec Radford et al. The idea is to train a vision encoder and a text encoder jointly to project the representation of images and their captions into the same embedding space, such that the caption embeddings are located near the embeddings of the images they describe.
5. Adversarial methodologies
6. What is label [flipping and smoothing](https://datascience.stackexchange.com/questions/55359/how-label-smoothing-and-label-flipping-increases-the-performance-of-a-machine-le/56662) and usage for making a model more robust against adversarial methodologies - 0

Label flipping is a training technique where one selectively manipulates the labels in order to make the model more robust against label noise and associated attacks - the specifics depend a lot on the nature of the noise. Label flipping bears no benefit only under the assumption that all labels are (and will always be) correct and that no adversaries exist. In cases where noise tolerance is desirable, training with label flipping is beneficial.

Label smoothing is a regularization technique (and then some) aimed at improving model performance. Its effect takes place irrespective of label correctness.

1. [Paper: when does label smoothing helps?](https://arxiv.org/abs/1906.02629) Smoothing the labels in this way prevents the network from becoming overconfident and label smoothing has been used in many state-of-the-art models, including image classification, language translation and speech recognition...Here we show empirically that in addition to improving generalization, label smoothing improves model calibration which can significantly improve beam-search. However, we also observe that if a teacher network is trained with label smoothing, knowledge distillation into a student network is much less effective.
2. [Label smoothing, python code, multi class examples](https://rickwierenga.com/blog/fast.ai/FastAI2019-12.html)

<figure><img src="../.gitbook/assets/gimg-bbb9e1ef0d66.png" alt=""><figcaption><p>Label smoothing.</p><p>Credit: <a href="https://lh4.googleusercontent.com/pScpTAmy9S8uTobVoSLAjSlASouxyA2iBDNxH8VEjBg4indhs57dHWYXoqEZSTfp6Hhwh9i0LboD65o1LXfxv61dMJwnz1dDbm1lhcvVYtvVbW8H6Rhia-lk0bLfDomS3z6kKNlZ">copied from the original hosted image</a>.</p></figcaption></figure>

1. [Label sanitazation against label flipping poisoning attacks](https://arxiv.org/abs/1803.00992) - In this paper we propose an efficient algorithm to perform optimal label flipping poisoning attacks and a mechanism to detect and relabel suspicious data points, mitigating the effect of such poisoning attacks.
2. [Adversarial label flips attacks on svm](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.398.7446&rep=rep1&type=pdf) - To develop a robust classification algorithm in the adversarial setting, it is important to understand the adversary’s strategy. We address the problem of label flips attack where an adversary contaminates the training set through flipping labels. By analyzing the objective of the adversary, we formulate an optimization framework for finding the label flips that maximize the classification error. An algorithm for attacking support vector machines is derived. Experiments demonstrate that the accuracy of classifiers is significantly degraded under the attack.
3. GAN
4. [Great advice for training gans](https://medium.com/@utk.is.here/keep-calm-and-train-a-gan-pitfalls-and-tips-on-training-generative-adversarial-networks-edd529764aa9), such as label flipping batch norm, etc read!
5. [Intro to Gans](https://medium.com/sigmoid/a-brief-introduction-to-gans-and-how-to-code-them-2620ee465c30)
6. [A fantastic series about gans, the following two what are gans and applications are there](https://medium.com/@jonathan_hui/gan-gan-series-2d279f906e7b)
   1. [What are a GANs?](https://medium.com/@jonathan_hui/gan-whats-generative-adversarial-networks-and-its-application-f39ed278ef09), and cool [applications](https://medium.com/@jonathan_hui/gan-some-cool-applications-of-gans-4c9ecca35900)
   2. [Comprehensive overview](https://medium.com/@jonathan_hui/gan-a-comprehensive-review-into-the-gangsters-of-gans-part-1-95ff52455672)
   3. [Cycle gan](https://medium.com/@jonathan_hui/gan-cyclegan-6a50e7600d7) - transferring styles
   4. [Super gan resolution](https://medium.com/@jonathan_hui/gan-super-resolution-gan-srgan-b471da7270ec) - super res images
   5. [Why gan so hard to train](https://medium.com/@jonathan_hui/gan-why-it-is-so-hard-to-train-generative-advisory-networks-819a86b3750b) - good for critique
   6. And how to improve gans performance
   7. [Dcgan good as a starting point in new projects](https://medium.com/@jonathan_hui/gan-dcgan-deep-convolutional-generative-adversarial-networks-df855c438f)
   8. [Labels to improve gans, cgan, infogan](https://medium.com/@jonathan_hui/gan-cgan-infogan-using-labels-to-improve-gan-8ba4de5f9c3d)
   9. [Stacked - labels, gan adversarial loss, entropy loss, conditional loss](https://medium.com/@jonathan_hui/gan-stacked-generative-adversarial-networks-sgan-d9449ac63db8) - divide and conquer
   10. [Progressive gans](https://medium.com/@jonathan_hui/gan-progressive-growing-of-gans-f9e4f91edf33) - mini batch discrimination
   11. [Using attention to improve gan](https://medium.com/@jonathan_hui/gan-self-attention-generative-adversarial-networks-sagan-923fccde790c)
   12. [Least square gan - lsgan](https://medium.com/@jonathan_hui/gan-lsgan-how-to-be-a-good-helper-62ff52dd3578)
   13. Unread:
       1. [Wasserstein gan, wgan gp](https://medium.com/@jonathan_hui/gan-wasserstein-gan-wgan-gp-6a1a2aa1b490)
       2. [Faster training for gans, lower training count rsgan ragan](https://medium.com/@jonathan_hui/gan-rsgan-ragan-a-new-generation-of-cost-function-84c5374d3c6e)
       3. [Addressing gan stability, ebgan began](https://medium.com/@jonathan_hui/gan-energy-based-gan-ebgan-boundary-equilibrium-gan-began-4662cceb7824)
       4. [What is wrong with gan cost functions](https://medium.com/@jonathan_hui/gan-what-is-wrong-with-the-gan-cost-function-6f594162ce01)
       5. [Using cost functions for gans inspite of the google brain paper](https://medium.com/@jonathan_hui/gan-does-lsgan-wgan-wgan-gp-or-began-matter-e19337773233)
       6. [Proving gan is js-convergence](https://medium.com/@jonathan_hui/proof-gan-optimal-point-658116a236fb)
       7. [Dragan on minimizing local equilibria, how to stabilize gans](https://medium.com/@jonathan_hui/gan-dragan-5ba50eafcdf2), reducing mode collapse
       8. [Unrolled gan for reducing mode collapse](https://medium.com/@jonathan_hui/gan-unrolled-gan-how-to-reduce-mode-collapse-af5f2f7b51cd)
       9. [Measuring gans](https://medium.com/@jonathan_hui/gan-how-to-measure-gan-performance-64b988c47732)
       10. Ways to improve gans performance
       11. [Introduction to gans](https://medium.freecodecamp.org/an-intuitive-introduction-to-generative-adversarial-networks-gans-7a2264a81394) with tf code
       12. [Intro to gans](https://medium.com/datadriveninvestor/deep-learning-generative-adversarial-network-gan-34abb43c0644)
       13. Intro to gan in KERAS
7. “GAN” using xgboost and gmm for density sampling
8. [Reverse engineering](https://ai.facebook.com/blog/reverse-engineering-generative-model-from-a-single-deepfake-image/)

