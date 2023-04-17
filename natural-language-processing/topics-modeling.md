# Topics Modeling

## Misc

1. [**Word cloud**](http://keyonvafa.com/inauguration-wordclouds/) **for topic modellng**
2. [**Topic modeling with sentiment per topic according to the data in the topic**](https://www.slideshare.net/jainayush91/topic-modelling-tutorial-on-usage-and-applications)
3. **(TopSBM) topic block modeling,** [**Topsbm** ](https://topsbm.github.io/)

## NMF (Non Negative Matrix Factorization )

1. **Non-negative Matrix factorization (NMF)**
2. [**Medium Article about LDA and**](https://medium.com/ml2vec/topic-modeling-is-an-unsupervised-learning-approach-to-clustering-documents-to-discover-topics-fdfbf30e27df) **NMF (Non-negative Matrix factorization)+ code**
3. [**Sklearn LDA and NMF for topic modelling**](http://scikit-learn.org/stable/auto\_examples/applications/plot\_topics\_extraction\_with\_nmf\_lda.html#sphx-glr-auto-examples-applications-plot-topics-extraction-with-nmf-lda-py)

## **LSA (TFIDF + SVD)**

1. [**A very good article about LSA (TFIDV X SVD), pLSA, LDA, and LDA2VEC.**](https://medium.com/nanonets/topic-modeling-with-lsa-psla-lda-and-lda2vec-555ff65b0b05) **Including code and explanation about Dirichlet probability.** [**Lda2vec code**](http://nbviewer.jupyter.org/github/cemoody/lda2vec/blob/master/examples/twenty\_newsgroups/lda2vec/lda2vec.ipynb)
2. [**A descriptive comparison for LSA pLSA and LDA**](https://www.reddit.com/r/MachineLearning/comments/10mdtf/lsa\_vs\_plsa\_vs\_lda/)

## **LDA (Latent Dirichlet Allocation)**

* **A** [**great summation**](https://cs.stanford.edu/\~ppasupat/a9online/1140.html) **about topic modeling, Pros and Cons! (LSA, pLSA, LDA)**

1. **(LDA) Latent Dirichlet Allocation**&#x20;
2. **LDA is already taken by the above algorithm!**
3. [**Latent Dirichlet allocation (LDA) -**](https://algorithmia.com/algorithms/nlp/LDA) **This algorithm takes a group of documents (anything that is made of up text), and returns a number of topics (which are made up of a number of words) most relevant to these documents.** &#x20;
4. [**Medium Article about LDA and**](https://medium.com/ml2vec/topic-modeling-is-an-unsupervised-learning-approach-to-clustering-documents-to-discover-topics-fdfbf30e27df) **NMF (Non-negative Matrix factorization)+ code**
5. [**Medium article on LDA - a good one with pseudo algorithm and proof**](https://medium.com/@jonathan\_hui/machine-learning-latent-dirichlet-allocation-lda-1d9d148f13a4)\

6. **In case LDA groups together two topics, we can influence the algorithm in a way that makes those two topics separable -** [**this is called Semi Supervised Guided LDA**](https://medium.freecodecamp.org/how-we-changed-unsupervised-lda-to-semi-supervised-guidedlda-e36a95f3a164)\

7. [**LDA tutorials plus code**](https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/)**, used this to build my own classes - using gensim mallet wrapper, doesn't work on pyLDAviz, so use** [**this**](http://jeriwieringa.com/2018/07/17/pyLDAviz-and-Mallet/#comment-4018495276) **to fix it**&#x20;
8. [**Introduction to LDA topic modelling, really good,**](http://www.vladsandulescu.com/topic-prediction-lda-user-reviews/) [**plus git code**](https://github.com/vladsandulescu/topics)
9. [**Sklearn examples using LDA and NMF**](http://scikit-learn.org/stable/auto\_examples/applications/plot\_topics\_extraction\_with\_nmf\_lda.html#sphx-glr-auto-examples-applications-plot-topics-extraction-with-nmf-lda-py)
10. [**Tutorial on lda/nmf on medium**](https://medium.com/mlreview/topic-modeling-with-scikit-learn-e80d33668730) **- using tfidf matrix as input!**
11. [**Gensim and sklearn LDA variants, comparison**](https://gist.github.com/aronwc/8248457)**,** [**python 3**](https://github.com/EricSchles/sklearn\_gensim\_example/blob/master/example.py)
12. [**Medium article on lda/nmf with code**](https://medium.com/mlreview/topic-modeling-with-scikit-learn-e80d33668730)
13. **One of the best explanation about** [**Tf-idf vs bow for LDA/NMF**](https://stackoverflow.com/questions/44781047/necessary-to-apply-tf-idf-to-new-documents-in-gensim-lda-model) **- tf for lda, tfidf for nmf, but tfidf can be used for top k selection in lda + visualization,** [**important paper**](http://www.cs.columbia.edu/\~blei/papers/BleiLafferty2009.pdf)
14. [**LDA is a probabilistic**](https://stackoverflow.com/questions/40171208/scikit-learn-should-i-fit-model-with-tf-or-tf-idf) **generative model that generates documents by sampling a topic for each word and then a word from the sampled topic. The generated document is represented as a bag of words.**\


    **NMF is in its general definition the search for 2 matrices W and H such that W\*H=V where V is an observed matrix. The only requirement for those matrices is that all their elements must be non negative.**



    **From the above definitions it is clear that in LDA only bag of words frequency counts can be used since a vector of reals makes no sense. Did we create a word 1.2 times? On the other hand we can use any non negative representation for NMF and in the example tf-idf is used.**

    \
    **As far as choosing the number of iterations, for the NMF in scikit learn I don't know the stopping criterion although I believe it is the relative improvement of the loss function being smaller than a threshold so you 'll have to experiment. For LDA I suggest checking manually the improvement of the log likelihood in a held out validation set and stopping when it falls under a threshold.**\
    \
    **The rest of the parameters depend heavily on the data so I suggest, as suggested by @rpd, that you do a parameter search.**\
    \
    **So to sum up, LDA can only generate frequencies and NMF can generate any non negative matrix.**
15. [**How to measure the variance for LDA and NMF, against PCA.**](https://stackoverflow.com/questions/48148689/how-to-compare-predictive-power-of-pca-and-nmf) **1. Variance score the transformation and inverse transformation of data, test for 1,2,3,4 PCs/LDs/NMs.**
16. [**Matching lda mallet performance with gensim and sklearn lda via hyper parameters**](https://groups.google.com/forum/#!topic/gensim/bBHkGogNrfg)
17. [**What is LDA?**](https://www.quora.com/Is-LDA-latent-dirichlet-allocation-unsupervised-or-supervised-learning)
    1. **It is unsupervised natively; it uses joint probability method to find topics(user has to pass # of topics to LDA api). If “Doc X word” is size of input data to LDA, it transforms it to 2 matrices:**
    2. **Doc X topic**
    3. **Word X topic**
    4. **further if you want, you can feed “Doc X topic” matrix to supervised algorithm if labels were given.**
18. **Medium on** [**LDA**](https://medium.com/ml2vec/topic-modeling-is-an-unsupervised-learning-approach-to-clustering-documents-to-discover-topics-fdfbf30e27df)**, explains the random probabilistic nature of LDA**![](https://lh6.googleusercontent.com/-16nr83feu9UQzaIoi4CIMYwSHRhH99p49scg\_Mnk9PH7EmMh-Q6410FLxPtwZCapOrKkq3J9MK7njHPD21o1TYxZYZopSHAoWCKFuwCMU8Rcy0kLIacqWcPqtETr8ZuTaxN6BLn)
19. **Machinelearningplus on** [**LDA in sklearn**](https://www.machinelearningplus.com/nlp/topic-modeling-python-sklearn-examples/) **- a great read, dont forget to read the** [**mallet**](https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/) **article.**
20. **Medium on** [**LSA pLSA, LDA LDA2vec**](https://medium.com/nanonets/topic-modeling-with-lsa-psla-lda-and-lda2vec-555ff65b0b05)**, high level theoretical - not clear**
21. [**Medium on LSI vs LDA vs HDP, HDP wins..**](https://medium.com/square-corner-blog/topic-modeling-optimizing-for-human-interpretability-48a81f6ce0ed)
22. **Medium on** [**LDA**](https://medium.com/@samsachedina/effective-data-science-latent-dirichlet-allocation-a109742f7d1c)**, some historical reference and general high level how to use exapmles.**
23. [**Incredibly useful response**](https://www.quora.com/What-are-good-ways-of-evaluating-the-topics-generated-by-running-LDA-on-a-corpus) **on LDA grid search params and about LDA expectations. Must read.**
24. [**Lda vs pLSA**](https://stats.stackexchange.com/questions/155860/latent-dirichlet-allocation-vs-plsa)**, talks about the sampling from a distribution of distributions in LDA**
25. [**BLog post on topic modelling**](http://mcburton.net/blog/joy-of-tm/) **- has some text about overfitting - undiscussed in many places.**
26. [**Perplexity vs coherence on held out unseen dat**](https://stats.stackexchange.com/questions/182010/when-is-it-ok-to-not-use-a-held-out-set-for-topic-model-evaluation)**a, not okay and okay, respectively. Due to how we measure the metrics, ie., read the formulas.** [**Also this**](https://transacl.org/ojs/index.php/tacl/article/view/582/158) **and** [**this**](https://stackoverflow.com/questions/11162402/lda-topic-modeling-training-and-testing)
27. **LDA as** [**dimentionality reduction** ](https://stackoverflow.com/questions/46504688/lda-as-the-dimension-reduction-before-or-after-partitioning)
28. [**LDA on alpha and beta to control density of topics**](https://stats.stackexchange.com/questions/364494/lda-and-test-data-perplexity)
29. **Jupyter notebook on** [**hacknews LDA topic modelling**](http://nbviewer.jupyter.org/github/bmabey/hacker\_news\_topic\_modelling/blob/master/HN%20Topic%20Model%20Talk.ipynb#topic=55\&lambda=1\&term=) **- missing code?**
30. [**Jupyter notebook**](http://nbviewer.jupyter.org/github/dolaameng/tutorials/blob/master/topic-finding-for-short-texts/topics\_for\_short\_texts.ipynb) **for kmeans, lda, svd,nmf comparison - advice is to keep nmf or other as a baseline to measure against LDA.**
31. [**Gensim on LDA**](https://rare-technologies.com/what-is-topic-coherence/) **with** [**code** ](https://nbviewer.jupyter.org/github/dsquareindia/gensim/blob/280375fe14adea67ce6384ba7eabf362b05e6029/docs/notebooks/topic\_coherence\_tutorial.ipynb)
32. [**Medium on lda with sklearn**](https://medium.com/mlreview/topic-modeling-with-scikit-learn-e80d33668730)
33. **Selecting the number of topics in LDA,** [**blog 1**](https://cran.r-project.org/web/packages/ldatuning/vignettes/topics.html)**,** [**blog2**](http://www.rpubs.com/MNidhi/NumberoftopicsLDA)**,** [**using preplexity**](https://stackoverflow.com/questions/21355156/topic-models-cross-validation-with-loglikelihood-or-perplexity)**,** [**prep and aic bic**](https://stats.stackexchange.com/questions/322809/inferring-the-number-of-topics-for-gensims-lda-perplexity-cm-aic-and-bic)**,** [**coherence**](https://stackoverflow.com/questions/17421887/how-to-determine-the-number-of-topics-for-lda)**,** [**coherence2**](https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/#17howtofindtheoptimalnumberoftopicsforlda)**,** [**coherence 3 with tutorial**](https://datascienceplus.com/evaluation-of-topic-modeling-topic-coherence/)**, un**[**clear**](https://community.rapidminer.com/discussion/51283/what-is-the-best-number-of-topics-on-lda)**,** [**unclear with analysis of stopword % inclusion**](https://markhneedham.com/blog/2015/03/24/topic-modelling-working-out-the-optimal-number-of-topics/)**,** [**unread**](https://www.quora.com/What-are-the-best-ways-of-selecting-number-of-topics-in-LDA)**,** [**paper: heuristic approach**](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4597325/)**,** [**elbow method**](https://www.knime.com/blog/topic-extraction-optimizing-the-number-of-topics-with-the-elbow-method)**,** [**using cv**](http://freerangestats.info/blog/2017/01/05/topic-model-cv)**,** [**Paper: new stability metric**](https://github.com/derekgreene/topic-stability) **+ gh code,**&#x20;
34. [**Selecting the top K words in LDA**](https://stats.stackexchange.com/questions/199263/choosing-words-in-a-topic-which-cut-off-for-lda-topics)
35. [**Presentation: best practices for LDA**](http://www.phusewiki.org/wiki/images/c/c9/Weizhong\_Presentation\_CDER\_Nov\_9th.pdf)
36. [**Medium on guidedLDA**](https://medium.freecodecamp.org/how-we-changed-unsupervised-lda-to-semi-supervised-guidedlda-e36a95f3a164) **- switching from LDA to a variation of it that is guided by the researcher / data**&#x20;
37. **Medium on lda -** [**another introductory**](https://towardsdatascience.com/thats-mental-using-lda-topic-modeling-to-investigate-the-discourse-on-mental-health-over-time-11da252259c3)**,** [**la times**](https://medium.com/swiftworld/topic-modeling-of-new-york-times-articles-11688837d32f)
38. [**Topic modelling through time**](https://tedunderwood.com/category/methodology/topic-modeling/)
39. [**Mallet vs nltk**](https://stackoverflow.com/questions/7476180/topic-modelling-in-mallet-vs-nltk)**,** [**params**](https://github.com/RaRe-Technologies/gensim/issues/193)**,** [**params**](https://groups.google.com/forum/#!topic/gensim/tOoc1Q0Ump0)
40. [**Paper: improving feature models**](http://aclweb.org/anthology/Q15-1022)
41. [**Lda vs w2v (doesn't make sense to compare**](https://stats.stackexchange.com/questions/145485/lda-vs-word2vec/145488)**,** [**again here**](https://stats.stackexchange.com/questions/145485/lda-vs-word2vec)
42. [**Adding lda features to w2v for classification**](https://stackoverflow.com/questions/48140319/add-lda-topic-modelling-features-to-word2vec-sentiment-classification)
43. [**Spacy and gensim on 20 news groups**](https://www.shanelynn.ie/word-embeddings-in-python-with-spacy-and-gensim/)
44. **The best topic modelling explanation including** [**Usages**](https://nlpforhackers.io/topic-modeling/)**, insights,  a great read, with code  - shows how to find similar docs by topic in gensim, and shows how to transform unseen documents and do similarity using sklearn:**&#x20;
    1. **Text classification – Topic modeling can improve classification by grouping similar words together in topics rather than using each word as a feature**
    2. **Recommender Systems – Using a similarity measure we can build recommender systems. If our system would recommend articles for readers, it will recommend articles with a topic structure similar to the articles the user has already read.**
    3. **Uncovering Themes in Texts – Useful for detecting trends in online publications for example**
    4. **A Form of Tagging - If document classification is assigning a single category to a text, topic modeling is assigning multiple tags to a text. A human expert can label the resulting topics with human-readable labels and use different heuristics to convert the weighted topics to a set of tags.**
    5. [**Topic Modelling for Feature Selection**](https://www.analyticsvidhya.com/blog/2016/08/beginners-guide-to-topic-modeling-in-python/) **- Sometimes LDA can also be used as feature selection technique. Take an example of text classification problem where the training data contain category wise documents. If LDA is running on sets of category wise documents. Followed by removing common topic terms across the results of different categories will give the best features for a category.**
45. [**Another great article about LDA**](https://www.analyticsvidhya.com/blog/2016/08/beginners-guide-to-topic-modeling-in-python/)**, including algorithm, parameters!! And Parameters of LDA**
    1. **Alpha and Beta Hyperparameters – alpha represents document-topic density and Beta represents topic-word density. Higher the value of alpha, documents are composed of more topics and lower the value of alpha, documents contain fewer topics. On the other hand, higher the beta, topics are composed of a large number of words in the corpus, and with the lower value of beta, they are composed of few words.**
    2. **Number of Topics – Number of topics to be extracted from the corpus. Researchers have developed approaches to obtain an optimal number of topics by using Kullback Leibler Divergence Score. I will not discuss this in detail, as it is too mathematical. For understanding, one can refer to this\[1] original paper on the use of KL divergence.**
    3. **Number of Topic Terms – Number of terms composed in a single topic. It is generally decided according to the requirement. If the problem statement talks about extracting themes or concepts, it is recommended to choose a higher number, if problem statement talks about extracting features or terms, a low number is recommended.**
    4. **Number of Iterations / passes – Maximum number of iterations allowed to LDA algorithm for convergence.**
46. **Ways to improve LDA:**
    1. **Reduce dimentionality of document-term matrix**
    2. **Frequency filter**
    3. **POS filter**
    4. **Batch wise LDA**
47. [**History of LDA**](http://qpleple.com/bib/#Newman10a) **- by the frech guy**
48. [**Multilingual - alpha is divided by topic count, reaffirms 7**](http://mallet.cs.umass.edu/topics-polylingual.php)
49. [**Topic modelling with lda and nmf on medium**](https://medium.com/ml2vec/topic-modeling-is-an-unsupervised-learning-approach-to-clustering-documents-to-discover-topics-fdfbf30e27df) **- has a very good simple example with probabilities**
50. **Code:** [**great for top docs, terms, topics etc.**](http://nbviewer.jupyter.org/github/bmabey/hacker\_news\_topic\_modelling/blob/master/HN%20Topic%20Model%20Talk.ipynb#topic=55\&lambda=1\&term=)
51. **Great article:** [**Many ways of evaluating topics by running LDA**](https://www.quora.com/What-are-good-ways-of-evaluating-the-topics-generated-by-running-LDA-on-a-corpus)
52. [**Difference between lda in gensim and sklearn a post on rare**](https://github.com/RaRe-Technologies/gensim/issues/457)
53. [**The best code article on LDA/MALLET**](https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/)**, and using** [**sklearn**](https://www.machinelearningplus.com/nlp/topic-modeling-python-sklearn-examples/) **(using clustering for getting group of sentences in each topic)**
54. [**LDA in gensim, a tutorial by gensim**](https://nbviewer.jupyter.org/github/rare-technologies/gensim/blob/develop/docs/notebooks/atmodel\_tutorial.ipynb)
55. &#x20;[**Lda on medium**](https://towardsdatascience.com/topic-modelling-in-python-with-nltk-and-gensim-4ef03213cd21)&#x20;
56. &#x20;[**What are the pros and cons of LDA and NMF in topic modeling? Under what situations should we choose LDA or NMF? Is there comparison of two techniques in topic modeling?**](https://www.quora.com/What-are-the-pros-and-cons-of-LDA-and-NMF-in-topic-modeling-Under-what-situations-should-we-choose-LDA-or-NMF-Is-there-comparison-of-two-techniques-in-topic-modeling)
57. [**What is the difference between NMF and LDA? Why are the priors of LDA sparse-induced?**](https://www.quora.com/What-is-the-difference-between-NMF-and-LDA-Why-are-the-priors-of-LDA-sparse-induced)
58. [**Exploring Topic Coherence over many models and many topics**](http://aclweb.org/anthology/D/D12/D12-1087.pdf) **lda nmf svd, using umass and uci coherence measures**
59. **\*\*\*** [**Practical topic findings for short sentence text**](http://nbviewer.jupyter.org/github/dolaameng/tutorials/blob/master/topic-finding-for-short-texts/topics\_for\_short\_texts.ipynb) **code**
60. [**What's the difference between SVD/NMF and LDA as topic model algorithms essentially? Deterministic vs prob based**](https://www.quora.com/Whats-the-difference-between-SVD-NMF-and-LDA-as-topic-model-algorithms-essentially)
61. [**What is the difference between NMF and LDA? Why are the priors of LDA sparse-induced?**](https://www.quora.com/What-is-the-difference-between-NMF-and-LDA-Why-are-the-priors-of-LDA-sparse-induced)
62. [**What are the relationships among NMF, tensor factorization, deep learning, topic modeling, etc.?**](https://www.quora.com/What-are-the-relationships-among-NMF-tensor-factorization-deep-learning-topic-modeling-etc)
63. [**Code: lda nmf**](https://www.kaggle.com/rchawla8/topic-modeling-with-lda-and-nmf-algorithms)
64. [**Unread a comparison of lda and nmf**](https://wiki.ubc.ca/Course:CPSC522/A\_Comparison\_of\_LDA\_and\_NMF\_for\_Topic\_Modeling\_on\_Literary\_Themes)
65. [**Presentation: lda sparse coding matrix factorization**](https://www.cs.cmu.edu/\~epxing/Class/10708-15/slides/LDA\_SC.pdf)
66. [**An experimental comparison between NMF and LDA for active cross-situational object-word learning**](https://ieeexplore.ieee.org/abstract/document/7846822)
67. [**Topic coherence in gensom with jupyter code**](https://markroxor.github.io/gensim/static/notebooks/topic\_coherence\_tutorial.html)
68. [**Topic modelling dynamic presentation**](http://chdoig.github.io/pygotham-topic-modeling/#/)
69. **Paper:** [**Topic modelling and event identification from twitter data**](https://arxiv.org/ftp/arxiv/papers/1608/1608.02519.pdf)**, says LDA vs NMI (NMF?) and using coherence to analyze**
70. [**Just another medium article about ™**](https://medium.com/square-corner-blog/topic-modeling-optimizing-for-human-interpretability-48a81f6ce0ed)
71. [**What is Wrong with Topic Modeling? (and How to Fix it Using Search-based SE)**](https://www.researchgate.net/publication/307303102\_What\_is\_Wrong\_with\_Topic\_Modeling\_and\_How\_to\_Fix\_it\_Using\_Search-based\_SE) **LDADE's tunings dramatically reduces topic instability.**&#x20;
72. [**Talk about topic modelling**](https://tedunderwood.com/category/methodology/topic-modeling/)
73. [**Intro to topic modelling**](http://blog.echen.me/2011/08/22/introduction-to-latent-dirichlet-allocation/)
74. [**Detecting topics in twitter**](https://github.com/heerme/twitter-topics) **github code**
75. [**Another topic model tutorial**](https://github.com/derekgreene/topic-model-tutorial/blob/master/2%20-%20NMF%20Topic%20Models.ipynb)
76. **(didnt read) NTM -** [**neural topic modeling using embedded spaces**](https://github.com/elbamos/NeuralTopicModels) **with github code**
77. [**Another lda tutorial**](https://blog.intenthq.com/blog/automatic-topic-modelling-with-latent-dirichlet-allocation)
78. [**Comparing tweets using lda**](https://ink.library.smu.edu.sg/cgi/viewcontent.cgi?article=2374\&context=sis\_research)
79. [**Lda and w2v as features for some classification task**](https://www.kaggle.com/vukglisovic/classification-combining-lda-and-word2vec)
80. [**Improving ™ with embeddings**](https://github.com/datquocnguyen/LFTM)
81. [**w2v/doc2v for topic clustering - need to see the code to understand how they got clean topics, i assume a human rewrote it**](https://towardsdatascience.com/automatic-topic-clustering-using-doc2vec-e1cea88449c)

## **Mallet LDA**

1. [**Diff between lda and mallet**](https://groups.google.com/forum/#!topic/gensim/\_VO4otCV6cU) **- The inference algorithms in Mallet and Gensim are indeed different. Mallet uses Gibbs Sampling which is more precise than Gensim's faster and online Variational Bayes. There is a way to get relatively performance by increasing number of passes.**
2. [**Mallet in gensim blog post**](https://rare-technologies.com/tutorial-on-mallet-in-python/)
3. **Alpha beta in mallet:** [**contribution**](https://datascience.stackexchange.com/questions/199/what-does-the-alpha-and-beta-hyperparameters-contribute-to-in-latent-dirichlet-a)
   1. [**The default for alpha is 5.**](https://stackoverflow.com/questions/44561609/how-does-mallet-set-its-default-hyperparameters-for-lda-i-e-alpha-and-beta)**0 divided by the number of topics. You can think of this as five "pseudo-words" of weight on the uniform distribution over topics. If the document is short, we expect to stay closer to the uniform prior. If the document is long, we would feel more confident moving away from the prior.**
   2. **With hyperparameter optimization, the alpha value for each topic can be different. They usually become smaller than the default setting.**
   3. **The default value for beta is 0.01. This means that each topic has a weight on the uniform prior equal to the size of the vocabulary divided by 100. This seems to be a good value. With optimization turned on, the value rarely changes by more than a factor of two.**

## **Visualization**

1. **How to interpret topics using pyldaviz: Let’s interpret the topic visualization. Notice how topics are shown on the left while words are on the right. Here are the main things you should consider:**
   1. **Larger topics are more frequent in the corpus.**
   2. **Topics closer together are more similar, topics further apart are less similar.**
   3. **When you select a topic, you can see the most representative words for the selected topic. This measure can be a combination of how frequent or how discriminant the word is. You can adjust the weight of each property using the slider.**
   4. **Hovering over a word will adjust the topic sizes according to how representative the word is for the topic.**
   5. **\*\*\*\***[**pyLDAviz paper\*\*\*!**](https://cran.r-project.org/web/packages/LDAvis/vignettes/details.pdf)
   6.  [**pyLDAviz - what am i looking at ?**](https://github.com/explosion/spacy-notebooks/blob/master/notebooks/conference\_notebooks/modern\_nlp\_in\_python.ipynb) **by spacy.** \
       **There are a lot of moving parts in the visualization. Here's a brief summary:**



       1. **On the left, there is a plot of the "distance" between all of the topics (labeled as the Intertopic Distance Map)**
       2. **The plot is rendered in two dimensions according a** [**multidimensional scaling (MDS)**](https://en.wikipedia.org/wiki/Multidimensional\_scaling) **algorithm. Topics that are generally similar should be appear close together on the plot, while dissimilar topics should appear far apart.**
       3. **The relative size of a topic's circle in the plot corresponds to the relative frequency of the topic in the corpus.**
       4. **An individual topic may be selected for closer scrutiny by clicking on its circle, or entering its number in the "selected topic" box in the upper-left.**
       5. **On the right, there is a bar chart showing top terms.**
       6. **When no topic is selected in the plot on the left, the bar chart shows the top-30 most "salient" terms in the corpus. A term's saliency is a measure of both how frequent the term is in the corpus and how "distinctive" it is in distinguishing between different topics.**
       7. **When a particular topic is selected, the bar chart changes to show the top-30 most "relevant" terms for the selected topic. The relevance metric is controlled by the parameter λλ, which can be adjusted with a slider above the bar chart.**
          1. **Setting the λλ parameter close to 1.0 (the default) will rank the terms solely according to their probability within the topic.**
          2. **Setting λλ close to 0.0 will rank the terms solely according to their "distinctiveness" or "exclusivity" within the topic — i.e., terms that occur only in this topic, and do not occur in other topics.**
          3. **Setting λλ to values between 0.0 and 1.0 will result in an intermediate ranking, weighting term probability and exclusivity accordingly.**
          4. **Rolling the mouse over a term in the bar chart on the right will cause the topic circles to resize in the plot on the left, to show the strength of the relationship between the topics and the selected term.**
   7. **A more detailed explanation of the pyLDAvis visualization can be found** [**here**](https://cran.r-project.org/web/packages/LDAvis/vignettes/details.pdf)**. Unfortunately, though the data used by gensim and pyLDAvis are the same, they don't use the same ID numbers for topics. If you need to match up topics in gensim's LdaMulticore object and pyLDAvis' visualization, you have to dig through the terms manually.**
   8. [**Youtube on LDAvis explained**](http://stat-graphics.org/movies/ldavis.html)
   9. **Presentation:** [**More visualization options including ldavis**](https://speakerdeck.com/bmabey/visualizing-topic-models?slide=17)
   10. [**A pointer to the ldaviz fix**](https://github.com/RaRe-Technologies/gensim/issues/2069) **->** [**fix**](http://jeriwieringa.com/2018/07/17/pyLDAviz-and-Mallet/#comment-4018495276)**,** [**git code**](https://github.com/jerielizabeth/Gospel-of-Health-Notebooks/blob/master/blogPosts/pyLDAvis\_and\_Mallet.ipynb)

## **COHERENCE (Topic)**

1. [**What is?**](https://www.quora.com/What-is-topic-coherence)**,** [**Wiki on pmi**](https://en.wikipedia.org/wiki/Pointwise\_mutual\_information#cite\_note-Church1990-1)
2. [**Datacamp on coherence metrics, a comparison, read me.**](https://datascienceplus.com/evaluation-of-topic-modeling-topic-coherence/)
3. **Paper:** [**explains what is coherence**](http://aclweb.org/anthology/J90-1003)

![](https://lh4.googleusercontent.com/Jw5TMIwMSsVYMPRQxe5ZWKC3IDdj8KBAhd4y7nr5nLQZsxdhzDFM8gUVXjVnfZnoqfX-G1t2JjrpxKz2-IyO4WU5VTIOHUJgavudWCaaA18j7bbOf\_nUpewy874W-a9SyaOWDSfQ)

1. [**Umass vs C\_v, what are the diff?** ](https://groups.google.com/forum/#!topic/gensim/CsscFah0Ax8)
2. **Paper: umass, uci, nmpi, cv, cp etv** [**Exploring the Space of Topic Coherence Measures**](http://svn.aksw.org/papers/2015/WSDM\_Topic\_Evaluation/public.pdf)
3. **Paper:** [**Automatic evaluation of topic coherence**](https://mimno.infosci.cornell.edu/info6150/readings/N10-1012.pdf)&#x20;
4. **Paper:** [**exploring the space of topic coherence methods**](https://dl.acm.org/citation.cfm?id=2685324)
5. **Paper:** [**Relation between mutial information / entropy and pmi**](https://svn.spraakdata.gu.se/repos/gerlof/pub/www/Docs/npmi-pfd.pdf)
6. **Stackexchange:** [**coherence / pmi how to calc**](https://stats.stackexchange.com/questions/158790/topic-similarity-semantic-pmi-between-two-words-wikipedia)
7. **Paper:** [**Machine Reading Tea Leaves: Automatically Evaluating Topic Coherence and Topic Model Quality**](http://www.aclweb.org/anthology/E14-1056) **- perplexity needs unseen data, coherence doesnt**
8. [**Evaluation of topic modelling techniques for twitter**](https://www.cs.toronto.edu/\~jstolee/projects/topic.pdf) **lda lda-u btm w2vgmm**
9. **Paper:** [**Topic coherence measures**](https://svn.aksw.org/papers/2015/WSDM\_Topic\_Evaluation/public.pdf)
10. [**topic modelling from different domains**](http://proceedings.mlr.press/v32/chenf14.pdf)
11. **Paper:** [**Optimizing Semantic Coherence in Topic Models**](https://mimno.infosci.cornell.edu/papers/mimno-semantic-emnlp.pdf)
12. **Paper:** [**L-EnsNMF: Boosted Local Topic Discovery via Ensemble of Nonnegative Matrix Factorization** ](http://www.joonseok.net/papers/lensnmf.pdf)
13. **Paper:** [**Content matching between TV shows and advertisements through Latent Dirichlet Allocation** ](http://arno.uvt.nl/show.cgi?fid=145381)
14. **Paper:** [**Full-Text or Abstract? Examining Topic Coherence Scores Using Latent Dirichlet Allocation**](http://www.saf21.eu/wp-content/uploads/2017/09/5004a165.pdf)
15. **Paper:** [**Evaluating topic coherence**](https://pdfs.semanticscholar.org/03a0/62fdcd13c9287a2d4e1d6d057fd2e083281c.pdf) **- Abstract: Topic models extract representative word sets—called topics—from word counts in documents without requiring any semantic annotations. Topics are not guaranteed to be well interpretable, therefore, coherence measures have been proposed to distinguish between good and bad topics. Studies of topic coherence so far are limited to measures that score pairs of individual words. For the first time, we include coherence measures from scientific philosophy that score pairs of more complex word subsets and apply them to topic scoring.**

**Conclusion: The results of the first experiment show that if we are using the one-any, any-any and one-all coherences directly for optimization they are leading to meaningful word sets. The second experiment shows that these coherence measures are able to outperform the UCI coherence as well as the UMass coherence on these generated word sets. For evaluating LDA topics any-any and one-any coherences perform slightly better than the UCI coherence. The correlation of the UMass coherence and the human ratings is not as high as for the other coherences.**

1. **Code:** [**Evaluating topic coherence, using gensim umass or cv parameter**](https://datascienceplus.com/evaluation-of-topic-modeling-topic-coherence/) **- To conclude, there are many other approaches to evaluate Topic models such as Perplexity, but its poor indicator of the quality of the topics.Topic Visualization is also a good way to assess topic models. Topic Coherence measure is a good way to compare difference topic models based on their human-interpretability.The u\_mass and c\_v topic coherences capture the optimal number of topics by giving the interpretability of these topics a number called coherence score.**
2. **Formulas:** [**UCI vs UMASS**\
   ](http://qpleple.com/topic-coherence-to-evaluate-topic-models/)![](https://lh6.googleusercontent.com/aWrfeNX1FDBZYrIxAUSFw2ZcRQXyHTuxZ\_rgRXBhMPjvMY0sCQx-OlFKBRgId3Eynhv2532ZA5FWxB3Jz4Y8rjfAg5lnjwfxhRcmqfNq7d9rYrxWZrp146xarFHL6OkLSIVXPLEe)
3. [**Inferring the number of topics for gensim's LDA - perplexity, CM, AIC, and BIC**](https://stats.stackexchange.com/questions/322809/inferring-the-number-of-topics-for-gensims-lda-perplexity-cm-aic-and-bic)
4. [**Perplexity as a measure for LDA**](https://groups.google.com/forum/#!topic/gensim/tgJLVulf5xQ)
5. [**Finding number of topics using perplexity**](https://groups.google.com/forum/#!topic/gensim/TpuYRxhyIOc)
6. [**Coherence for tweets**](http://terrierteam.dcs.gla.ac.uk/publications/fang\_sigir\_2016\_examine.pdf)
7. **Presentation** [**Twitter DLA**](https://www.slideshare.net/akshayubhat/twitter-lda)**,** [**tweet pooling improvements**](http://users.cecs.anu.edu.au/\~ssanner/Papers/sigir13.pdf)**,** [**hierarchical summarization of tweets**](https://www.researchgate.net/publication/322359369\_Hierarchical\_Summarization\_of\_News\_Tweets\_with\_Twitter-LDA)**,** [**twitter LDA in java**](https://sites.google.com/site/lyangwww/code-data) [**on github**](https://github.com/minghui/Twitter-LDA)\
   **Papers:** [**TM of twitter timeline**](https://medium.com/@alexisperrier/topic-modeling-of-twitter-timelines-in-python-bb91fa90d98d)**,** [**in twitter aggregation by conversatoin**](https://www.aaai.org/ocs/index.php/ICWSM/ICWSM16/paper/download/13162/12778)**,** [**twitter topics using LDA**](http://uu.diva-portal.org/smash/get/diva2:904196/FULLTEXT01.pdf)**,** [**empirical study**](https://snap.stanford.edu/soma2010/papers/soma2010\_12.pdf) **,** &#x20;
8. [**Using regularization to improve PMI score and in turn coherence for LDA topics**](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.230.7738\&rep=rep1\&type=pdf)
9. [**Improving model precision - coherence using turkers for LDA**](https://pdfs.semanticscholar.org/1d29/f7a9e3135bba0339b9d70ecbda9d106b01d2.pdf)
10. [**Gensim**](https://radimrehurek.com/gensim/models/coherencemodel.html) **-** [ **paper about their algorithm and PMI/UCI etc.**](http://svn.aksw.org/papers/2015/WSDM\_Topic\_Evaluation/public.pdf)
11. [**Advice for coherence,**](https://gist.github.com/dsquareindia/ac9d3bf57579d02302f9655db8dfdd55) **then** [**Good vs bad model (50 vs 1 iterations) measuring u\_mass coherence**](https://markroxor.github.io/gensim/static/notebooks/topic\_coherence\_tutorial.html) **-** [**2nd code**](https://gist.github.com/dsquareindia/ac9d3bf57579d02302f9655db8dfdd55) **- “In your data we can see that there is a peak between 0-100 and a peak between 400-500. What I would think in this case is that "does \~480 topics make sense for the kind of data I have?" If not, you can just do an np.argmax for 0-100 topics and trade-off coherence score for simpler understanding. Otherwise just do an np.argmax on the full set.”**
12. [**Diff term weighting schemas for topic modeling, code plus paper**](https://github.com/cipriantruica/TM\_TESTS)
13. [**Workaround for pyLDAvis using LDA-Mallet**](http://jeriwieringa.com/2018/07/17/pyLDAviz-and-Mallet/#comment-4018495276)
14. [**pyLDAvis paper**](http://www.aclweb.org/anthology/W14-3110)
15. [**Visualizing LDA topics results** ](https://de.dariah.eu/tatom/topic\_model\_visualization.html)
16. [**Visualizing trends, topics, sentiment, heat maps, entities**](https://github.com/Lissy93/twitter-sentiment-visualisation) **- really good**
17. **Topic stability Metric, a novel method, compared against jaccard, spearman, silhouette.:** [**Measuring LDA Topic Stability from Clusters of Replicated Runs**](https://arxiv.org/pdf/1808.08098.pdf)\


## **LDA2VEC**

1. **“if you want to rework your own topic models that, say, jointly correlate an article’s topics with votes or predict topics over users then you might be interested in** [**lda2vec**](https://github.com/cemoody/lda2vec)**.”**
2. [**Datacamp intro**](https://www.datacamp.com/community/tutorials/lda2vec-topic-model)
3. [**Original blog**](https://multithreaded.stitchfix.com/blog/2016/05/27/lda2vec/#topic=38\&lambda=1\&term=) **- I just learned about these papers which are quite similar:** [**Gaussian LDA for Topic Word Embeddings**](http://www.aclweb.org/anthology/P15-1077) **and** [**Nonparametric Spherical Topic Modeling with Word Embeddings**](http://arxiv.org/abs/1604.00126)**.**
4. [**Moody’s Slide Share**](https://www.slideshare.net/ChristopherMoody3/word2vec-lda-and-introducing-a-new-hybrid-algorithm-lda2vec-57135994) **(excellent read)**
5. [**Docs**](http://lda2vec.readthedocs.io/en/latest/?badge=latest)
6. [**Original Git**](https://github.com/cemoody/lda2vec) **+** [**Excellent notebook example**](http://nbviewer.jupyter.org/github/cemoody/lda2vec/blob/master/examples/twenty\_newsgroups/lda2vec/lda2vec.ipynb#topic=0\&lambda=1\&term=)
7. [**Tf implementation**](https://github.com/meereeum/lda2vec-tf)**,** [**another more recent one tf 1.5**](https://github.com/nateraw/Lda2vec-Tensorflow)
8. [**Another blog explaining about lda etc**](https://datawarrior.wordpress.com/tag/lda2vec/)**,** [**post**](https://datawarrior.wordpress.com/2016/02/15/lda2vec-a-hybrid-of-lda-and-word2vec/)**,** [**post**](https://datawarrior.wordpress.com/2016/04/20/local-and-global-words-and-topics/)
9. [**Lda2vec in tf**](https://github.com/meereeum/lda2vec-tf)**,** [**tf 1.5**](https://github.com/nateraw/Lda2vec-Tensorflow)**,**&#x20;
10. [**Comparing lda2vec to lda**](https://medium.com/scaleabout/a-gentle-introduction-to-doc2vec-db3e8c0cce5e)
11. **Youtube:** [**lda/doc2vec with pca examples**](https://www.youtube.com/watch?v=i3Opb3-QNX4)
12. [**Example on gh**](https://github.com/BoPengGit/LDA-Doc2Vec-example-with-PCA-LDAvis-visualization/blob/master/Doc2Vec/Doc2Vec2.py) **on jupyter**

## **TOP2VEC**

1. [Git](https://github.com/ddangelov/Top2Vec), [paper](https://arxiv.org/pdf/2008.09470.pdf)
2. Topic modeling with distillibert [on medium](https://towardsdatascience.com/topic-modeling-with-bert-779f7db187e6), [bertTopic](https://towardsdatascience.com/interactive-topic-modeling-with-bertopic-1ea55e7d73d8)!, c-tfidf, umap, hdbscan, merging similar topics, visualization, [berTopic (same method as the above)](https://github.com/MaartenGr/BERTopic)
3. [Medium with the same general method](https://towardsdatascience.com/topic-modeling-with-bert-779f7db187e6)
4. [new way of modeling topics](https://towardsdatascience.com/top2vec-new-way-of-topic-modelling-bea165eeac4a)
