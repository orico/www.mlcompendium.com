# Validation

### DATASETS RELIABILITY & CORRECTNESS&#x20;

1\. [Clever Hans effect](https://thegradient.pub/nlps-clever-hans-moment-has-arrived/?fbclid=IwAR3vSx9EjXcSPhXU3Jyf7aWpTqhbVDARnh3qGpSw0rysv9rLeGyZFFCPnJA) - in relations to cues left in the dataset that models find, instead of actually solving the defined task!

* Ablating, i.e. removing, part of a model and observing the impact this has on performance is a common method for verifying that the part in question is useful. If performance doesn't go down, then the part is useless and should be removed. Carrying this method over to datasets, it should become common practice to perform dataset ablations, as well, for example:
* Provide only incomplete input (as done in the reviewed paper): This verifies that the complete input is required. If not, the dataset contains cues that allow taking shortcuts.
* Shuffle the input: This verifies the importance of word (or sentence) order. If a bag-of-words/sentences gives similar results, even though the task requires sequential reasoning, then the model has not learned sequential reasoning and the dataset contains cues that allow the model to "solve" the task without it.
* Assign random labels: How much does performance drop if ten percent of instances are relabeled randomly? How much with all random labels? If scores don't change much, the model probably didn't learning anything interesting about the task.
* Randomly replace content words: How much does performance drop if all noun phrases and/or verb phrases are replaced with random noun phrases and verbs? If not much, the dataset may provide unintended non-content cues, such as sentence length or distribution of function words.

[2. Paper](https://arxiv.org/abs/1908.05267?fbclid=IwAR1xOHxCF3gewyijMYAfZJSysu88Y8lgRIT2OiG-jWQav4zcbPqGYSoFFkk)\


### UNIT / DATA TESTS

1. A great :P [unit test and logging](https://towardsdatascience.com/unit-testing-and-logging-for-data-science-d7fb8fd5d217?fbclid=IwAR3pze0DtV-2Q4L4ysPyjrInk7LB89mdiodxlEUTv4rv37ZoDzl\_2I4ZbgA) post on medium - it’s actually mine :)
2. A mind blowing [lecture](https://www.youtube.com/watch?v=1fHGXOfiDO0\&feature=youtu.be\&fbclid=IwAR1bKByLgdYBDoBEr-e6Pw0Un5o0wvOg1yp4C-q4AoWZ1QuBEopTFFn0Gdw) about unit testing your data using Voluptuous & engrade & TDDA lecture
3. [Great expectations](https://greatexpectations.io), [article](https://github.blog/2020-10-01-keeping-your-data-pipelines-healthy-with-the-great-expectations-github-action/), “TDDA” for Unit tests and CI
4. [DataProfiler git](https://github.com/capitalone/DataProfiler)
5. [Unit tests in python](https://jeffknupp.com/blog/2013/12/09/improve-your-python-understanding-unit-testing/)
6. [Unit tests in python - youtube](https://www.youtube.com/watch?v=6tNS--WetLI)
7. [Unit tests asserts](https://docs.python.org/3/library/unittest.html#unittest.TestCase.debug)
8. [Auger - automatic unit tests, has a blog post inside](https://github.com/laffra/auger), doesn't work with py 3+
9. [A rather naive unit tests article aimed for DS](https://medium.com/@danielhen/unit-tests-for-data-science-the-main-use-cases-1928d9e7a4d4)
10. A good pytest [tutorial](https://www.tutorialspoint.com/pytest/index.htm)
11. [Mock](https://medium.com/@yasufumy/python-mock-basics-674c33de1ced), [mock 2](https://medium.com/python-pandemonium/python-mocking-you-are-a-tricksy-beast-6c4a1f8d19b2)

### REGULATION FOR AI

1. [Preparing for EU regulations](https://towardsdatascience.com/how-ai-leaders-should-prepare-for-the-looming-eu-regulations-99e9d4f4c039) by MonaLabs
2. [EU regulation DOC](https://drive.google.com/file/d/1ZaBPsfor\_aHKNeeyXxk9uJfTru747EOn/view)
3. [EIOPA](https://www.eiopa.europa.eu/content/eiopa-publishes-report-artificial-intelligence-governance-principles\_en) - regulation for insurance companies.
4. Ethics and regulations in Israel
   1. [First Report by the intelligence committee](https://www.globes.co.il/news/article.aspx?did=1001307714) headed by prof. Itzik ben israel and prof. evyatar matanya&#x20;
   2. [Second report by AI and data science committee ](https://innovationisrael.org.il/sites/default/files/%D7%93%D7%95%D7%97%20%D7%A1%D7%95%D7%A4%D7%99%20%D7%A1%D7%99%D7%9B%D7%95%D7%9D%20%D7%95%D7%95%D7%A2%D7%93%D7%AA%20%D7%AA%D7%9C%D7%9D%20%D7%9C%D7%AA%D7%9B%D7%A0%D7%99%D7%AA%20%D7%9E%D7%95%D7%A4%20%D7%9C%D7%90%D7%95%D7%9E%D7%99%D7%AA%20%D7%91%D7%91%D7%99%D7%A0%D7%94%20%D7%9E%D7%9C%D7%90%D7%9B%D7%95%D7%AA%D7%99%D7%AA%20-.pdf)
   3. Third by meizam leumi for AI systems in [ethics and regulation in israel](https://machinelearning.co.il/4330/israelaiethicsreport/#more-4330), [lecture](https://machinelearning.co.il/3349/googleai/)

### FAIRNESS, ACCOUNTABILITY & TRANSPARENCY ML

1. FATML [website](https://www.fatml.org) - The past few years have seen growing recognition that machine learning raises novel challenges for ensuring non-discrimination, due process, and understandability in decision-making. In particular, policymakers, regulators, and advocates have expressed fears about the potentially discriminatory impact of machine learning, with many calling for further technical research into the dangers of inadvertently encoding bias into automated decisions.

At the same time, there is increasing alarm that the complexity of machine learning may reduce the justification for consequential decisions to “the algorithm made me do it.”

1. [Principles and best practices](https://www.fatml.org/resources/principles-and-best-practices), [projects](https://www.fatml.org/resources/relevant-projects)
2. [FAccT](https://facctconference.org) - A computer science conference with a cross-disciplinary focus that brings together researchers and practitioners interested in fairness, accountability, and transparency in socio-technical systems.
3. [Paper - there is no fairness, enforcing fairness can improve accuracy](https://openreview.net/forum?id=wXoHN-Zoel\&fbclid=IwAR1MZArpfpu8L8ildamF0ngnUbKgD8-9NFBCXVo0JKwS6yP9g-2BJmWUv68)
4. [Google on responsible ai practices](https://ai.google/responsibilities/responsible-ai-practices/) see also PAIR
5. [Bengio on ai](https://www.wired.com/story/ai-pioneer-algorithms-understand-why/?fbclid=IwAR03uWEmVSjrOmP4dp77v\_mdjPAXOsKPams\_xsUOKameKbuzY8JN4brGC9o)
6. [Poisoning attacks on fairness](https://arxiv.org/pdf/2004.07401.pdf) - Research in adversarial machine learning has shown how the performance of machine learning models can be seriously compromised by injecting even a small fraction of poisoning points into the training data. We empirically show that our attack is effective not only in the white-box setting, in which the attacker has full access to the target model, but also in a more challenging black-box scenario in which the attacks are optimized against a substitute model and then transferred to the target model
7. A [series of articles](https://jonathan-hui.medium.com/ai-bias-fairness-series-ce21ebf7b2e9) about Bias & Fairness by Johnathan Hui
   1. [In Clinical research](https://jonathan-hui.medium.com/bias-in-clinical-research-data-science-machine-learning-deep-learning-40a8786a5046) - Selection , Sample , Time , Attrition , Survivorship, reporting, funding, citation, Volunteer , self-selection , non-response, pre-screening , healthy person, membership, ascertainment, performance, berkson admission, neyman, measurement, observer, expectation, response, self reporting, social desirability, recall, acquiescence agreement, leading, courtesy, attention verification, lead time, immortal time, misclassification, chronological, detection, spectrum, cofounder, susceptibility, collider, simpson,  ommited, allocation, channeling.
   2. [AI](https://jonathan-hui.medium.com/ai-bias-b85c86bbca90) - known cases in Vision, NLP - sentiment, embedding, language models, historical, compass, recommender, datasets.
   3. [address AI Bias with Fairness criteria and tools](https://jonathan-hui.medium.com/address-ai-bias-with-fairness-criteria-tools-9af1ab8e4289) - per population, predictive parity, calibration by group

####

#### FAIRNESS TOOLS

1. [PII tools, by gensim](https://pii-tools.com)
2. [Fair-learn](https://github.com/fairlearn/fairlearn) A Python package to assess and improve fairness of machine learning models.\
   ![](https://lh5.googleusercontent.com/ovdlVfds0jLUJzmmntUN70j5Qbsfq9hberlTf\_evGgDKVGvFVHblHc-EbrbhmTviVRUVXJG9B2TlkcgSwO7vwt43y7tsia1gTjPJitTY2pCNAH\_PWKxkrsXNcfKKHASqT3rW23FC)
3. [Sk-lego](https://scikit-lego.readthedocs.io/en/latest/fairness.html)

![](https://lh6.googleusercontent.com/624RfKvyH\_U6OG\_VHISCDieoZ2Z4hil1tB9IyFynrssQme2iRPITK8am770Q\_yg8FG6UJzs0FIiwx1-OoxQEOXSFPGBoZk0fwqQ4sInTpBRdmo62AIxFZ\_wZywz3nCJLdAucfz9X)

1. Regression\
   ![](https://lh5.googleusercontent.com/h\_vzduMzENSsIUcgRY09p2XPtyrF6Mr5Wqho5GFZfdfjynkMzwkAGhABGv1cYOZ1RE\_PViDDdt\_J2WTt8kkWMiPOIv9d\_zXZP\_17LgFGl\_qnG-z-82\_7rP\_RUrbJ3JiTefBY1XTx)
2. classification![](https://lh6.googleusercontent.com/IbIrp6\_AZtn2sebHBGICWiHsWmXwgSFN2Zmo\_8Aqo4aVkmyETQvM-gvubm71wXCuL\_yu7E7OliwZYTY0nq4wlbZngzkdBVwX6U6VZt9-lYS-9RWyXNYRTOe5VacTZqHgGaX5CI\_8)
3. ![](https://lh6.googleusercontent.com/cbsPagQpH6fQyie5FVQphEAtkYdo6Z4\_jDzaP3ZkB-CtsJiN5-6et3ggYM9-oTohaITrjetZfQoqSL818tfK6SaHUFn6KTeSNpsp4GgH2xFw6ttPUwu5zf7mxxD2ekooCqI0wNd5)
4. ![](https://lh6.googleusercontent.com/upoEK0-G4\_0fe8xJ01s8PjtLQiI6Hz49BFIqjOmV14zKrKlRbFGF6pDwXSRxE8zkRqIO0iywNDzQ55Vwh2ac6xpZPCOU5646Bvs59xUwkCOo3EAekaVLlO9rHP53ag4TE0R1\_6vV)
5. ![](https://lh6.googleusercontent.com/Ze5Oc1TTNzIPaCJnkdy0iflUutgPb2w7nl2zd7s4uya\_kz0tTR0RMFvJGrFFMs4GKVYYWuo2sc5qIPKzZBpHmTKtH0KJYu4AfrP8pc8xbmVq1vuKJ1zcBrTUAVCuARQ41GdCcEdZ)
6. information filter\
   ![](https://lh3.googleusercontent.com/0-2-4owRs592iwho\_Yn62nZVWpYdCs6f9ZQyudZmqAoli1KbuTwQLOI8YlP-ZLzK5c-eWmzERHC976Dp7pLJVT2UEHRf\_kee-g3ltI8kDhm6-ATzE39-KqK80t4chbk9Bao3B27F)

M. Zafar et al. (2017), Fairness Constraints: Mechanisms for Fair Classification

M. Hardt, E. Price and N. Srebro (2016), Equality of Opportunity in Supervised Learning

###

### INTERPRETABLE / EXPLAINABLE AI (XAI)

![](https://lh3.googleusercontent.com/gQgeZyxlXU37RydzNxXz1VitIZ-vdWr0YGy59EphP1cD8KqEE3VB58CGxxORvdmNuSLeRcRaytp7nJkFZveApPd4Fq8xEOV51ZSuXJsFdkU9EpL8d1cQRKzoCEpBjqARmiRD0NEV)

1. [A curated document about XAI research resources. ](https://docs.google.com/spreadsheets/d/1uQy6a3BfxOXI8Nh3ECH0bqqSc95zpy4eIp\_9JAMBkKg/edit?usp=sharing)
2. Interpretability and Explainability in Machine Learning [course](https://interpretable-ml-class.github.io) / slides. Understanding, evaluating, rule based, prototype based, risk scores, generalized additive models, explaining black box, visualizing, feature importance, actionable explanations, casual models, human in the loop, connection with debugging.&#x20;
3. [Explainable Machine Learning: Understanding the Limits & Pushing the Boundaries](https://drive.google.com/file/d/1xn2dCDAeEEhB\_rex202KxMPqIPj31fZ4/view) a tutorial by Hima Lakkaraju (tutorial [VIDEO](https://www.chilconference.org/tutorial\_T04.html), [youtube](https://www.youtube.com/watch?v=K6-ujR\_67eY), [twitter](https://twitter.com/hima\_lakkaraju/status/1390759698224271361))\
   ![](https://lh3.googleusercontent.com/rO4qszA6Hz3L21ZL3YOJB3GNG9u-Q0rGGQ0QxamCYq6MLwHPxkHhk5GUGhVpMKTM0EJH0SHDIr5Tts9vCvjTKWZzrKDdoaE8jfdLDV3Dstu66HiNYvKmoRBQDAEothlrQM7FSLdD)
4. [explainML tutorial](https://explainml-tutorial.github.io/neurips20)
5. [When not to trust explanations :)](https://docs.google.com/presentation/d/10a0PNKwoV3a1XChzvY-T1mWudtzUIZi3sCMzVwGSYfM/edit#slide=id.p)
6. From the above image: [Paper: Principles and practice of explainable models](https://arxiv.org/abs/2009.11698) - a really good review for everything XAI - “a survey to help industry practitioners (but also data scientists more broadly) understand the field of explainable machine learning better and apply the right tools. Our latter sections build a narrative around a putative data scientist, and discuss how she might go about explaining her models by asking the right questions. From an organization viewpoint, after motivating the area broadly, we discuss the main developments, including the principles that allow us to study transparent models vs opaque models, as well as model-specific or model-agnostic post-hoc explainability approaches. We also briefly reflect on deep learning models, and conclude with a discussion about future research directions.”
7. [Book: interpretable machine learning](https://christophm.github.io/interpretable-ml-book/agnostic.html), [christoph mulner](https://christophm.github.io)
8. (great) [Interpretability overview,](https://thegradient.pub/interpretability-in-ml-a-broad-overview/?fbclid=IwAR2ltYQWbS5jixIJzAnFg8dz1A-9y9eGIMxQfpB\_Pp5x9knP1Y4JhQg3xgI) transparent (simultability, decomposability, algorithmic transparency) post-hoc interpretability (text explanation, visual local, explanation by example,), evaluation, utility.&#x20;
9. [Medium: the great debate](https://medium.com/swlh/the-great-ai-debate-interpretability-1d139167b55)

![](https://lh3.googleusercontent.com/6CjEUFdZIAgWwd3d\_jpzNzsllK2nmSX0SOoH2klh9W3k2djHHvjTy8Sf-AZRkGiitO1mJ95mAjUiMW4HFxEJqmRCICBw1luP2EiXU4fiQniqnIqEl\_NodTsK8iYNC-7mQilNXGuu)

1. [Paper: pitfalls to avoid when interpreting ML models](https://arxiv.org/abs/2007.04131) “y. A growing number of techniques provide model interpretations, but can lead to wrong conclusions if applied incorrectly. We illustrate pitfalls of ML model interpretation such as bad model generalization, dependent features, feature interactions or unjustified causal interpretations. Our paper addresses ML practitioners by raising awareness of pitfalls and pointing out solutions for correct model interpretation, as well as ML researchers by discussing open issues for further research.” - mulner et al.

![](https://lh5.googleusercontent.com/KoycayTSdi3gu8cp6TrExpk-yJrhzlZfEiz1RzwtprTwJCnkz8fEpMhV9DlMpyPf-f\_qqbfoVEQDABxGzQSwAjbX5S4p0dJL1dso\_MdlO\_SQRyInrUA3Lu70jdPHU5wZbKlmfvGZ)

1. \*\*\* [whitening a black box.](https://francescopochetti.com/whitening-a-black-box-how-to-interpret-a-ml-model/) This is very good, includes eli5, lime, shap, many others.
2. Book: [exploratory model analysis](https://pbiecek.github.io/ema/)&#x20;
3. [Alibi-explain](https://github.com/SeldonIO/alibi) - White-box and black-box ML model explanation library. [Alibi](https://docs.seldon.io/projects/alibi) is an open source Python library aimed at machine learning model inspection and interpretation. The focus of the library is to provide high-quality implementations of black-box, white-box, local and global explanation methods for classification and regression models.

![](https://lh3.googleusercontent.com/GclfWroRlYDUXz8j-72u-4\_uudzQIUJbPR5BxUku\_yE5cAOwfYL6rr1RH9nTbq6VXtaMv1KsnOr3DQTdEqFTT1CTdV6KCZXTqIoYmefzQEDtyrwAirvjbEhIQ2ARmwKGGJEczoSW)

1. [Hands on explainable ai](https://www.youtube.com/watch?v=1mNhPoab9JI\&fbclid=IwAR1cV\_\_3zBClI-mq3XpJfgn691xB7EM5gdZpejJ86wnrsVoiGmQFY9P5Uho) youtube, [git](https://github.com/PacktPublishing/Hands-On-Explainable-AI-XAI-with-Python?fbclid=IwAR012IQFa4ce3camoD13iIRyCfQlWPi3HwQs8VDjIGgFnGdcm3xkq7zir-U)
2. [Explainable methods](https://towardsdatascience.com/interpretable-machine-learning-with-xgboost-9ec80d148d27) are not always consistent and do not agree with each other, this article has a make-sense explanation and flow for using shap and its many plots.

![](https://lh3.googleusercontent.com/kfXRcqozOopXu52ZgoLpL3JLi4ZFPTUpC54bD2rOy1nq3LY1hoj5ywOM-ExGR8blbP4MAv5hvCqTmMMpIW\_l31Si3ABloWld-L34Jo0rUfvDcixs\_fE-JInKCHwzPlTCITuQzz3l)

1. Intro to shap and lime, [part 1](https://blog.dominodatalab.com/shap-lime-python-libraries-part-1-great-explainers-pros-cons/), [part 2](https://blog.dominodatalab.com/shap-lime-python-libraries-part-2-using-shap-lime/)
2. Lime
   1. [\*\*\* how lime works behind the scenes](https://medium.com/analytics-vidhya/explain-your-model-with-lime-5a1a5867b423)
   2. [LIME to interpret models](https://www.oreilly.com/learning/introduction-to-local-interpretable-model-agnostic-explanations-lime) NLP and IMAGE, [github](https://github.com/marcotcr/lime)- In the experiments in [our research paper](http://arxiv.org/abs/1602.04938), we demonstrate that both machine learning experts and lay users greatly benefit from explanations similar to Figures 5 and 6 and are able to choose which models generalize better, improve models by changing them, and get crucial insights into the models' behavior.
3. Anchor
   1. [Anchor from the authors of Lime,](https://github.com/marcotcr/anchor) - An anchor explanation is a rule that sufficiently “anchors” the prediction locally – such that changes to the rest of the feature values of the instance do not matter. In other words, for instances on which the anchor holds, the prediction is (almost) always the same.
4. Shap:&#x20;
   1. Medium [Intro to lime and shap](https://towardsdatascience.com/explain-nlp-models-with-lime-shap-5c5a9f84d59b)
   2. \*\*\*\* In depth [SHAP](https://towardsdatascience.com/introducing-shap-decision-plots-52ed3b4a1cba)
   3. [Github](https://github.com/slundberg/shap)
   4. [Country happiness using shap](https://sararobinson.dev/2019/03/24/preventing-bias-machine-learning.html)
   5. [Stackoverflow example, predicting tags, pandas keras etc](https://stackoverflow.blog/2019/05/06/predicting-stack-overflow-tags-with-googles-cloud-ai/)
   6. [Intro to shapely and shap](https://towardsdatascience.com/a-new-perspective-on-shapley-values-an-intro-to-shapley-and-shap-6f1c70161e8d?)
   7. [Fiddler on shap](https://medium.com/fiddlerlabs/case-study-explaining-credit-modeling-predictions-with-shap-2a7b3f86ec12)
   8. [Shapash - a web app for Lime and shap. ](https://github.com/MAIF/shapash)
5. SHAP advanced
   1. [Official shap tutorial on their plots, you can never read this too many times.](https://slundberg.github.io/shap/notebooks/plots/decision\_plot.html)
   2. [What are shap values on kaggle](https://www.kaggle.com/dansbecker/shap-values) - whatever you do start with this
   3. [Shap values on kaggle #2](https://www.kaggle.com/dansbecker/advanced-uses-of-shap-values) - continue with this
   4. How to calculate Shap values per class based on this graph

![](https://lh6.googleusercontent.com/axOwNRnxPTrRBwWDcOA8D\_kCv5NyjBVA-wS3vo1k\_TqgwtBNaRUb\_OUsJaMf3x2cORwhY83SqcDEQkuHTbZYLRhL\_byw64PjxFmRJLBnOwll1XRDbIYkoUm-TAogy\_2lgRBgCqh3)

1. Shap [intro](https://towardsdatascience.com/explain-your-model-with-the-shap-values-bc36aac4de3d), [part 2](https://towardsdatascience.com/explain-any-models-with-the-shap-values-use-the-kernelexplainer-79de9464897a) with many algo examples and an explanation about the four plots.
2. [A thorough post about the many ways of explaining a model, from regression, to bayes, to trees, forests, lime, beta, feature selection/elimination](https://lilianweng.github.io/lil-log/2017/08/01/how-to-explain-the-prediction-of-a-machine-learning-model.html#interpretable-models)
3. [Trusting models](https://arxiv.org/pdf/1602.04938.pdf)
4. [3. Interpret using uncertainty](https://becominghuman.ai/using-uncertainty-to-interpret-your-model-67a97c28fea5)
5. [Keras-vis](https://github.com/raghakot/keras-vis) for cnns, 3 methods, activation maximization, saliency and class activation maps
6. [The notebook!](https://github.com/FraPochetti/KagglePlaygrounds/blob/master/InterpretableML.ipynb) [Blog](https://francescopochetti.com/whitening-a-black-box-how-to-interpret-a-ml-model/)
7. [More resources!](https://docs.google.com/spreadsheets/d/1uQy6a3BfxOXI8Nh3ECH0bqqSc95zpy4eIp\_9JAMBkKg/edit#gid=0)
8. [Visualizing the impact of feature attribution baseline](https://distill.pub/2020/attribution-baselines/) - Path attribution methods are a gradient-based way of explaining deep models. These methods require choosing a hyperparameter known as the baseline input. What does this hyperparameter mean, and how important is it? In this article, we investigate these questions using image classification networks as a case study. We discuss several different ways to choose a baseline input and the assumptions that are implicit in each baseline. Although we focus here on path attribution methods, our discussion of baselines is closely connected with the concept of missingness in the feature space - a concept that is critical to interpretability research.
9. WHAT IF TOOL - GOOGLE, [notebook](https://colab.research.google.com/github/PAIR-code/what-if-tool/blob/master/WIT\_Smile\_Detector.ipynb), [walkthrough](https://pair-code.github.io/what-if-tool/learn/tutorials/walkthrough/)
10. [Language interpretability tool (LIT) -](https://pair-code.github.io/lit/) The Language Interpretability Tool (LIT) is an open-source platform for visualization and understanding of NLP models.
11. [Stop Explaining Black Box Machine Learning Models for High Stakes Decisions and Use Interpretable Models Instead](https://arxiv.org/abs/1811.10154) - “trying to \textit{explain} black box models, rather than creating models that are \textit{interpretable} in the first place, is likely to perpetuate bad practices and can potentially cause catastrophic harm to society. There is a way forward -- it is to design models that are inherently interpretable. This manuscript clarifies the chasm between explaining black boxes and using inherently interpretable models, outlines several key reasons why explainable black boxes should be avoided in high-stakes decisions, identifies challenges to interpretable machine learning, and provides several example applications where interpretable models could potentially replace black box models in criminal justice, healthcare, and computer vision.”
12. [Using genetic algorithms](https://towardsdatascience.com/interpreting-black-box-machine-learning-models-with-genetic-algorithms-a803bfd134cb)
13. [ Google’s what-if tool](https://pair-code.github.io/what-if-tool/demos/image.html) from [PAIR](https://pair.withgoogle.com)
14. [Boruta](https://github.com/scikit-learn-contrib/boruta\_py) ([medium](https://towardsdatascience.com/boruta-explained-the-way-i-wish-someone-explained-it-to-me-4489d70e154a)) was designed to automatically perform feature selection on a dataset using randomized features, i.e., measuring valid features against their shadow/noisy counterparts.
15. [InterpretML](https://interpret.ml) by Microsoft, [git](https://github.com/interpretml/interpret).

### WHY WE SHOULDN’T TRUST MODELS

1. [Clever Hans effect for NLP](https://thegradient.pub/nlps-clever-hans-moment-has-arrived/)
   1. Datasets need more love
   2. Datasets ablation and public beta
   3. Inter-prediction agreement
2. Behavioral testing and CHECKLIST
   1. [Blog](https://amitness.com/2020/07/checklist/), [Youtube](https://www.youtube.com/watch?v=L3gaWctPg6E), [paper](https://arxiv.org/pdf/2005.04118.pdf), [git](https://github.com/marcotcr/checklist)
   2. [Yonatan hadar on the subject in hebrew](https://www.facebook.com/groups/MDLI1/permalink/1627671704063538/)

### BIAS

1. arize.ai on [model bias](https://arize.com/understanding-bias-in-ml-models/#MLMonitoring).

![](<.gitbook/assets/image (3).png>)

### DEBIASING MODELS

1. [Adversarial removal of demographic features](https://arxiv.org/abs/1808.06640) - “We show that demographic information of authors is encoded in -- and can be recovered from -- the intermediate representations learned by text-based neural classifiers. The implication is that decisions of classifiers trained on textual data are not agnostic to -- and likely condition on -- demographic attributes. “\
   “we explore several techniques to improve the effectiveness of the adversarial component. Our main conclusion is a cautionary one: do not rely on the adversarial training to achieve invariant representation to sensitive features.”\
   \

2. [Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection](https://arxiv.org/abs/2004.07667) (paper) , [github](https://github.com/shauli-ravfogel/nullspace\_projection), [presentation](https://docs.google.com/presentation/d/1Xi5HLpvvRE8BqcNBZMyPS4gBa0i0lqZvRebz-AZxAPA/edit) by Shauli et al. - removing biased information such as gender from an embedding space using nullspace projection.\
   The objective is this: give a representation of text, for example BERT embeddings of many resumes/CVs, we want to achieve a state where a certain quality, for example a gender representation of the person who wrote this resume is not encoded in X. they used the light version definition for “not encoded”, i.e., you cant predict the quality from the representation with a higher than random score, using a linear model. I.e., every linear model you will train, will not be able to predict the person’s gender out of the embedding space and will reach a 50% accuracy.\
   This is done by an iterative process that includes. 1. Linear model training to predict the quality of the concept from the representation. 2. Performing ‘projection to null space’ for the linear classifier, this is an acceptable linear algebra calculation that has a meaning of zeroing the representation from the projection on the separation place that the linear model is representing, making the model useless. I.e., it will always predict the zero vector. This is done iteratively on the neutralized output, i.e., in the second iteration we look for an alternative way to predict the gender out of X, until we reach 50% accuracy (or some other metric you want to measure) at this point we have neutralized all the linear directions in the embedding space, that were predictive to the gender of the author.

For a matrix W, the null space is a sub-space of all X such that WX=0, i.e., W maps X to the zero vector, this is a linear projection of the zero vector into a subspace. For example you can take a 3d vectors and calculate its projection on XY.

1. Can we extinct predictive samples? Its an open question, Maybe we can use influence functions?

[Understanding Black-box Predictions via Influence Functions](https://arxiv.org/pdf/1703.04730.pdf) - How can we explain the predictions of a blackbox model? In this paper, we use influence functions — a classic technique from robust statistics — to trace a model’s prediction through the learning algorithm and back to its training data, thereby identifying training points most responsible for a given prediction.

We show that even on non-convex and non-differentiable models where the theory breaks down, approximations to influence functions can still provide valuable information. On linear models and convolutional neural networks, we demonstrate that influence functions are useful for multiple purposes: understanding model behavior, debugging models, detecting dataset errors, and even creating visually indistinguishable training-set attacks.

1. [Removing ‘gender bias using pair mean pca](https://stackoverflow.com/questions/48019843/pca-on-word2vec-embeddings)
2. [Bias detector by intuit](https://github.com/intuit/bias-detector) - Based on first and last name/zip code the package analyzes the probability of the user belonging to different genders/races. Then, the model predictions per gender/race are compared using various bias metrics.
3.
4.

###

### PRIVACY

1. [Privacy in DataScience](http://www.unsupervised-podcast.xyz/ab55d406) podcast
2. [Fairness in AI](http://www.unsupervised-podcast.xyz/5d7fc118)

### DIFFERENTIAL PRIVACY

1. [Differential privacy](https://georgianpartners.com/what-is-differential-privacy/) has emerged as a major area of research in the effort to prevent the identification of individuals and private data. It is a mathematical definition for the privacy loss that results to individuals when their private information is used to create AI products. It works by injecting noise into a dataset, during a machine learning training process, or into the output of a machine learning model, without introducing significant adverse effects on data analysis or model performance. It achieves this by calibrating the noise level to the sensitivity of the algorithm. The result is a differentially private dataset or model that cannot be reverse engineered by an attacker, while still providing useful information. Uses BOTLON & EPSILON
2. [youtube](https://www.youtube.com/watch?v=gI0wk1CXlsQ\&feature=emb\_title)

### ANONYMIZATION

1. [Using NER (omri mendels)](https://towardsdatascience.com/nlp-approaches-to-data-anonymization-1fb5bde6b929)

### DE-ANONYMIZATION

1. GPT2 - [Of language datasets\
   ](https://arxiv.org/pdf/2012.07805.pdf)![](https://lh5.googleusercontent.com/XkrwLQ2tm0xAA3bvGOQ5H3WkwWOgSwpzFal4rvRrTmcB6vzSrbGO-OK8Q8vxdQ4zhbT\_\_MJyfbpwnIesc5BPmCdhr210Vlqy7pjipEbgezxW9WcP1CxL7uQsPQuIgmGCr1LHJY9w)
