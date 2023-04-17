# Annotation & Disagreement

## Tools

1. [**Snorkel**](https://www.snorkel.org/use-cases/) **- using weak supervision to create less noisy labelled datasets**
   1. [**Git**](https://github.com/snorkel-team/snorkel)
   2. [**Medium**](https://towardsdatascience.com/introducing-snorkel-27e4b0e6ecff)
2. [**Snorkel metal**](https://jdunnmon.github.io/metal\_deem.pdf) **weak supervision for multi-task learning.** [**Conversation**](https://spectrum.chat/snorkel/help/hierarchical-labelling-example\~aa4d8617-d287-43a6-865e-7c9034888363)**,** [**git**](https://github.com/HazyResearch/metal/blob/master/tutorials/Multitask.ipynb)
   1. **Yes, the Snorkel project has included work before on hierarchical labeling scenarios. The main papers detailing our results include the DEEM workshop paper you referenced (**[**https://dl.acm.org/doi/abs/10.1145/3209889.3209898**](https://dl.acm.org/doi/abs/10.1145/3209889.3209898)**) and the more complete paper presented at AAAI (**[**https://arxiv.org/abs/1810.02840**](https://arxiv.org/abs/1810.02840)**). Before the Snorkel and Snorkel MeTaL projects were merged in Snorkel v0.9, the Snorkel MeTaL project included an interface for explicitly specifying hierarchies between tasks which was utilized by the label model and could be used to automatically compile a multi-task end model as well (demo here:** [**https://github.com/HazyResearch/metal/blob/master/tutorials/Multitask.ipynb**](https://github.com/HazyResearch/metal/blob/master/tutorials/Multitask.ipynb)**). That interface is not currently available in Snorkel v0.9 (no fundamental blockers; just hasn't been ported over yet).**
   2. **There are, however, still a number of ways to model such situations. One way is to treat each node in the hierarchy as a separate task and combine their probabilities post-hoc (e.g., P(credit-request) = P(billing) \* P(credit-request | billing)). Another is to treat them as separate tasks and use a multi-task end model to implicitly learn how the predictions of some tasks should affect the predictions of others (e.g., the end model we use in the AAAI paper). A third option is to create a single task with all the leaf categories and modify the output space of the LFs you were considering for the higher nodes (the deeper your hierarchy is or the larger the number of classes, the less apppealing this is w/r/t to approaches 1 and 2).**
3. [**mechanical turk calculator**](https://morninj.github.io/mechanical-turk-cost-calculator/)
4. [**Mturk alternatives**](https://moneypantry.com/amazon-mechanical-turk-crowdsourcing-alternatives/)
   1. [**Workforce / onespace**](https://www.crowdsource.com/workforce/)
   2. [**Jobby**](https://www.jobboy.com/)
   3. [**Shorttask**](http://www.shorttask.com/)
   4. [**Samasource**](https://www.samasource.org/team)
   5. **Figure 8 -** [**pricing**](https://siftery.com/crowdflower/pricing) **-** [**definite guide**](https://www.earnonlineguys.com/figure-eight-tasks-guide/)
5. [**Brat nlp annotation tool**](http://brat.nlplab.org/?fbclid=IwAR1bDCM3j3nEQb3Hrf9dGCwyRvDVMBXoob4WtVLCWAMBgPraZmkSi123IrI)
6. [**Prodigy by spacy**](https://prodi.gy/)**,**&#x20;
   1. [**seed-small sample, many sample tutorial on youtube by ines**](https://www.youtube.com/watch?v=5di0KlKl0fE)
   2. [**How to use prodigy, tutorial on medium plus notebook code inside**](https://medium.com/@david.campion/text-classification-be-lazy-use-prodigy-b0f9d00e9495)
7. [**Doccano**](https://github.com/chakki-works/doccano) **- prodigy open source alternative butwith users management & statistics out of the box**
8. **Medium** [**Lighttag - has some cool annotation metrics\tests**](https://medium.com/@TalPerry/announcing-lighttag-the-easy-way-to-annotate-text-afb7493a49b8)
9. [Loopr](https://loopr.ai/products/labeling-platform).ai - An AI powered semi-automated and automated annotation process for high quality data.object detection, analytics, nlp, active learning.
10. **Medium** [**Assessing annotator disagreement**](https://towardsdatascience.com/assessing-annotator-disagreements-in-python-to-build-a-robust-dataset-for-machine-learning-16c74b49f043)
11. [**A great python package for measuring disagreement on GH**](https://github.com/o-P-o/disagree)
12. [**Reliability is key, and not just mechanical turk**](https://www.youtube.com/watch?v=ktZLuXPXPEI)
13. [**7 myths about annotation**](https://www.aaai.org/ojs/index.php/aimagazine/article/viewFile/2564/2468)
14. [**Annotating twitter sentiment using humans, 3 classes, 55% accuracy using SVMs.**](http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0155036) **they talk about inter agreement etc. and their DS is** [**partially publicly available**](https://www.clarin.si/repository/xmlui/handle/11356/1054)**.**
15. [**Exploiting disagreement** ](https://s3.amazonaws.com/academia.edu.documents/8026932/10.1.1.2.8084.pdf?AWSAccessKeyId=AKIAIWOWYYGZ2Y53UL3A\&Expires=1534444363\&Signature=3dHHw3EmAjPXFxwutVbtsZWEIzw%3D\&response-content-disposition=inline%3B%20filename%3DExploiting\_agreement\_and\_disagreement\_of.pdf)
16. [**Vader annotation**](http://comp.social.gatech.edu/papers/icwsm14.vader.hutto.pdf)
    1. **They must pass an english exam**
    2. **They get control questions to establish their reliability**
    3. **They get a few sentences over and over again to establish inter disagreement**
    4. **Two or more people get a overlapping sentences to establish disagreement**
    5. **5 judges for each sentence (makes 4 useless)**
    6. **They dont know each other**
    7. **Simple rules to follow**
    8. **Random selection of sentences**
    9. **Even classes**
    10. **No experts**
    11. **Measuring reliability kappa/the other kappa.**
17. [**Label studio**\
    ](https://labelstud.io/)![](https://lh3.googleusercontent.com/X2kRKqlPnkMZyspKgiJYHR5vyE2NnRfkYJZMxBs\_rfFeGaMl0L07hqCO8VRGnTV\_E9qhroCDYLIlQ1e78EgraeE6wwPE3WJDkzVmR6kQTgv4I-npCh3UkKnuBE\_C1Lo9dQ3QxcEg)

**Ideas:**&#x20;

1. **Active learning for a group (or single) of annotators, we have to wait for all annotations to finish each big batch in order to retrain the model.**
2. **Annotate a small group, automatic labelling using knn**
3. **Find a nearest neighbor for out optimal set of keywords per “category,**&#x20;
4. **For a group of keywords, find their knn neighbors in w2v-space, alternatively find k clusters in w2v space that has those keywords. For a new word/mean sentence vector in the ‘category’ find the minimal distance to the new cluster (either one of approaches) and this is new annotation.**

## Myths

1. [**7 myths of annotation**](https://www.aaai.org/ojs/index.php/aimagazine/article/viewFile/2564/2468)
   1. **Myth One: One Truth Most data collection efforts assume that there is one correct interpretation for every input example.**&#x20;
   2. **Myth Two: Disagreement Is Bad To increase the quality of annotation data, disagreement among the annotators should be avoided or reduced.**&#x20;
   3. **Myth Three: Detailed Guidelines Help When specific cases continuously cause disagreement, more instructions are added to limit interpretations.**&#x20;
   4. **Myth Four: One Is Enough Most annotated examples are evaluated by one person.**&#x20;
   5. **Myth Five: Experts Are Better Human annotators with domain knowledge provide better annotated data.**&#x20;
   6. **Myth Six: All Examples Are Created Equal The mathematics of using ground truth treats every example the same; either you match the correct result or not.**&#x20;
   7. **Myth Seven: Once Done, Forever Valid Once human annotated data is collected for a task, it is used over and over with no update. New annotated data is not aligned with previous data.**

## Crowd Sourcing

#### [**Crowd Sourcing** ](https://www.youtube.com/watch?v=ktZLuXPXPEI)

![](https://lh3.googleusercontent.com/CpbWZ2kVN\_c84uZnRgfBAxTVBxBQArQDbMhZj12n8n8zRZIB-1FwOyEx7Yn2P\_sZ6qclUnfimvkKUsmSTXC3eFFIM49oHGhwMctXkPZUGFGXTAO3LlhZJv7Gw1TGr\_pDjRsIiCSc)

![](https://lh3.googleusercontent.com/Xo5pBUmwOyqKqnZJvJc2kyjzPZYiZLY4acF\_oK6Su6WsYCVuJygvdgDgjLRhPWdbcVsxO8qs6C1pHuH0ZWVVZ5-Z-F1fRlojJ-MYcaMUx56tE0Z2OxzJ02ieMNEhIAHiLnMwZKPi)

![](https://lh3.googleusercontent.com/Hx9UzYlcDRUIpf9Pt-f4xI9M8EwPapcEcwwXcmKry8VC0OzyI4kbrp7h4E7nOXeMMdR1wdd\_Dwa54THEBpvcwZbjmWHBQQEAzBGtB8RyF40xbx6AV4L9BErGcbRFM-AMHuN7GTq\_)![](https://lh4.googleusercontent.com/1VEsT95na9TLGXNUBwAGMKOdTJDI4cJ5rCirq\_WYhCne-xBmDTjcpJ4Qmoyh7OHW5ilBCnjpJ4U1opy1TK7v6-i4AmsqAbUm42YGg1Ee\_90HFblseEd1K6PyfTA7NTow6B6WsZtE)\


![](https://lh3.googleusercontent.com/m1MAdhxW1T3\_-s0i6PHH-xCBfBpQLCqtVpL-WfUvVyR3A\_NT274te37PLRYjfCELOS0YB4zUNCAswBcG0fY4fMDlWh-hmz9kMCVfiM5xqyyZDc5NEfkIYt57O105II8kU5ccVnIG)

![](https://lh4.googleusercontent.com/s8A8VcNA22GZ5FtBnQaAJvxyJmw7jgEIp4LFw28z5OxoZwAfuoShsSSDSRa7Loqud-caBFY9lQK1xhbUrlwyhox2btt7hLMfbb\_L59BzFGxxgX35p-5bJdInEIkuWf6vBmmioaWe)

* **Conclusions:**&#x20;
  * **Experts are the same as a crowd**
  * **Costs a lot less \$$$.**

## Disagreement

## **Inter agreement**

**\*\*\*** [**The best tutorial on agreements, cohen, david, kappa, krip etc.**](https://dkpro.github.io/dkpro-statistics/inter-rater-agreement-tutorial.pdf)

1. **Cohens kappa (two people)**

&#x20;**but you can use it to map a group by calculating agreement for each pair**

1. [**Why cohens kappa should be avoided as a performance measure in classification**](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0222916)
2. [**Why it should be used as a measure of classification**](https://thedatascientist.com/performance-measures-cohens-kappa-statistic/)
3. [**Kappa in plain english**](https://stats.stackexchange.com/questions/82162/cohens-kappa-in-plain-english)
4. [**Multilabel using kappa**](https://stackoverflow.com/questions/52272901/multi-label-annotator-agreement-with-cohen-kappa)
5. [**Kappa and the relation with accuracy**](https://gis.stackexchange.com/questions/110188/how-are-kappa-and-overall-accuracy-related-with-respect-to-thematic-raster-data) **(redundant, % above chance, should not be used due to other reasons researched here)**

**The Kappa statistic varies from 0 to 1, where.**

* **0 = agreement equivalent to chance.**
* **0.1 – 0.20 = slight agreement.**
* **0.21 – 0.40 = fair agreement.**
* **0.41 – 0.60 = moderate agreement.**
* **0.61 – 0.80 = substantial agreement.**
* **0.81 – 0.99 = near perfect agreement**
* **1 = perfect agreement.**

1. **Fleiss’ kappa, from 3 people and above.**

**Kappa ranges from 0 to 1, where:**

* **0 is no agreement (or agreement that you would expect to find by chance),**
* **1 is perfect agreement.**
* **Fleiss’s Kappa is an extension of Cohen’s kappa for three raters or more. In addition, the assumption with Cohen’s kappa is that your raters are deliberately chosen and fixed. With Fleiss’ kappa, the assumption is that your raters were chosen at random from a larger population.**
* [**Kendall’s Tau**](https://www.statisticshowto.datasciencecentral.com/kendalls-tau/) **is used when you have ranked data, like two people ordering 10 candidates from most preferred to least preferred.**
* **Krippendorff’s alpha is useful when you have multiple raters and multiple possible ratings.**

1. **Krippendorfs alpha**&#x20;

* [**Ignores missing data entirely**](https://deepsense.ai/multilevel-classification-cohen-kappa-and-krippendorff-alpha/)**.**
* **Can handle various sample sizes, categories, and numbers of raters.**
* **Applies to any** [**measurement level**](https://www.statisticshowto.datasciencecentral.com/scales-of-measurement/) **(i.e. (**[**nominal, ordinal, interval, ratio**](https://www.statisticshowto.datasciencecentral.com/nominal-ordinal-interval-ratio/)**).**
* **Values range from 0 to 1, where 0 is perfect disagreement and 1 is perfect agreement. Krippendorff suggests: “\[I]t is customary to require α ≥ .800. Where tentative conclusions are still acceptable, α ≥ .667 is the lowest conceivable limit (2004, p. 241).”**
* [**Supposedly multi label**](https://stackoverflow.com/questions/57256287/calculate-kappa-score-for-multi-label-image-classifcation)

1. **MACE - the new kid on the block. -**

&#x20;**learns in an unsupervised fashion to**&#x20;

1. **a) identify which annotators are trustworthy and**
2. &#x20;**b) predict the correct underlying labels. We match performance of more complex state-of-the-art systems and perform well even under adversarial conditions**
3. [**MACE**](https://www.isi.edu/publications/licensed-sw/mace/) **does exactly that. It tries to find out which annotators are more trustworthy and upweighs their answers.**
4. [**Git**](https://github.com/dirkhovy/MACE) **-**

**When evaluating redundant annotatio**

**ns (like those from Amazon's MechanicalTurk), we usually want to**

1. **aggregate annotations to recover the most likely answer**
2. **find out which annotators are trustworthy**
3. **evaluate item and task difficulty**

**MACE solves all of these problems, by learning competence estimates for each annotators and computing the most likely answer based on those competences.**

1.

**Calculating agreement**

1. **Compare against researcher-ground-truth**
2. **Self-agreement**
3. **Inter-agreement**
   1. [**Medium**](https://towardsdatascience.com/inter-rater-agreement-kappas-69cd8b91ff75)
   2. [**Kappa**](https://stats.stackexchange.com/questions/82162/cohens-kappa-in-plain-english) **cohen**
   3. [**Multi annotator with kappa (which isnt), is this okay?**](https://stackoverflow.com/questions/52272901/multi-label-annotator-agreement-with-cohen-kappa)
   4. **Github computer Fleiss Kappa** [**1**](https://gist.github.com/skylander86/65c442356377367e27e79ef1fed4adee)
   5. [**Fleiss Kappa Example**](https://www.wikiwand.com/en/Fleiss'\_kappa#/Worked\_example)
   6. [**GWET AC1**](https://stats.stackexchange.com/questions/235929/fleiss-kappa-alternative-for-ranking)**,** [**paper**](https://s3.amazonaws.com/sitesusa/wp-content/uploads/sites/242/2014/05/J4\_Xie\_2013FCSM.pdf)**: as an alternative to kappa, and why**
   7. [**Website, krippensorf vs fleiss calculator**](https://nlp-ml.io/jg/software/ira/)

## **Troubling shooting agreement metrics**

1. **Imbalance data sets, i.e., why my** [**Why is reliability so low when percentage of agreement is high?**](https://www.researchgate.net/post/Why\_is\_reliability\_so\_low\_when\_percentage\_of\_agreement\_is\_high)
2. [**Interpretation of kappa values**](https://towardsdatascience.com/interpretation-of-kappa-values-2acd1ca7b18f)
3. [**Interpreting agreement**](http://web2.cs.columbia.edu/\~julia/courses/CS6998/Interrater\_agreement.Kappa\_statistic.pdf)**, Accuracy precision kappa**

## **Machine Vision annotation**

1. [**CVAT**](https://venturebeat.com/2019/03/05/intel-open-sources-cvat-a-toolkit-for-data-labeling/)
