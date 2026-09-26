# Annotation & Disagreement

Labels are work product, and annotators disagree, so the page has to face myths, disagreement, and agreement metrics before it trusts a tool shelf.
The notes move from myths through disagreement, inter-agreement, troubleshooting, crowdsourcing, and vision annotation, then the tools that run that loop.
The same notes are in [Active Learning](../problem-framing/active-learning.md) and [Label Noise](../problem-framing/label-algorithms.md#label-noise).

## Myths

Most annotation projects start from beliefs about what a good label is, and the 7 myths of annotation, from "Truth Is a Lie: Crowd Truth and the Seven Myths of Human Annotation" (its address is with the crowd-sourcing sources further down), argue that human annotation rests on an antiquated ideal of a single correct truth. The seven myths are:

1. Myth One: One Truth. Most data collection efforts assume that there is one correct interpretation for every input example.
2. Myth Two: Disagreement Is Bad. To increase the quality of annotation data, disagreement among the annotators should be avoided or reduced.
3. Myth Three: Detailed Guidelines Help. When specific cases continuously cause disagreement, more instructions are added to limit interpretations.
4. Myth Four: One Is Enough. Most annotated examples are evaluated by one person.
5. Myth Five: Experts Are Better. Human annotators with domain knowledge provide better annotated data.
6. Myth Six: All Examples Are Created Equal. The mathematics of using ground truth treats every example the same; either you match the correct result or not.
7. Myth Seven: Once Done, Forever Valid. Once human annotated data is collected for a task, it is used over and over with no update. New annotated data is not aligned with previous data.

## Disagreement

If disagreement is not simply bad, as the second myth claims, then it is something to measure rather than remove, and the rest of the page is how to measure it.

## Inter agreement

Measuring disagreement means choosing an agreement metric, and the choice depends on how many raters there are and what kind of labels they give. The same notes are in [Ground Truth](../language-ai/sentiment-analysis.md#ground-truth).

The DKPro Agreement 2.0 slides from the Ubiquitous Knowledge Processing (UKP) Lab at Technische Universität Darmstadt are the place to start, marked \*\*\* in the original notes: [The best tutorial on agreements, cohen, david, kappa, krip etc.](https://dkpro.github.io/dkpro-statistics/inter-rater-agreement-tutorial.pdf)

The first metric is Cohens kappa, for two people, but you can use it to map a group by calculating agreement for each pair. Whether it belongs in classification evaluation is disputed. [Why cohens kappa should be avoided as a performance measure in classification](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0222916) shows that Cohen's Kappa and the Matthews Correlation Coefficient are correlated in most situations but differ in unbalanced ones, where Kappa can give a worse classifier a higher score. The other side is [Why it should be used as a measure of classification](https://thedatascientist.com/performance-measures-cohens-kappa-statistic/), which argues that Cohen's kappa statistic is a very good measure that can handle both multi-class and imbalanced class problems very well. For the intuition, [Kappa in plain english](https://stats.stackexchange.com/questions/82162/cohens-kappa-in-plain-english) is the question from a reader of a data mining book asking what kappa actually tells you about a classifier. [Multilabel using kappa](https://stackoverflow.com/questions/52272901/multi-label-annotator-agreement-with-cohen-kappa) is the case of two annotators labelling documents with multiple labels each and trying sklearn's cohen_kappa_score. [Kappa and the relation with accuracy](https://gis.stackexchange.com/questions/110188/how-are-kappa-and-overall-accuracy-related-with-respect-to-thematic-raster-data) is marked (redundant, % above chance, should not be used due to other reasons researched here).

The Kappa statistic varies from 0 to 1, where:

- 0 = agreement equivalent to chance.
- 0.1 – 0.20 = slight agreement.
- 0.21 – 0.40 = fair agreement.
- 0.41 – 0.60 = moderate agreement.
- 0.61 – 0.80 = substantial agreement.
- 0.81 – 0.99 = near perfect agreement
- 1 = perfect agreement.

With three people or more, Cohen's kappa gives way to Fleiss’ kappa, from 3 people and above. Kappa ranges from 0 to 1, where 0 is no agreement (or agreement that you would expect to find by chance) and 1 is perfect agreement. Fleiss’s Kappa is an extension of Cohen’s kappa for three raters or more. In addition, the assumption with Cohen’s kappa is that your raters are deliberately chosen and fixed. With Fleiss’ kappa, the assumption is that your raters were chosen at random from a larger population. Two other metrics cover the remaining cases: [Kendall’s Tau](https://www.statisticshowto.datasciencecentral.com/kendalls-tau/) is used when you have ranked data, like two people ordering 10 candidates from most preferred to least preferred, and Krippendorff’s alpha is useful when you have multiple raters and multiple possible ratings.

Krippendorfs alpha is the most general of them. The deepsense.ai post on multilevel classification, Cohen kappa and Krippendorff alpha, written while building a classifier that predicts cancer type from genetic signatures in The Genome Cancer Atlas, notes that it [Ignores missing data entirely](https://deepsense.ai/multilevel-classification-cohen-kappa-and-krippendorff-alpha/). It can handle various sample sizes, categories, and numbers of raters, and it applies to any [measurement level](https://www.statisticshowto.datasciencecentral.com/scales-of-measurement/), i.e. [nominal, ordinal, interval, ratio](https://www.statisticshowto.datasciencecentral.com/nominal-ordinal-interval-ratio/). Values range from 0 to 1, where 0 is perfect disagreement and 1 is perfect agreement. Krippendorff suggests: “\[I]t is customary to require α ≥ .800. Where tentative conclusions are still acceptable, α ≥ .667 is the lowest conceivable limit (2004, p. 241).” It is [Supposedly multi label](https://stackoverflow.com/questions/57256287/calculate-kappa-score-for-multi-label-image-classifcation), the question about computing kappa for multi-label image classification when sklearn's cohen_kappa_score raises "multilabel-indicator is not supported".

All of these metrics score the raters as a group; MACE - the new kid on the block - scores each one. It learns in an unsupervised fashion to a) identify which annotators are trustworthy and b) predict the correct underlying labels. We match performance of more complex state-of-the-art systems and perform well even under adversarial conditions. MACE does exactly that: it tries to find out which annotators are more trustworthy and upweighs their answers. The code is the Multi-Annotator Competence Estimation tool on [Git](https://github.com/dirkhovy/MACE).

When evaluating redundant annotations (like those from Amazon's MechanicalTurk), we usually want to

1. aggregate annotations to recover the most likely answer
2. find out which annotators are trustworthy
3. evaluate item and task difficulty

MACE solves all of these problems, by learning competence estimates for each annotators and computing the most likely answer based on those competences.

Calculating agreement then comes in three forms: compare against researcher-ground-truth, self-agreement, and inter-agreement. For inter-agreement, Amir Ziai's Inter-rater agreement Kappas on [Medium](https://medium.com/data-science/inter-rater-agreement-kappas-69cd8b91ff75) defines inter-rater reliability as the degree of agreement among raters, a score of how much homogeneity or consensus there is in the ratings. The [Kappa](https://stats.stackexchange.com/questions/82162/cohens-kappa-in-plain-english) plain-English thread above applies here too, as does [Multi annotator with kappa (which isnt), is this okay?](https://stackoverflow.com/questions/52272901/multi-label-annotator-agreement-with-cohen-kappa). For three raters or more, the Github gist to compute Fleiss' kappa using numpy is [1](https://gist.github.com/skylander86/65c442356377367e27e79ef1fed4adee), and [Fleiss Kappa Example](https://www.wikiwand.com/en/Fleiss'_kappa#/Worked_example) is the worked example. When ratings are scores rather than categories, [GWET AC1](https://stats.stackexchange.com/questions/235929/fleiss-kappa-alternative-for-ranking) is the answer to a question asking for a weighted alternative to Fleiss' kappa for scores from 1 to 6, and the [paper](https://s3.amazonaws.com/sitesusa/wp-content/uploads/sites/242/2014/05/J4_Xie_2013FCSM.pdf) behind it splits agreement measures into a classical descriptive approach and a modeling approach. A Website, krippensorf vs fleiss calculator, used to sit here and is kept at the end of the page.

## Troubling shooting agreement metrics

Metrics can still disagree with intuition, most often on imbalance data sets, i.e., why my agreement looks high while the metric is low. [Why is reliability so low when percentage of agreement is high?](https://www.researchgate.net/post/Why_is_reliability_so_low_when_percentage_of_agreement_is_high) is that exact question. [Interpretation of kappa values](https://medium.com/data-science/interpretation-of-kappa-values-2acd1ca7b18f) explains that the kappa statistic is frequently used to test interrater reliability, which represents the extent to which the collected data are correct representations of the variables measured. The notes on Interpreting agreement, Accuracy precision kappa used to sit here and are kept at the end of the page.

## Crowd Sourcing

Once agreement can be measured, the practical question is who should annotate: experts or a crowd.

#### Crowd Sourcing 

Professor Kenneth Benoit's Alan Turing Institute talk, "Ground truth? The uses and abuses of human annotation in text analysis", is the video for this question.

{% embed url="https://www.youtube.com/watch?v=ktZLuXPXPEI" %}

{% endembed %}

The figures below are notes from that [Crowd Sourcing ](https://www.youtube.com/watch?v=ktZLuXPXPEI) talk.

<figure><img src="../.gitbook/assets/gimg-3613f234815e.png" alt=""><figcaption><p>Crowdsourcing annotation notes.</p><p>Credit: <a href="https://lh3.googleusercontent.com/CpbWZ2kVN_c84uZnRgfBAxTVBxBQArQDbMhZj12n8n8zRZIB-1FwOyEx7Yn2P_sZ6qclUnfimvkKUsmSTXC3eFFIM49oHGhwMctXkPZUGFGXTAO3LlhZJv7Gw1TGr_pDjRsIiCSc">copied from the original hosted image</a>.</p></figcaption></figure>

<figure><img src="../.gitbook/assets/gimg-94052c60284f.png" alt=""><figcaption><p>Crowdsourcing annotation notes.</p><p>Credit: <a href="https://lh3.googleusercontent.com/Xo5pBUmwOyqKqnZJvJc2kyjzPZYiZLY4acF_oK6Su6WsYCVuJygvdgDgjLRhPWdbcVsxO8qs6C1pHuH0ZWVVZ5-Z-F1fRlojJ-MYcaMUx56tE0Z2OxzJ02ieMNEhIAHiLnMwZKPi">copied from the original hosted image</a>.</p></figcaption></figure>

<figure><img src="../.gitbook/assets/gimg-10136540d579.png" alt=""><figcaption><p>Crowdsourcing annotation notes.</p><p>Credit: <a href="https://lh3.googleusercontent.com/Hx9UzYlcDRUIpf9Pt-f4xI9M8EwPapcEcwwXcmKry8VC0OzyI4kbrp7h4E7nOXeMMdR1wdd_Dwa54THEBpvcwZbjmWHBQQEAzBGtB8RyF40xbx6AV4L9BErGcbRFM-AMHuN7GTq_">copied from the original hosted image</a>.</p></figcaption></figure>
<figure><img src="../.gitbook/assets/gimg-b0c33763ac4e.png" alt=""><figcaption><p>Crowdsourcing annotation notes.</p><p>Credit: <a href="https://lh4.googleusercontent.com/1VEsT95na9TLGXNUBwAGMKOdTJDI4cJ5rCirq_WYhCne-xBmDTjcpJ4Qmoyh7OHW5ilBCnjpJ4U1opy1TK7v6-i4AmsqAbUm42YGg1Ee_90HFblseEd1K6PyfTA7NTow6B6WsZtE">copied from the original hosted image</a>.</p></figcaption></figure>

<figure><img src="../.gitbook/assets/gimg-6ebd6808923c.png" alt=""><figcaption><p>Crowdsourcing annotation notes.</p><p>Credit: <a href="https://lh3.googleusercontent.com/m1MAdhxW1T3_-s0i6PHH-xCBfBpQLCqtVpL-WfUvVyR3A_NT274te37PLRYjfCELOS0YB4zUNCAswBcG0fY4fMDlWh-hmz9kMCVfiM5xqyyZDc5NEfkIYt57O105II8kU5ccVnIG">copied from the original hosted image</a>.</p></figcaption></figure>

<figure><img src="../.gitbook/assets/gimg-bd667d6736a1.png" alt=""><figcaption><p>Crowdsourcing annotation notes.</p><p>Credit: <a href="https://lh4.googleusercontent.com/s8A8VcNA22GZ5FtBnQaAJvxyJmw7jgEIp4LFw28z5OxoZwAfuoShsSSDSRa7Loqud-caBFY9lQK1xhbUrlwyhox2btt7hLMfbb_L59BzFGxxgX35p-5bJdInEIkuWf6vBmmioaWe">copied from the original hosted image</a>.</p></figcaption></figure>

The conclusions answer the fifth myth directly: experts are the same as a crowd, and the crowd costs a lot less \$$$.

## Machine Vision annotation

Text is not the only thing a crowd labels; images need their own tooling. The same notes are in [Deep Neural Machine Vision](../deep-learning/deep-neural-machine-vision.md).

For vision, the tool to look at is [CVAT](https://venturebeat.com/2019/03/05/intel-open-sources-cvat-a-toolkit-for-data-labeling/).

The full addresses for the crowd-sourcing and weak-supervision sources used on this page are:

- Snorkel metal, from the weak-supervision tools below: [https://jdunnmon.github.io/metal_deem.pdf](https://jdunnmon.github.io/metal_deem.pdf)
- Mturk alternatives, Lauren Bennett's list of the best MTurk alternatives, some of which pay more: [https://moneypantry.com/amazon-mechanical-turk-crowdsourcing-alternatives/](https://moneypantry.com/amazon-mechanical-turk-crowdsourcing-alternatives/)
- Jobby, now a site for job opportunities, career advice, and hiring tips: [https://www.jobboy.com/](https://www.jobboy.com/)
- Shorttask, which connects online job seekers with task providers: [http://www.shorttask.com/](http://www.shorttask.com/)
- Samasource, now Sama, a team of experts delivering data solutions: [https://www.samasource.org/team](https://www.samasource.org/team)
- The definite guide to Appen (Figure Eight) micro tasks: [https://www.earnonlineguys.com/figure-eight-tasks-guide/](https://www.earnonlineguys.com/figure-eight-tasks-guide/)
- 7 myths about annotation, the paper behind the Myths section: [https://www.aaai.org/ojs/index.php/aimagazine/article/viewFile/2564/2468](https://www.aaai.org/ojs/index.php/aimagazine/article/viewFile/2564/2468)

## Tools

With the people and the metrics settled, the tools are what run the annotation loop day to day, starting with ways to need fewer human labels.

[Snorkel](https://www.snorkel.org/use-cases/) - using weak supervision to create less noisy labelled datasets. Its [Git](https://github.com/snorkel-team/snorkel) repo is the system for quickly generating training data with weak supervision; a Medium introduction to it used to sit here and is kept at the end of the page. Snorkel metal extends that to weak supervision for multi-task learning; the Conversation about it is kept at the end of the page, and the [git](https://github.com/HazyResearch/metal/blob/master/tutorials/Multitask.ipynb) notebook is the Snorkel MeTaL multi-task tutorial. The Snorkel team's answer on hierarchical labels explains where that work stands:

"Yes, the Snorkel project has included work before on hierarchical labeling scenarios. The main papers detailing our results include the DEEM workshop paper you referenced ([https://dl.acm.org/doi/abs/10.1145/3209889.3209898](https://dl.acm.org/doi/abs/10.1145/3209889.3209898)) and the more complete paper presented at AAAI ([https://arxiv.org/abs/1810.02840](https://arxiv.org/abs/1810.02840)). Before the Snorkel and Snorkel MeTaL projects were merged in Snorkel v0.9, the Snorkel MeTaL project included an interface for explicitly specifying hierarchies between tasks which was utilized by the label model and could be used to automatically compile a multi-task end model as well (demo here: [https://github.com/HazyResearch/metal/blob/master/tutorials/Multitask.ipynb](https://github.com/HazyResearch/metal/blob/master/tutorials/Multitask.ipynb)). That interface is not currently available in Snorkel v0.9 (no fundamental blockers; just hasn't been ported over yet).

There are, however, still a number of ways to model such situations. One way is to treat each node in the hierarchy as a separate task and combine their probabilities post-hoc (e.g., P(credit-request) = P(billing) \* P(credit-request | billing)). Another is to treat them as separate tasks and use a multi-task end model to implicitly learn how the predictions of some tasks should affect the predictions of others (e.g., the end model we use in the AAAI paper). A third option is to create a single task with all the leaf categories and modify the output space of the LFs you were considering for the higher nodes (the deeper your hierarchy is or the larger the number of classes, the less apppealing this is w/r/t to approaches 1 and 2)."

When the labels do come from people, the budget comes first, and the Mechanical Turk Cost Calculator is the [mechanical turk calculator](https://morninj.github.io/mechanical-turk-cost-calculator/). The Mturk alternatives are Workforce / onespace, Jobby, Shorttask, Samasource, and Figure 8, with its pricing and the definite guide; their addresses are in the list above.

For doing the annotation itself, [Brat nlp annotation tool](http://brat.nlplab.org/) is a web-based annotation tool for textual annotation. [Prodigy by spacy](https://prodi.gy/) is a downloadable annotation tool for LLMs, NLP and computer vision tasks such as named entity recognition, text classification, object detection, image segmentation, and evaluation. Explosion's video TRAINING AN INSULTS CLASSIFIER with Prodigy in ~1 hour is the [seed-small sample, many sample tutorial on youtube by ines](https://www.youtube.com/watch?v=5di0KlKl0fE), and David Campion's "Text Classification: Be lazy, use Prodigy !" is [How to use prodigy, tutorial on medium plus notebook code inside](https://medium.com/@david.campion/text-classification-be-lazy-use-prodigy-b0f9d00e9495). [Doccano](https://github.com/chakki-works/doccano) - prodigy open source alternative butwith users management & statistics out of the box. The Medium announcement of [Lighttag - has some cool annotation metrics\tests](https://medium.com/@TalPerry/announcing-lighttag-the-easy-way-to-annotate-text-afb7493a49b8). Loopr.ai - An AI powered semi-automated and automated annotation process for high quality data.object detection, analytics, nlp, active learning; its address is kept at the end of the page.

Once annotations exist, the disagreement between them has to be read. Oliver Price's Medium post [Assessing annotator disagreement](https://medium.com/data-science/assessing-annotator-disagreements-in-python-to-build-a-robust-dataset-for-machine-learning-16c74b49f043) starts from Matthew Honnibal's 2018 PyData Berlin point that a solid annotation schema and attentive annotators are building blocks of a successful model. [A great python package for measuring disagreement on GH](https://github.com/o-P-o/disagree) is the package to visualise, evaluate, and manage annotated data. The Kenneth Benoit talk from the crowd-sourcing section is the reminder that [Reliability is key, and not just mechanical turk](https://www.youtube.com/watch?v=ktZLuXPXPEI), and the 7 myths about annotation above are the same warning in paper form.

Two annotation studies show the whole loop in practice. Multilingual Twitter Sentiment Classification: The Role of Human Annotators is [Annotating twitter sentiment using humans, 3 classes, 55% accuracy using SVMs.](http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0155036) It finds that model quality depends much more on the quality and size of training data than on the type of model; they talk about inter agreement etc. and their DS is [partially publicly available](https://www.clarin.si/repository/xmlui/handle/11356/1054) as Twitter sentiment for 15 European languages. [Exploiting disagreement ](https://web.eecs.umich.edu/~mihalcea/papers/chklovski.ranlp03.pdf) turns that disagreement into something to use rather than remove. [Vader annotation](https://web.archive.org/web/20160327132241/http://comp.social.gatech.edu/papers/icwsm14.vader.hutto.pdf) is the annotation protocol behind VADER, and its rules read as a checklist:

1. They must pass an english exam
2. They get control questions to establish their reliability
3. They get a few sentences over and over again to establish inter disagreement
4. Two or more people get a overlapping sentences to establish disagreement
5. 5 judges for each sentence (makes 4 useless)
6. They dont know each other
7. Simple rules to follow
8. Random selection of sentences
9. Even classes
10. No experts
11. Measuring reliability kappa/the other kappa.

For running all of that on one platform, [Label studio](https://labelstud.io/) is a multi-modal data labeling and annotation platform for agent traces, LLM evals, RLHF, computer vision, document AI, NLP, audio transcription, and more.

<figure><img src="../.gitbook/assets/gimg-31c51da0af26.png" alt=""><figcaption><p>Label Studio.</p><p>Credit: <a href="https://lh3.googleusercontent.com/X2kRKqlPnkMZyspKgiJYHR5vyE2NnRfkYJZMxBs_rfFeGaMl0L07hqCO8VRGnTV_E9qhroCDYLIlQ1e78EgraeE6wwPE3WJDkzVmR6kQTgv4I-npCh3UkKnuBE_C1Lo9dQ3QxcEg">copied from the original hosted image</a>.</p></figcaption></figure>

The tools also suggest ways to label less. Ideas:

1. Active learning for a group (or single) of annotators, we have to wait for all annotations to finish each big batch in order to retrain the model.
2. Annotate a small group, automatic labelling using knn
3. Find a nearest neighbor for out optimal set of keywords per “category,
4. For a group of keywords, find their knn neighbors in w2v-space, alternatively find k clusters in w2v space that has those keywords. For a new word/mean sentence vector in the ‘category’ find the minimal distance to the new cluster (either one of approaches) and this is new annotation.

## Deprecated links

{% hint style="warning" %}
These links and images no longer work. The original wording is kept here. A same-resource copy, when one was checked, is used above.
{% endhint %}

- Medium This address no longer opens: https://towardsdatascience.com/introducing-snorkel-27e4b0e6ecff
- Conversation This address no longer opens: https://spectrum.chat/snorkel/help/hierarchical-labelling-example~aa4d8617-d287-43a6-865e-7c9034888363
- Workforce / onespace This address no longer opens: https://www.crowdsource.com/workforce/
- pricing This address no longer opens: https://siftery.com/crowdflower/pricing
- Loopr This address no longer opens: https://loopr.ai/products/labeling-platform
- Assessing annotator disagreement This address no longer opens: https://towardsdatascience.com/assessing-annotator-disagreements-in-python-to-build-a-robust-dataset-for-machine-learning-16c74b49f043
- Exploiting disagreement This address no longer opens: https://s3.amazonaws.com/academia.edu.documents/8026932/10.1.1.2.8084.pdf?AWSAccessKeyId=AKIAIWOWYYGZ2Y53UL3A&Expires=1534444363&Signature=3dHHw3EmAjPXFxwutVbtsZWEIzw%3D&response-content-disposition=inline%3B%20filename%3DExploiting_agreement_and_disagreement_of.pdf
- Vader annotation This address no longer opens: http://comp.social.gatech.edu/papers/icwsm14.vader.hutto.pdf
- Medium This address no longer opens: https://towardsdatascience.com/inter-rater-agreement-kappas-69cd8b91ff75
- Website, krippensorf vs fleiss calculator This address no longer opens: https://nlp-ml.io/jg/software/ira/
- Interpretation of kappa values This address no longer opens: https://towardsdatascience.com/interpretation-of-kappa-values-2acd1ca7b18f
- Interpreting agreement This address no longer opens: http://web2.cs.columbia.edu/~julia/courses/CS6998/Interrater_agreement.Kappa_statistic.pdf
- MACE. This address no longer opens: https://www.isi.edu/publications/licensed-sw/mace/
- Medium. This address no longer opens: https://medium.com/data-science/introducing-snorkel-27e4b0e6ecff
