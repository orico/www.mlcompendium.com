# Sentiment Analysis

This page collects sentiment datasets, tools (VADER, TextBlob), and annotator-agreement notes.

## Databases

This section lists sentiment lexicons and review or tweet datasets.

1. [Sentiment databases](https://medium.com/@datamonsters/sentiment-analysis-tools-overview-part-1-positive-and-negative-words-databases-ae35431a470c)
2. Movie reviews: [IMDB reviews dataset on Kaggle](https://www.kaggle.com/c/word2vec-nlp-tutorial/data)
3. Sentiwordnet – mapping wordnet senses to a polarity model: [SentiWordnet Site](https://github.com/aesuli/SentiWordNet)
4. [Twitter airline sentiment on Kaggle](https://www.kaggle.com/crowdflower/twitter-airline-sentiment)
5. [First GOP Debate Twitter Sentiment](https://www.kaggle.com/crowdflower/first-gop-debate-twitter-sentiment)
6. [Amazon fine foods reviews](https://www.kaggle.com/snap/amazon-fine-food-reviews)

## Tools

This section points at sentiment analyzers, VADER, TextBlob, and related papers.

1. ** Many [Sentiment tools,](https://medium.com/@datamonsters/sentiment-analysis-tools-overview-part-2-7f3a75c262a3)
2. [NTLK sentiment analyzer](http://www.nltk.org/api/nltk.sentiment.html)
3. Vader (NTLK, standalone):
   1. [Vader/Sentiwordnet/etc python code examples - possibly good for ensembles](https://nlpforhackers.io/sentiment-analysis-intro/)
   2. **[Intro into Vader](http://t-redactyl.io/blog/2017/04/using-vader-to-handle-sentiment-analysis-with-social-media-text.html)
   3. [Why vader?](https://www.quora.com/Which-is-the-superior-Sentiment-Analyzer-Vader-or-TextBlob)
   4. **[Vader - a clear explanation about the paper’s methodology](https://www.ijariit.com/manuscripts/v4i1/V4I1-1307.pdf)
   5. Simple Intro to [Vader](https://medium.com/@aneesha/quick-social-media-sentiment-analysis-with-vader-da44951e4116)
   6. [A very lengthy and overly complex explanation about using NTLK vader](https://programminghistorian.org/en/lessons/sentiment-analysis)
   7. [Vader tutorial, +-0.2 for neutrals.](https://www.learndatasci.com/tutorials/sentiment-analysis-reddit-headlines-pythons-nltk/)
4. Text BLob:
   1. Text blob classification
   2. [Python code](https://planspace.org/20150607-textblob_sentiment/)
   3. [More code](https://textminingonline.com/getting-started-with-textblob)
   4. [A lengthy tutorial](https://www.analyticsvidhya.com/blog/2018/02/natural-language-processing-for-beginners-using-textblob/)
   5. **[Text blob sentiment analysis tutorial on medium](https://medium.com/@rahulvaish/textblob-and-sentiment-analysis-python-a687e9fabe96)
   6. [A lengthy intro plus code about text blob](https://aparrish.neocities.org/textblob.html)
5. [Comparative opinion mining a review paper - has some info about unsupervised as well](https://arxiv.org/pdf/1712.08941.pdf)
6. [Another reference list, has some unsupervised.](http://scholar.google.co.il/scholar_url?url=http://www.nowpublishers.com/article/DownloadSummary/INR-011&hl=en&sa=X&scisig=AAGBfm0NN0Pge4htltclF-D6H4BpxocqwA&nossl=1&oi=scholarr)
7. Sentiwordnet3.0 [paper](https://www.researchgate.net/profile/Fabrizio_Sebastiani/publication/220746537_SentiWordNet_30_An_Enhanced_Lexical_Resource_for_Sentiment_Analysis_and_Opinion_Mining/links/545fbcc40cf27487b450aa21.pdf)
8. sentiment presentation
9. [Hebrew Psychological Lexicons](https://github.com/natalieShapira/HebrewPsychologicalLexicons) by Natalie Shapira

   This is the official code accompanying a paper on the [Hebrew Psychological Lexicons](https://www.aclweb.org/anthology/2021.clpsych-1.6.pdf) was presented at CLPsych 2021.

<figure><img src="../.gitbook/assets/image (17).png" alt=""><figcaption><p>Summary Hebrew Psych Lexicon</p></figcaption></figure>

Reference papers:

1. [Twitter as a corpus for SA and opinion mining](http://crowdsourcing-class.org/assignments/downloads/pak-paroubek.pdf)

## Ground Truth

This section records how VADER raters were screened and how multilingual tweet agreement was measured.

1. For sentiment In Vader -
   1. “Screening for English language reading comprehension – each rater had to individually score an 80% or higher on a standardized college-level reading comprehension test.
   2. Complete an online sentiment rating training and orientation session, and score 90% or higher for matching the known (prevalidated) mean sentiment rating of lexical items which included individual words, emoticons, acronyms, sentences, tweets, and text snippets (e.g., sentence segments, or phrases).
   3. Every batch of 25 features contained five “golden items” with a known (pre-validated) sentiment rating distribution. If a worker was more than one standard deviation away from the mean of this known distribution on three or more of the five golden items, we discarded all 25 ratings in the batch from this worker.
   4. Bonus to incentivize and reward the highest quality work. Asked workers to select the valence score that they thought “most other people” would choose for the given lexical feature (early/iterative pilot testing revealed that wording the instructions in this manner garnered a much tighter standard deviation without significantly affecting the mean sentiment rating, allowing us to achieve higher quality (generalized) results while being more economical).
   5. Compensated AMT workers $0.25 for each batch of 25 items they rated, with an additional $0.25 incentive bonus for all workers who successfully matched the group mean (within 1.5 standard deviations) on at least 20 of 25 responses in each batch. Using these four quality control methods, we achieved remarkable value in the data obtained from our AMT workers – we paid incentive bonuses for high quality to at least 90% of raters for most batches.

<figure><img src="../.gitbook/assets/gimg-6eb994bb955a.png" alt=""><figcaption><p>VADER ground-truth rating quality controls.</p><p>Credit: <a href="https://lh3.googleusercontent.com/69nazHo5T9cGMIhgljIDJ4muIjo-fa3PGetGTJwMsktsM699NA2a212TbyqityPup5Q3mVztCO9ieDKSk8y_qDUrTt4DNsCXkjK0Hg70JLyu-xzdqIQScsuc6Va2M2sH_Bp0o8Z_">copied from the original hosted image</a>.</p></figcaption></figure>

[Multilingual Twitter Sentiment Classification: The Role of Human Annotators](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0155036)

- 1.6 million tweets labelled
- 13 languages
- Evaluated 6 pretrained classification models
- 10 CFV
- SVM / NB
- Annotator agreements.
   - about 15% were intentionally duplicated to be annotated twice,
   - by the same annotator
   - by two different annotators
- Self-agreement from multiple annotations of the same annotator
- Inter-agreement from multiple annotations by different annotators
- The confidence intervals for the agreements are estimated by bootstrapping [[12](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0155036#pone.0155036.ref012)].
- It turns out that the self-agreement is a good measure to identify low quality annotators,
- the inter-annotator agreement provides a good estimate of the objective difficulty of the task, unless it is too low.

Alpha was developed to measure the agreement between human annotators, but can also be used to measure the agreement between classification models and a gold standard. It generalizes several specialized agreement measures, takes ordering of classes into account, and accounts for the agreement by chance. Alpha is defined as follows:

<figure><img src="../.gitbook/assets/gimg-dff32c031222.png" alt=""><figcaption><p>Alpha agreement definition.</p><p>Credit: <a href="https://lh4.googleusercontent.com/_7WwUqxDoCvZwOyBlIUEe0k4IWAq1dlTS_kgyBiddpOgIbUS-HcArQzOE3gHDurmR0pceyxF71PZU-NsY5Q65fe_3cFpnak029I3RNnJ_ofWTGjuHwIIYo-GacTF6bKpNSP50FPP">copied from the original hosted image</a>.</p></figcaption></figure>

[Method cont here](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0194317) in a second paper

## Deprecated links

{% hint style="warning" %}
These links and images no longer work. The original wording is kept here. A same-resource copy, when one was checked, is used above.
{% endhint %}

- Sentiwordnet – mapping wordnet senses to a polarity model: SentiWordnet Site This address no longer opens: http://sentiwordnet.isti.cnr.it/
- Text blob classification This address no longer opens: http://rwet.decontextualize.com/book/textblob/
- presentation This address no longer opens: https://web.stanford.edu/class/cs124/lec/sentiment.pdf
