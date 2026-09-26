# Name Matching

Matching person names across two lists is fuzzy string matching with a harder twist: spelling, romanization, and culture all vary for the same identity. The page first reads how others solved it, then lists the public name datasets to test against, and ends with the libraries that do the matching.

The same notes are in [String Matching](string-matching.md).

## Articles

The problem is best understood through people who had to solve it. [Analytics Vidhya](https://medium.com/analytics-vidhya/fuzzy-name-matching-datasets-1ae28884f226) on fuzzy name matching datasets, by Zaki Jefferson, frames it across datasets. [Fuzzy matching people names](https://medium.com/data-science/fuzzy-matching-people-names-6e738d6b8fe) starts from a concrete case: given two unordered lists with real people names, match identities in between. Name Matching Across datasets — POC by Centere of Excellence in AI National Informatics Centre was the government proof of concept; its address no longer opens and is kept at the end of the page. For the algorithms themselves, [fuzzy name matching algorithms](https://medium.com/data-science/python-tutorial-fuzzy-name-matching-algorithms-7a6f43322cc5) is Felix Kuestahler's Python tutorial, the fifth article in his series on Python data exploration.

## Datasets

An algorithm is only as convincing as the names it was tested on. The largest is the [first and last name dataset](https://github.com/philipperemy/name-dataset), philipperemy/name-dataset, the Python library for names built from facebook 533M records, by philippe remy. The data.world open data community keeps [data.world name datasets](https://data.world/datasets/names), and FiveThirtyEight's Most Common Name Dataset is on [Kaggle](https://www.kaggle.com/datasets/fivethirtyeight/fivethirtyeight-most-common-name-dataset).

<figure><img src="../.gitbook/assets/image (32).png" alt=""><figcaption><p>Kaggle name datasets by fivethirtyeight.</p></figcaption></figure>

The figure shows those FiveThirtyEight name datasets as they appear on Kaggle. When gender is the attribute to predict or control for, the UCI Machine Learning Repository has the [gender by name dataset](https://archive.ics.uci.edu/ml/datasets/Gender+by+Name). For evaluation across cultures, the [paper](http://www.lrec-conf.org/proceedings/lrec2008/pdf/291_paper.pdf) — a ground truth dataset for matching coltural diverse romanized person names.

## Tools

With articles and data in hand, the remaining piece is a library that matches at scale. [Dedupe](https://www.reddit.com/r/datasets/comments/4zrozk/request_name_matching_dataset/) — a python library for accurate and scalable fuzzy matching record deduplication and entity resolution. [name](https://github.com/bradhackinen/nama) — fast flexible name matching for large datasets. The athenianco repository is the [name matcher](https://github.com/athenianco/names-matcher) shown below.

<figure><img src="../.gitbook/assets/image (13).png" alt=""><figcaption><p>Name matcher by athenianco.</p></figcaption></figure>

To test any of these tools, the Kaggle, Name datasets, by fivethirtyeight are also linked at a fixed version: [https://www.kaggle.com/fivethirtyeight/fivethirtyeight-most-common-name-dataset/version/108](https://www.kaggle.com/fivethirtyeight/fivethirtyeight-most-common-name-dataset/version/108)

## Deprecated links

{% hint style="warning" %}
These links and images no longer work. The original wording is kept here. A same-resource copy, when one was checked, is used above.
{% endhint %}

- Fuzzy matching people names by vadim markovtsev. This address no longer opens: https://towardsdatascience.com/fuzzy-matching-people-names-6e738d6b8fe
- Name Matching Across datasets - POC by Centere of Excellence in AI National Informatics Centre. This address no longer opens: https://ai.nic.in/AI/NameMatchingCaseML
- fuzzy name matching algorithms by felix kuestahler. This address no longer opens: https://towardsdatascience.com/python-tutorial-fuzzy-name-matching-algorithms-7a6f43322cc5
