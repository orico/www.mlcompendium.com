# Multi Label Classification

**\(what is?\)** [**Multilabel classification**](https://mlr-org.github.io/mlr-tutorial/devel/html/multilabel/index.html) **is a classification problem where multiple target labels can be assigned to each observation instead of only one like in multiclass classification.**

**Two different approaches exist for multilabel classification:**

*  **Problem transformation methods try to transform the multilabel classification into binary or multiclass classification problems.** 
* **Algorithm adaptation methods adapt multiclass algorithms so they can be applied directly to the problem.**

**I.e., the** [**Two approaches**](https://mlr-org.github.io/mlr-tutorial/devel/html/multilabel/index.html) **are:** 

* **Use a classifier that does multi label**
* **Use any classifier with a wrapper that compares each two labels**

**great** [**PDF**](https://users.ics.aalto.fi/jesse/talks/Multilabel-Part01.pdf) **that explains about multi label classification and especially metrics,** [**part 2 here**](https://users.ics.aalto.fi/jesse/talks/Multilabel-Part02.pdf)

[**An awesome Paper**](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.104.9401&rep=rep1&type=pdf) **that explains all of these methods in detail, also available** [**here**](https://www.researchgate.net/publication/273859036_Multi-Label_Classification_An_Overview)**!**

**PT1: for each sample select one label, remove all others.**

**PT2: remove every sample which has multi labels.**

**PT3: for every combo of labels create a single-label, i.e. A&B, A&C etc..**

**PT4: \(most common\) create L datasets, for each label learn a binary representation, i.e., is it there or not.**

**PT5: duplicate each sample with only one of its labels**

**PT6: read the paper**

**There are other approaches for doing it within algorithms, they rely on the ideas PT3\4\5\6 implemented in the algorithms, or other tricks.**

**They also introduce Label cardinality and label density.**

[**Efficient net**](https://medium.com/gumgum-tech/multi-label-classification-for-threat-detection-part-1-60318b90ce11)**,** [**part 2**](https://medium.com/gumgum-tech/multi-label-image-classifier-for-threat-detection-with-fp16-inference-part-2-40fe0f9a93b3) **- EfficientNet is based on a network derived from a neural architecture search and novel compound scaling method is applied to iteratively build more complex network which achieves state of the art accuracy on multiclass classification tasks. Compound scaling refers to increasing the network dimensions in all three scaling formats using a novel strategy.  
  
Multi label confusion matrices with sklearn** 

[ **Scikit multilearn package**](http://scikit.ml/index.html)

