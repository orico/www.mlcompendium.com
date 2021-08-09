# Untitled



## **DECISION TREES**

1. [**Using hellinger distance to split supervised datasets, instead of gini and entropy. Claims better results.**](https://medium.com/@evgeni.dubov/classifying-imbalanced-data-using-hellinger-distance-f6a4330d6f9a)

**Visualize decision** [**trees**](https://towardsdatascience.com/interactive-visualization-of-decision-trees-with-jupyter-widgets-ca15dd312084)**,** [**forests**](https://towardsdatascience.com/how-to-visualize-a-decision-tree-from-a-random-forest-in-python-using-scikit-learn-38ad2d75f21c)  
****

### [**CART TREES**](http://machinelearningmastery.com/classification-and-regression-trees-for-machine-learning/) ****

**explains about the similarities and how to measure. which is the best split? based on SSE and GINI \(good info about gini here\).**

* **For classification the Gini cost function is used which provides an indication of how “pure” the leaf nodes are \(how mixed the training data assigned to each node is\).**

**Gini = sum\(pk \* \(1 – pk\)\)**

* **Early stop - 1 sample per node is overfitting, 5-10 are good**
* **Pruning - evaluate what happens if the lead nodes are removed, if there is a big drop, we need it.**

### **KDTREE** 

1. [**Similar to a binary search tree, just by using the median and selecting a feature randomly for each level.**](https://www.youtube.com/watch?v=TLxWtXEbtFE)
2. [**Used to find nearest neighbours.**](https://www.youtube.com/watch?v=Y4ZgLlDfKDg) ****
3. [**Many applications of using KD tree, reduce color space, Database key search, etc**](https://www.quora.com/What-is-a-kd-tree-and-what-is-it-used-for)

### **RANDOM FOREST**

[**Using an ensemble of trees to create a high dimensional and sparse representation of the data and classifying using a linear classifier**](http://scikit-learn.org/stable/auto_examples/ensemble/plot_feature_transformation.html#sphx-glr-auto-examples-ensemble-plot-feature-transformation-py)  
****

[**How do deal with imbalanced data in Random-forest**](http://statistics.berkeley.edu/sites/default/files/tech-reports/666.pdf) **-** 

1. **One is based on cost sensitive learning.** 
2. **Other is based on a sampling technique** 

### **EXTRA TREES**

1. [**A comparison between random forest and extra trees** ](https://www.thekerneltrip.com/statistics/random-forest-vs-extra-tree/)**Fig. 1: Comparison of random forests and extra trees in presence of irrelevant predictors. In blue are presented the results from the random forest and red for the extra trees. The results are quite striking: Extra Trees perform consistently better when there are a few relevant predictors and many noisy ones**![Comparison of random forests and extra trees in presence of irrelevant predictors](https://lh3.googleusercontent.com/frZzCFNyzH8WZmbb0IIy_-e-wsqwclzspkGC9p2AIpRHOH1L-AEWAfQqvy96s26rts-VmSNHN8LSJMvNMjXtIv5qcE3j_MZQjnbM2ped7g7oy0Nli59cv1YhM_cGH2G2Ne67MSwM)
2. [**Difference between RF and ET**](https://stats.stackexchange.com/questions/175523/difference-between-random-forest-and-extremely-randomized-trees)
3. [**Differences \#2**](https://stackoverflow.com/questions/22409855/randomforestclassifier-vs-extratreesclassifier-in-scikit-learn)

