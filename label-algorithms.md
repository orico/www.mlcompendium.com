# Label Algorithms

### Unbalanced labels

1. [imbalance learn](https://imbalanced-learn.org/stable/auto_examples/over-sampling/plot_comparison_over_sampling.html#sphx-glr-auto-examples-over-sampling-plot-comparison-over-sampling-py) - is an open-source, MIT-licensed library that provides tools when dealing with classification with imbalanced classes. 
2. [Classifying Job Titles With Noisy Labels Using REINFORCE ](https://medium.com/@ziprecruiter.engineering/classifying-job-titles-with-noisy-labels-using-reinforce-ce1a4bde05e2)this article has a very nice trick in adding a reward component to the loss function in order to mitigate for unbalanced class label problem, instead of the usual balancing.

![Imbalance Learn comparison](.gitbook/assets/image%20%282%29.png)

### **Label Propagation / Spreading**

**Note: very much related to weakly and semi supervision, i.e., we have small amounts of labels and we want to generalize the labels to other samples, see also weak supervision methods.**

1. **Step 1:** [**build a laplacian**](https://en.wikipedia.org/wiki/Laplacian_matrix) **graph using KNN, distance metric is minkowski with p=2, i.e. euclidean distance.**
2. [**Step by step tutorial**](https://medium.com/@graphml/introduction-to-label-propagation-with-networkx-part-1-abcbe954a2e8)**,** [**part 2**](https://medium.com/@graphml/introduction-to-label-propagation-with-networkx-part-2-cd041fa44e1)
3. [**Spreading**](https://scikit-learn.org/stable/modules/generated/sklearn.semi_supervised.LabelSpreading.html) **\(propagation upgrade\), Essentially a community graph algorithm, however it resembles KNN in its nature, using semi supervised data set \(i.e., labeled and unlabeled data\) to spread or propagate labels to unlabeled data, with small incrementations in the algorithm, using KNN-like methodology, each unlabeled sample will be given a label based on its 1st order friends, if there is a tie, a random label is chosen. Nodes are connected by using a euclidean distance.**
4. [**Difference**](https://www.researchgate.net/post/What_is_the_difference_between_Label_propagation_and_Label_spreading_in_semi-supervised_learning_context) **between propagation and spreading is a laplacian matrix, vs normalized LM**
5. [**Laplacian matrix on youtube, videos 30-33**](https://www.youtube.com/watch?v=siCPjpUtE0A&list=PLLssT5z_DsK9JDLcT8T62VtzwyW9LNepV&index=33)
6. [**Really good example notebook**](https://github.com/DavidBrear/sklearn-cookbook/blob/master/Chapter%204/4.1.1%20Label%20Propagation%20with%20Semi-Supervised%20Learning.ipynb)
7. [**Spreading vs propagation**](https://www.researchgate.net/post/What_is_the_difference_between_Label_propagation_and_Label_spreading_in_semi-supervised_learning_context)
8. [**https://en.wikipedia.org/wiki/Label\_Propagation\_Algorithm**](https://en.wikipedia.org/wiki/Label_Propagation_Algorithm)
9. **Youtube** [**1**](https://www.youtube.com/watch?v=UWf8hxeehOg)**,** [**2**](https://www.youtube.com/watch?v=hmashUPJwSQ)**,** [**3**](https://www.youtube.com/watch?v=F4f247IyOTs)**,**
10. [**Medium**](https://medium.com/@graphml/introduction-to-label-propagation-with-networkx-part-1-abcbe954a2e8)**,**
11. [**Sklearn**](https://scikit-learn.org/stable/modules/label_propagation.html)**,** [**1**](https://scikit-learn.org/stable/modules/generated/sklearn.semi_supervised.LabelPropagation.html)**,** [**2**](https://scikit-learn.org/stable/modules/generated/sklearn.semi_supervised.LabelPropagation.html)**,** [**3**](https://scikit-learn.org/stable/auto_examples/semi_supervised/plot_label_propagation_digits.html)**,** [**4**](https://plot.ly/scikit-learn/plot-label-propagation-structure/)**, 5,**

![](https://lh3.googleusercontent.com/RvKaNtYZDEWL0GUPmS-z4SlFVQvjBMV2Y1rSIwhncDXEMYeSxOsQ2CgEdAIcY5zM0d_ECzRpmaMJ887wktGP-oS408o-Uwt9d3ECUzELSP6anOh0WoWGruUvy02cQTMTMfPv7hMC)

1. [**Git**](https://github.com/benedekrozemberczki/LabelPropagation)**,** [**incremental LP**](https://github.com/johny-c/incremental-label-propagation)
2. [**Git2**](https://github.com/yamaguchiyuto/label_propagation%5C)
   1. **Harmonic Function \(HMN\) \[Zhu+, ICML03\]**
   2. **Local and Global Consistency \(LGC\) \[Zhou+, NIPS04\]**
   3. **Partially Absorbing Random Walk \(PARW\) \[Wu+, NIPS12\]**
   4. **OMNI-Prop \(OMNIProp\) \[Yamaguchi+, AAAI15\]**
   5. **Confidence-Aware Modulated Label Propagation \(CAMLP\) \[Yamaguchi+, SDM16\]**
3. 
![](https://lh6.googleusercontent.com/O7nhJu4DU47zpTRkJy53CloKGW6Msk7jZIhMdsI3VePsRgzJji3XCG0Nmlpv4F3rBmb4eS-fTRMUyuTfwaHE9k687ScSFYQmadOkIKRNaRMBvW-PiRs1vGeINYTV8uYZ3tjmcdRk)

1. **Presentation** [**1**](http://www.leonidzhukov.net/hse/2015/networks/lectures/lecture17.pdf)**,**[**2** ](https://www.slideshare.net/dav009/label-propagation-semisupervised-learning-with-applications-to-nlp)

**Neo4j** [**1**](https://dzone.com/articles/graph-algorithms-in-neo4j-label-propagation)**, 2, 3,**

