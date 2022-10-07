# Clustering Algorithms

1. [Vidhya on clustering and methods](https://www.analyticsvidhya.com/blog/2016/11/an-introduction-to-clustering-and-different-methods-of-clustering/?utm\_source=facebook.com)
2. [KNN](https://www.youtube.com/watch?v=4ObVzTuFivY) [intuition 2](https://www.youtube.com/watch?v=UqYde-LULfs), [thorough explanation 3](https://towardsdatascience.com/introduction-to-k-nearest-neighbors-3b534bb11d26)  - classify a new sample by looking at the majority vote of its K-nearest neighbours. k=1 special case. Even amount of classes needs an odd K that is not a multiple of the amount of classes in order to break ties.&#x20;
3. [Determinging the number of clusters, a comparison of several methods, elbow, silhouette etc](https://www.datanovia.com/en/lessons/determining-the-optimal-number-of-clusters-3-must-know-methods/)
4. [A good visual example of kmeans / gmm](https://medium.com/sfu-cspmp/distilling-gaussian-mixture-models-701fa9546d9)
5. [Kmeans with DTW, probably fixed length vectors, using tslearn](https://towardsdatascience.com/how-to-apply-k-means-clustering-to-time-series-data-28d04a8f7da3)
6. [Kmeans for variable length](https://medium.com/@iliazaitsev/how-to-classify-a-dataset-with-observations-of-various-length-96fab8e95baf), [notebook](https://github.com/devforfu/Blog/blob/master/trees/scikit\_learn.py)

TOOLS

1. [pyClustering\
   ](https://pyclustering.github.io/docs/0.10.1/html/index.html)![](https://lh5.googleusercontent.com/Wyc8biCZCBvmybSOytsjJYmdhQUVq5F5Kl4tj6luvww9uXVywkBWzCHlsnUaz07KTyIRi98\_vIembQVnhWWRv6DYK\_DhUKC9NNg8mRJPk0cg0Ov4EV66pg7dZW4K7HPEq-xy6axz)

###

### Block Modeling - Distance Matrices

1. [Biclustering and spectral co clustering](https://scikit-learn.org/stable/modules/biclustering.html)
2. [Clustering correlation, or distance matrices.](https://stats.stackexchange.com/questions/138325/clustering-a-correlation-matrix)

![](https://lh6.googleusercontent.com/SdPIjYLt8PksdDmmQDPUn24U1DyNOGyZfsV3V8OxqdU62NzahrACouK7eD5hUkjL\_brbtfRq4uvEUk6FiHR\_vLzr2hbnT774XElKXsZmK3RGnuLGyzFXtxTJyNmnsnrbfxj7Bvv3)

1. Any of the “precomputed” algorithms in sklearn, just remember to [do 1-distanceMatrix](https://github.com/scikit-learn/scikit-learn/issues/6787). I.e., using dbscan/hdbscan/optics, you need a dissimilarity matrix.
2.

### [Kmeans](https://github.com/jakevdp/sklearn\_pycon2015/blob/master/notebooks/04.2-Clustering-KMeans.ipynb)

1. Sensitive to outliers, can skew results (because we rely on the mean)

### [K-mediods](https://en.wikipedia.org/wiki/K-medoids)

&#x20;\- basically k-means with a most center object rather than a center virtual point that was based on mean distance from all points, we keep choosing medoids samples based on minimised SSE

* k-medoid is a classical partitioning technique of clustering that clusters the data set of n objects into k clusters known a priori.
* It is more robust to noise and outliers as compared to [k-means](https://en.wikipedia.org/wiki/K-means) because it minimizes a sum of pairwise dissimilarities instead of a sum of squared Euclidean distances.
* A [medoid](https://en.wikipedia.org/wiki/Medoid) can be defined as the object of a cluster whose average dissimilarity to all the objects in the cluster is minimal. i.e. it is a most centrally located point in the cluster.
* Does Not scale to many samples, its O(K\*n-K)^2
* Randomized resampling can assure efficiency and quality.

[From youtube (okay video)](https://www.youtube.com/watch?v=OWpRBCrx5-M)\


![](https://lh4.googleusercontent.com/rUA\_KIAXZ3nSbbFGo0YsQCF7M5JpTm8Sr2jsdfeIuc2RWeF4OqjRTOE0wVGl7tRkJeiIwnPQJyGS-mKI-PFr\_BUR5e8oWQhw1EGnamVbpmXm0rme2Clfn9Bf--6ZZbgNbsslkOIk)

### X-means![](https://lh6.googleusercontent.com/aSlcmQ3DlWOozVCc4583cI-f-wplzHhygD-ecO7r-J9AtqZQhyWSZkvcClpmuHZdvHUp3MZUCNthXaNG-FB8LqmKhwMmZxiPOO665C4Q\_bp9mB6sIhwbxFw2NwrkaOThSruvIc1Q)

X-means([paper](https://www.cs.cmu.edu/\~dpelleg/download/xmeans.pdf)): \


1. [Theory](https://stats.stackexchange.com/questions/13103/x-mean-algorithm-bic-calculation-question) behind bic calculation with a formula.
2. Code: [Calculate bic in k-means](https://stats.stackexchange.com/questions/90769/using-bic-to-estimate-the-number-of-k-in-kmeans?rq=1)

![](https://lh4.googleusercontent.com/ZOcoLxyDBb42-vW0xKR-8ZjEkmUXh-zFunErX1oKHsS4ZLeaEE-momDpCW7OwVH\_npu66xmojiqd3CwbvQWJkluwutnqBkEDSMluluap5T09YGlUmfWoYQ43XG1U26BHR4wf9Qa9)

### G-means

G-means [Improves on X-means](https://papers.nips.cc/paper/2526-learning-the-k-in-k-means.pdf) in the paper: The G-means algorithm starts with a small number of k-means centers, and grows the number of centers. Each iteration of the algorithm splits into two those centers whose data appear not to come from a Gaussian distribution using the Anderson Darling test. Between each round of splitting, we run k-means on the entire dataset and all the centers to refine the current solution. We can initialize with just k = 1, or we can choose some larger value of k if we have some prior knowledge about the range of k. G-means repeatedly makes decisions based on a statistical test for the data assigned to each enter. If the data currently assigned to a k-means center appear to be Gaussian, then we want to represent that data with only one center.

![](https://lh3.googleusercontent.com/tW\_fWHRABqO3bsqiSobm5FUlkW5sHnoWAFDJZSIGAiiSkYHtBZvUeTmFrR02xPRUQm-rvvoOoeBRh5nmyoz7SyZ4eKj9REFgpGt2lf-SACUCcckg4KiNcTV8Kd2pjtIkfzavzbVU)

### GMM - Gaussian Mixture Models

?- [What is GMM](https://datascience.stackexchange.com/questions/14435/how-to-get-the-probability-of-belonging-to-clusters-for-k-means) in short its knn with mean/variance centroids, a sample can be in several centroids with a certain probability.\


Let us briefly talk about a probabilistic generalisation of k-means: the [Gaussian Mixture Model](https://en.wikipedia.org/wiki/Mixture\_model)(GMM).

In k-means, you carry out the following procedure:

\- specify k centroids, initialising their coordinates randomly

\- calculate the distance of each data point to each centroid

\- assign each data point to its nearest centroid

\- update the coordinates of the centroid to the mean of all points assigned to it

\- iterate until convergence.

In a GMM, you carry out the following procedure:

\- specify k multivariate Gaussians (termed components), initialising their mean and variance randomly

\- calculate the probability of each data point being produced by each component (sometimes termed the responsibility each component takes for the data point)

\- assign each data point to the component it belongs to with the highest probability

\- update the mean and variance of the component to the mean and variance of all data points assigned to it

\- iterate until convergence

You may notice the similarity between these two procedures. In fact, k-means is a GMM with fixed-variance components. Under a GMM, the probabilities (I think) you're looking for are the responsibilities each component takes for each data point.\


1. [Gmm code on sklearn](https://scikit-learn.org/stable/auto\_examples/mixture/plot\_gmm.html#sphx-glr-auto-examples-mixture-plot-gmm-py) using ellipsoids
2. [How to select the K  using bic](https://scikit-learn.org/stable/auto\_examples/mixture/plot\_gmm\_selection.html#sphx-glr-auto-examples-mixture-plot-gmm-selection-py)
3. [Density estimation for gmm - nice graph](https://scikit-learn.org/stable/auto\_examples/mixture/plot\_gmm\_pdf.html#sphx-glr-auto-examples-mixture-plot-gmm-pdf-py)

### KMEANS++ / Kernel Kmeans

1. [A comparison of kmeans++ vs kernel kmeans](https://sandipanweb.wordpress.com/2016/08/29/kernel-k-means-and-cluster-evaluation/)
2. [Kernel Kmeans is part of TSLearn ](http://tslearn.readthedocs.io/en/latest/gen\_modules/clustering/tslearn.clustering.GlobalAlignmentKernelKMeans.html)
3. [Elbow method](https://blog.cambridgespark.com/how-to-determine-the-optimal-number-of-clusters-for-k-means-clustering-14f27070048f),&#x20;
4. [elbow and mean silhouette](https://www.datanovia.com/en/lessons/determining-the-optimal-number-of-clusters-3-must-know-methods/#elbow-method),&#x20;
5. [elbow on medium using mean distance per cluster from the center](https://towardsdatascience.com/what-is-k-ddf36926a752)
6. [Kneed a library to find the knee in a curve](https://github.com/arvkevi/kneed)
   1. [how to?](https://stackoverflow.com/questions/47623915/how-to-detect-in-real-time-a-knee-elbow-maximal-curvature-in-a-curve)

### KNN

1. [Nearpy](https://github.com/pixelogik/NearPy), knn in scale! On github
2. [finding the optimal K](https://towardsdatascience.com/how-to-find-the-optimal-value-of-k-in-knn-35d936e554eb)
3. [Benchmark of nearest neighbours libraries](https://github.com/erikbern/ann-benchmarks/)
4. [billion scale aprox nearest neighbour search](https://big-ann-benchmarks.com/)

### DBSCAN

1. [How to use effectively](https://towardsdatascience.com/how-to-use-dbscan-effectively-ed212c02e62)
2. [a DBSCAN visualization - very good!](https://www.naftaliharris.com/blog/visualizing-dbscan-clustering/)
3. [DBSCAN for GPS.](https://geoffboeing.com/2014/08/clustering-to-reduce-spatial-data-set-size/)
4. [A practical guide to dbscan - pretty good](https://towardsdatascience.com/a-practical-guide-to-dbscan-method-d4ec5ab2bc99)
5. [Custom DBSCAN  “predict”](https://stackoverflow.com/questions/27822752/scikit-learn-predicting-new-points-with-dbscan)
6. [Haversine distances for](https://kanoki.org/2019/12/27/how-to-calculate-distance-in-python-and-pandas-using-scipy-spatial-and-distance-functions/) dbscan
7. Optimized dbscans:
   1. [muDBSCAN](https://githubmemory.com/repo/AdityaAS/MuDBSCAN), [paper](https://adityaas.github.io/) - A fast, exact, and scalable algorithm for DBSCAN clustering. This repository contains the implementation for the distributed spatial clustering algorithm proposed in the paper μDBSCAN: An Exact Scalable DBSCAN Algorithm for Big Data Exploiting Spatial Locality&#x20;
   2. [Dbscan multiplex](https://github.com/GGiecold/DBSCAN\_multiplex) - A fast and memory-efficient implementation of DBSCAN (Density-Based Spatial Clustering of Applications with Noise).
   3. [Fast dbscan](https://github.com/harmslab/fast\_dbscan) - A lightweight, fast dbscan implementation for use on peptide strings. It uses pure C for the distance calculations and clustering. This code is then wrapped in python.
   4. [Faster dbscan paper](https://arxiv.org/pdf/1702.08607.pdf)

### ST-DBSCAN

1. [Paper - st-dbscan an algo for clustering spatio temporal data](https://www.sciencedirect.com/science/article/pii/S0169023X06000218)
2. [Popular git](https://github.com/eubr-bigsea/py-st-dbscan)
3. [git](https://github.com/gitAtila/ST-DBSCAN)

### HDBSCAN\*

(what is?) HDBSCAN is a clustering algorithm developed by [Campello, Moulavi, and Sander](http://link.springer.com/chapter/10.1007%2F978-3-642-37456-2\_14). It extends DBSCAN by converting it into a hierarchical clustering algorithm, and then using a technique to extract a flat clustering based in the stability of clusters.

* [Github code](https://github.com/scikit-learn-contrib/hdbscan)
* (great) [Documentation](http://hdbscan.readthedocs.io/en/latest/basic\_hdbscan.html)  with examples, for clustering, outlier detection, comparison, benchmarking and analysis!
* ([jupytr example](http://nbviewer.jupyter.org/github/scikit-learn-contrib/hdbscan/blob/master/notebooks/How%20HDBSCAN%20Works.ipynb)) - take a look and see how to use it, usage examples are also in the docs and github

What are the algorithm’s [steps](http://nbviewer.jupyter.org/github/scikit-learn-contrib/hdbscan/blob/master/notebooks/How%20HDBSCAN%20Works.ipynb):

1. Transform the space according to the density/sparsity.
2. Build the minimum spanning tree of the distance weighted graph.
3. Construct a cluster hierarchy of connected components.
4. Condense the cluster hierarchy based on minimum cluster size.
5. Extract the stable clusters from the condensed tree.

### OPTICS

([What is?](https://en.wikipedia.org/wiki/OPTICS\_algorithm)) Ordering points to identify the clustering structure (OPTICS) is an algorithm for finding density-based[\[1\]](https://en.wikipedia.org/wiki/OPTICS\_algorithm#cite\_note-1) [clusters](https://en.wikipedia.org/wiki/Cluster\_analysis) in spatial data

* Its basic idea is similar to [DBSCAN](https://en.wikipedia.org/wiki/DBSCAN),[\[3\]](https://en.wikipedia.org/wiki/OPTICS\_algorithm#cite\_note-3)&#x20;
* it addresses one of DBSCAN's major weaknesses: the problem of detecting meaningful clusters in data of varying density.&#x20;
* (How?) the points of the database are (linearly) ordered such that points which are spatially closest become neighbors in the ordering.&#x20;
* a special distance is stored for each point that represents the density that needs to be accepted for a cluster in order to have both points belong to the same cluster. (This is represented as a [dendrogram](https://en.wikipedia.org/wiki/Dendrogram).)

### SVM CLUSTERING

[Paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2099486/)

An SVM-based clustering algorithm is introduced that clusters data with no a priori knowledge of input classes.&#x20;

1. The algorithm initializes by first running a binary SVM classifier against a data set with each vector in the set randomly labelled, this is repeated until an initial convergence occurs.&#x20;
2. Once this initialization step is complete, the SVM confidence parameters for classification on each of the training instances can be accessed.&#x20;
3. The lowest confidence data (e.g., the worst of the mislabelled data) then has its' labels switched to the other class label.&#x20;
4. The SVM is then re-run on the data set (with partly re-labelled data) and is guaranteed to converge in this situation since it converged previously, and now it has fewer data points to carry with mislabelling penalties.&#x20;
5. This approach appears to limit exposure to the local minima traps that can occur with other approaches. Thus, the algorithm then improves on its weakly convergent result by SVM re-training after each re-labeling on the worst of the misclassified vectors – i.e., those feature vectors with confidence factor values beyond some threshold.&#x20;
6. The repetition of the above process improves the accuracy, here a measure of separability, until there are no misclassifications. Variations on this type of clustering approach are shown.

### COP-CLUSTERING

Constrained K-means algorithm, [git](https://github.com/Behrouz-Babaki/COP-Kmeans), [paper](https://web.cse.msu.edu/\~cse802/notes/ConstrainedKmeans.pdf), is a semi-supervised algorithm.\


Clustering is traditionally viewed as an unsupervised method for data analysis. However, in some cases information about the problem domain is available in addition to the data instances themselves. In this paper, we demonstrate how the popular k-means clustering algorithm can be profitably modified to make use of this information. In experiments with artificial constraints on six data sets, we observe improvements in clustering accuracy. We also apply this method to the real-world problem of automatically detecting road lanes from GPS data and observe dramatic increases in performance.\
\


In the context of partitioning algorithms, instance level constraints are a useful way to express a priori knowledge about which instances should or should not be grouped together. Consequently, we consider two types of pairwise constraints:\
• Must-link constraints specify that two instances have to be in the same cluster.\
• Cannot-link constraints specify that two instances must not be placed in the same cluster.![](https://lh6.googleusercontent.com/mluNAa5\_RoVGMVfqqJRR01zRsquiK9uReJsPRxXrh0lxoXSChR-OutR\_n4mg4CtILYTTIefFBpNPO3eU0YRYIQaW\_3WD3hZrsd8erIrB9qivtCL4kLzw42-EUT-X8rqp7VQFRmJL)
