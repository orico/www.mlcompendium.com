# Nearest Neighbors

Nearest neighbours classify a point by the points closest to it, which is simple until the data is large and high-dimensional. This page answers how to do that search at scale: first a library for approximate search, then the question of the optimal K, then the benchmarks that compare libraries up to billion-scale search.

### KNN

The method needs a distance, so it depends on how features are scaled, and it is a common interview topic. The same notes are in [Interview questions](../ai-product/data-science-management.md#interview-questions) and [Normalization & Scaling](../data/normalization-and-scaling.md).

The first problem is speed. [Nearpy](https://github.com/pixelogik/NearPy) is a Python framework on github for fast (approximated) nearest neighbour search in large, high-dimensional data sets using different locality-sensitive hashes: knn in scale! The second problem is finding the optimal K, how many neighbours should vote. Once several libraries can do the search, they have to be compared: the [Benchmark of nearest neighbours libraries](https://github.com/erikbern/ann-benchmarks/) is erikbern's ann-benchmarks, benchmarks of approximate nearest neighbor libraries in Python. At the far end of scale, [billion scale aprox nearest neighbour search](https://big-ann-benchmarks.com/) is the NeurIPS'23 Competition Track: Big-ANN.
