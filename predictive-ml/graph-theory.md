# Graph Theory

When data is a set of relationships rather than a table, the questions become how nodes connect, which nodes matter, and which groups form. The page walks the graph topics first, from general libraries through connectivity, community detection, and node2vec embeddings to how communities are evaluated, then the Neo4j centrality, community, and experimental algorithms, and ends with graph and GNN courses and the Python tools to run it all.

The same notes are in [Block Modeling - Distance Matrices](clustering-algorithms.md#block-modeling---distance-matrices), [Deep walk](../deep-learning/graph-neural-nets.md#deep-walk), [Fraud Detection](../ai-product/fraud-detection.md), [GRAPH NEURAL NETWORKS (GNN)](../deep-learning/graph-neural-nets.md#graph-neural-networks-gnn), [Root Cause Effects (RCE/RCA)](../decision-intelligence/root-cause-effects-rce-rca.md), and [Social Network Analysis](social-network-analysis.md).

## Graph Topics

The general-purpose starting point is a single library that covers many unsupervised graph methods at once. [General purpose and community detection GIT](https://github.com/benedekrozemberczki/karateclub) is Karate Club, an API oriented open-source Python framework for unsupervised learning on graphs (CIKM 2020).

Connectivity is the first structural question: where does the graph come apart? Min-cut: [1](https://github.com/gsw73/min-cut/blob/master/karger_min_cut.py) implements the Karger min cut algorithm on a graph read from a text file. [2](https://github.com/ChuntaoLu/Algorithms-Design-and-Analysis/blob/master/week3%20Karger%20min%20cut/min_cut.py) is the Karger min cut from a repository of algorithms learned in Stanford's Algorithms: Design and Analysis. Min cut [3](https://github.com/WithaK16/kargerMinCut/blob/master/kargerMinCut.py) is another Karger min cut algorithm implementation in Python.

Cutting a graph leads directly to finding its communities. [Louvain community](https://github.com/taynaud/python-louvain/) is taynaud/python-louvain, Louvain community detection. Girvan–Newman [gist](https://gist.github.com/chelsea1992/6c725a24d358763097bebe8223c2014a) is community detection with the Girvan-Newman algorithm, which several repositories implement as divisive hierarchical clustering: [this worked](https://github.com/ZwEin27/Community-Detection) is ZwEin27/Community-Detection, and [this is potentially good too](https://github.com/riteshkasat/Community-Detection-Algorithm) is riteshkasat/Community-Detection-Algorithm. The remaining implementations are [another](https://github.com/ServiceCutter/girvan-newman), the Girvan-Newman clustering used by the Service Cutter; [another](https://github.com/ZwEin27/Community-Detection), the same ZwEin27 repository again; and [another](https://github.com/kjahan/community), kjahan's Python implementation of the Girvan-Newman algorithm.

Communities describe groups; embeddings give each node a vector a model can use. [Node2vec](https://github.com/eliorc/Medium/blob/master/Nod2Vec-FIFA17-Example.ipynb) is the FIFA17 example notebook in eliorc's repository of code related to his Medium blog posts. The [paper](https://arxiv.org/pdf/1607.00653.pdf) is node2vec: Scalable Feature Learning for Networks. [medium1](https://towardsdatascience.medium.com/think-your-data-different-ddc435f70850) is Think your Data Different, on applying deep learning to graph datasets in social networks, recommender systems, and biology, where data is inherently structured as a graph. [medium 2](https://towardsdatascience.medium.com/node2vec-embeddings-for-graph-data-32a866340fef) is node2vec: Embeddings for Graph Data, which argues that quality embeddings are the opposite of "garbage in, garbage out". The implementation of the node2vec algorithm — tutorial — is the [code](https://github.com/eliorc/node2vec), and the [git code](https://github.com/eliorc/Medium/blob/master/Nod2Vec-FIFA17-Example.ipynb) is the same FIFA17 notebook from the Medium posts. The [original py2 code](https://github.com/aditya-grover/node2vec) is aditya-grover/node2vec, and [taboola code for their medium paper](https://github.com/taboola/node2vec-example/blob/master/node2vec.ipynb) is the taboola/node2vec-example notebook.

The same notes are in [Node2vec](../deep-learning/graph-neural-nets.md#node2vec).

Once communities are found, they have to be judged. [Evaluation metrics for community detection](https://stackoverflow.com/questions/28952104/evaluation-metrics-for-community-detection-algorithms) is the Stack Overflow question on evaluation metrics for community detection algorithms. [Review for community detection algorithms](https://arxiv.org/pdf/0906.0612.pdf) is Community detection in graphs, and the same review's abstract page is the [paper](https://arxiv.org/abs/0906.0612): community structure, or clustering, is the organization of vertices in clusters with many edges joining vertices of the same cluster and comparatively few edges joining vertices of different clusters. [Term: community structure](https://en.wikipedia.org/wiki/Community_structure#Algorithms_for_finding_communities) is the Wikipedia entry on community structure and its algorithms for finding communities, and [Term: modularity of networks](https://en.wikipedia.org/wiki/Modularity_(networks) is the Wikipedia entry on modularity (networks).

Two papers are still on the reading pile. [Unread paper](http://science.sciencemag.org/content/328/5980/876) is the first. [Unread comparison of community detection algos](https://arxiv.org/abs/1406.2205) compares community detection algorithms for multiplex graphs, where a multiplex is a set of graphs on the same vertex set, i.e. a generalized graph for multiple relationships with parallel edges between vertices.

Community detection can also be treated as clustering a matrix. [Clustering adjacency matrices](https://stats.stackexchange.com/questions/125295/the-best-way-for-clustering-an-adjacency-matrix) is Fahd's Cross Validated question on the best way for clustering an adjacency matrix, built from partial correlations of neural time series. [Spectral-clustering](https://calculatedcontent.com/2012/10/09/spectral-clustering/) is Spectral Clustering: A quick overview, from calculated | content. [Finding natural groups in undirected graphs](https://stats.stackexchange.com/questions/142297/finding-natural-groups-clusters-in-an-undirected-graph-over-several-undirect) is the question on finding natural groups or clusters in one undirected graph or over several graphs that may share nodes. For more, [Awesome community detection on github](https://github.com/benedekrozemberczki/awesome-community-detection) is a curated list of community detection research papers with implementations.

The ready-made versions of these algorithms are in the Neo4j Graph Data Science library, and [Various algorithms](https://neo4j.com/docs/graph-algorithms/current/algorithms/closeness-centrality/) is its manual; the subsections below follow its chapters.

### Centrality algorithms

Centrality answers which nodes matter most, and [5. Centrality algorithms](https://neo4j.com/docs/graph-algorithms/current/algorithms/centrality/) is the Neo4j chapter with explanations and examples for each centrality algorithm in the library:

- [5.1. The PageRank algorithm](https://neo4j.com/docs/graph-algorithms/current/algorithms/page-rank/)
- [5.2. The Betweenness Centrality algorithm](https://neo4j.com/docs/graph-algorithms/current/algorithms/betweenness-centrality/)
- [5.3. The Closeness Centrality algorithm](https://neo4j.com/docs/graph-algorithms/current/algorithms/closeness-centrality/)
- [5.4. The Degree Centrality algorithm](https://neo4j.com/docs/graph-algorithms/current/algorithms/degree-centrality/)

### Community detection algorithms

After the important nodes come the groups, and [6. Community detection algorithms](https://neo4j.com/docs/graph-algorithms/current/algorithms/community/) is the chapter with explanations and examples for each community detection algorithm in the library, including Louvain from the topics above:

- [6.1. The Louvain algorithm](https://neo4j.com/docs/graph-algorithms/current/algorithms/louvain/)
- [6.2. The Label Propagation algorithm](https://neo4j.com/docs/graph-algorithms/current/algorithms/label-propagation/)
- [6.3. The Connected Components algorithm](https://neo4j.com/docs/graph-algorithms/current/algorithms/connected-components/)

### Experimental algorithms

Beyond the stable chapters, [7. Experimental algorithms](https://neo4j.com/docs/graph-algorithms/current/experimental-algorithms/) extends the same manual to procedures, path finding, similarity, and link prediction:

- [7.1. Procedures](https://neo4j.com/docs/graph-algorithms/current/experimental-procedures/)
- [7.2. Centrality algorithms](https://neo4j.com/docs/graph-algorithms/current/experimental-algorithms/centrality/)
- [7.3. Community detection algorithms](https://neo4j.com/docs/graph-algorithms/current/experimental-algorithms/community/)
- [7.4. Path finding algorithms](https://neo4j.com/docs/graph-algorithms/current/experimental-algorithms/pathfinding/)
- [7.5. Similarity algorithms](https://neo4j.com/docs/graph-algorithms/current/experimental-algorithms/similarity/)
- [7.6. Link Prediction algorithms](https://neo4j.com/docs/graph-algorithms/current/experimental-algorithms/linkprediction/)
- [7.7. Preprocessing functions and](https://neo4j.com/docs/graph-algorithms/current/experimental-algorithms/preprocessing/)

## Graph/GNN courses

The algorithms above are classical; the courses carry the same graphs into machine learning and graph neural networks.

The same notes are in [GNN courses](../deep-learning/graph-neural-nets.md#gnn-courses).

[Machine learning with graphs by Stanford](http://web.stanford.edu/class/cs224w/) is CS224W. [Graph deep learning course](https://geometricdeeplearning.com/lectures/) is the geometric deep learning course on grids, groups, graphs, geodesics, and gauges. Its keynote version is Michael Bronstein's ICLR 2021 talk "Geometric Deep Learning: The Erlangen Programme of ML" — graphs, sets, groups, GNNs — on [YouTube](https://www.youtube.com/watch?v=w6Pw4MOzMuo).

## Graph Tools

To run any of this in Python, two libraries do the graph handling. [Graph-tool](https://graph-tool.skewed.de/) is an efficient Python module for manipulation and statistical analysis of graphs. [NetworkX](https://github.com/networkx/networkx) is a Python package for the creation, manipulation, and study of the structure, dynamics, and functions of complex networks.
