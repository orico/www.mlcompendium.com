# Graph Theory

This page collects graph topics such as centrality, community detection, and experimental algorithms, then courses and tools. It is a shelf for graph and GNN learning resources alongside Neo4j-oriented algorithm notes.

The same notes are in [Block Modeling - Distance Matrices](clustering-algorithms.md#block-modeling---distance-matrices), [Deep walk](../deep-learning/graph-neural-nets.md#deep-walk), [Fraud Detection](../ai-product/fraud-detection.md), [GRAPH NEURAL NETWORKS (GNN)](../deep-learning/graph-neural-nets.md#graph-neural-networks-gnn), [Root Cause Effects (RCE/RCA)](../decision-intelligence/root-cause-effects-rce-rca.md), and [Social Network Analysis](social-network-analysis.md).

## Graph Topics

This section covers community detection, connectivity, embeddings, and related graph topics.

- Karate Club: An API Oriented Open-source Python Framework for Unsupervised Learning on Graphs (CIKM 2020) - benedekrozemberczki/karateclub. [General purpose and community detection GIT](https://github.com/benedekrozemberczki/karateclub)
2. Connectivity
- Implements Karger Min Cut algorithm on a graph input from a text file. Min-cut: [1](https://github.com/gsw73/min-cut/blob/master/karger_min_cut.py)
- Algorithms I learned from Stanford's Algorithms: Design and Analysis - Algorithms-Design-and-Analysis/week3 Karger min cut/min_cut.py at master · ChuntaoLu/Algorithms-Design-and-Analysis. [2](https://github.com/ChuntaoLu/Algorithms-Design-and-Analysis/blob/master/week3%20Karger%20min%20cut/min_cut.py)
- karger min cut algorithm implementation in python. Min cut [3](https://github.com/WithaK16/kargerMinCut/blob/master/kargerMinCut.py)
- GitHub - taynaud/python-louvain: Louvain Community Detection. [Louvain community](https://github.com/taynaud/python-louvain/)
- Community Detection, Girvan-Newman algorithm. Girvan–Newman [gist](https://gist.github.com/chelsea1992/6c725a24d358763097bebe8223c2014a)
- Implement a community detection algorithm using a divisive hierarchical clustering (Girvan-Newman algorithm) - ZwEin27/Community-Detection. [this worked](https://github.com/ZwEin27/Community-Detection)
- This project implements community detection algorithm using divisive hierarchical clustering (Girvan-Newman algorithm) - riteshkasat/Community-Detection-Algorithm. [this is potentially good too](https://github.com/riteshkasat/Community-Detection-Algorithm)
- Implementation of the Girvan-Newman clustering algorithm used by the Service Cutter - ServiceCutter/girvan-newman. [another](https://github.com/ServiceCutter/girvan-newman)
- Implement a community detection algorithm using a divisive hierarchical clustering (Girvan-Newman algorithm) - ZwEin27/Community-Detection. [another](https://github.com/ZwEin27/Community-Detection)
- A Python implementation of Girvan-Newman algorithm - kjahan/community. [another](https://github.com/kjahan/community)
- Code related to blog posts on my Medium page. [Node2vec](https://github.com/eliorc/Medium/blob/master/Nod2Vec-FIFA17-Example.ipynb)
- [paper](https://arxiv.org/pdf/1607.00653.pdf)
- [medium1](https://towardsdatascience.medium.com/think-your-data-different-ddc435f70850)
- [medium 2](https://towardsdatascience.medium.com/node2vec-embeddings-for-graph-data-32a866340fef)
- Implementation of the node2vec algorithm. — tutorial — [code](https://github.com/eliorc/node2vec)
- Code related to blog posts on my Medium page. [git code](https://github.com/eliorc/Medium/blob/master/Nod2Vec-FIFA17-Example.ipynb)
- Contribute to aditya-grover/node2vec development by creating an account on GitHub. [original py2 code](https://github.com/aditya-grover/node2vec)
- Contribute to taboola/node2vec-example development by creating an account on GitHub. [taboola code for their medium paper](https://github.com/taboola/node2vec-example/blob/master/node2vec.ipynb)

The same notes are in [Node2vec](../deep-learning/graph-neural-nets.md#node2vec).

7. [Evaluation metrics for community detection](https://stackoverflow.com/questions/28952104/evaluation-metrics-for-community-detection-algorithms)
- [Review for community detection algorithms](https://arxiv.org/pdf/0906.0612.pdf)
- Community detection in graphs. Community detection in graphs. — [paper](https://arxiv.org/abs/0906.0612)
- Community structure. Community structure - Wikipedia. [Term: community structure](https://en.wikipedia.org/wiki/Community_structure#Algorithms_for_finding_communities)
- Modularity (networks. Modularity (networks - Wikipedia. [Term: modularity of networks](https://en.wikipedia.org/wiki/Modularity_(networks)
11. [Unread paper](http://science.sciencemag.org/content/328/5980/876)
- Multiplex is a set of graphs on the same vertex set, i.e. [Unread comparison of community detection algos](https://arxiv.org/abs/1406.2205)
13. [Clustering adjacency matrices](https://stats.stackexchange.com/questions/125295/the-best-way-for-clustering-an-adjacency-matrix)
- Spectral Clustering: A quick overview – calculated | content. [Spectral-clustering](https://calculatedcontent.com/2012/10/09/spectral-clustering/)
15. [Finding natural groups in undirected graphs](https://stats.stackexchange.com/questions/142297/finding-natural-groups-clusters-in-an-undirected-graph-over-several-undirect)
- A curated list of community detection research papers with implementations. [Awesome community detection on github](https://github.com/benedekrozemberczki/awesome-community-detection)
- This is the manual for Neo4j Graph Data Science library version 2026.09. [Various algorithms](https://neo4j.com/docs/graph-algorithms/current/algorithms/closeness-centrality/)

### Centrality algorithms

This subsection lists Neo4j centrality algorithms.

[5. Centrality algorithms](https://neo4j.com/docs/graph-algorithms/current/algorithms/centrality/)

- This section describes the PageRank algorithm in the Neo4j Graph Data Science library. [5.1. The PageRank algorithm](https://neo4j.com/docs/graph-algorithms/current/algorithms/page-rank/)
- This is the manual for Neo4j Graph Data Science library version 2026.09. [5.2. The Betweenness Centrality algorithm](https://neo4j.com/docs/graph-algorithms/current/algorithms/betweenness-centrality/)
- This is the manual for Neo4j Graph Data Science library version 2026.09. [5.3. The Closeness Centrality algorithm](https://neo4j.com/docs/graph-algorithms/current/algorithms/closeness-centrality/)
- This is the manual for Neo4j Graph Data Science library version 2026.09. [5.4. The Degree Centrality algorithm](https://neo4j.com/docs/graph-algorithms/current/algorithms/degree-centrality/)

### Community detection algorithms

This subsection lists Neo4j community detection algorithms.

[6. Community detection algorithms](https://neo4j.com/docs/graph-algorithms/current/algorithms/community/)

- This section describes the Louvain algorithm in the Neo4j Graph Data Science library. [6.1. The Louvain algorithm](https://neo4j.com/docs/graph-algorithms/current/algorithms/louvain/)
- This section describes the Label Propagation algorithm in the Neo4j Graph Data Science library. [6.2. The Label Propagation algorithm](https://neo4j.com/docs/graph-algorithms/current/algorithms/label-propagation/)
- This is the manual for Neo4j Graph Data Science library version 2026.09. [6.3. The Connected Components algorithm](https://neo4j.com/docs/graph-algorithms/current/algorithms/connected-components/)

### Experimental algorithms

This subsection lists Neo4j experimental graph algorithms.

[7. Experimental algorithms](https://neo4j.com/docs/graph-algorithms/current/experimental-algorithms/)

- This is the manual for Neo4j Graph Data Science library version 2026.09. [7.1. Procedures](https://neo4j.com/docs/graph-algorithms/current/experimental-procedures/)
- This is the manual for Neo4j Graph Data Science library version 2026.09. [7.2. Centrality algorithms](https://neo4j.com/docs/graph-algorithms/current/experimental-algorithms/centrality/)
- This is the manual for Neo4j Graph Data Science library version 2026.09. [7.3. Community detection algorithms](https://neo4j.com/docs/graph-algorithms/current/experimental-algorithms/community/)
- This is the manual for Neo4j Graph Data Science library version 2026.09. [7.4. Path finding algorithms](https://neo4j.com/docs/graph-algorithms/current/experimental-algorithms/pathfinding/)
- This is the manual for Neo4j Graph Data Science library version 2026.09. [7.5. Similarity algorithms](https://neo4j.com/docs/graph-algorithms/current/experimental-algorithms/similarity/)
- This is the manual for Neo4j Graph Data Science library version 2026.09. [7.6. Link Prediction algorithms](https://neo4j.com/docs/graph-algorithms/current/experimental-algorithms/linkprediction/)
- This is the manual for Neo4j Graph Data Science library version 2026.09. [7.7. Preprocessing functions and](https://neo4j.com/docs/graph-algorithms/current/experimental-algorithms/preprocessing/)

## Graph/GNN courses

This section lists courses on machine learning with graphs and geometric deep learning.

The same notes are in [GNN courses](../deep-learning/graph-neural-nets.md#gnn-courses).

- CS224W | Home. CS224W | Home. [Machine learning with graphs by Stanford](http://web.stanford.edu/class/cs224w/)
- Grids, Groups, Graphs, Geodesics, and Gauges. [Graph deep learning course](https://geometricdeeplearning.com/lectures/)
- ICLR 2021 Keynote - "Geometric Deep Learning: The Erlangen Programme of ML" - M Bronstein, by Michael Bronstein. — graphs, sets, groups, GNNs. [YouTube](https://www.youtube.com/watch?v=w6Pw4MOzMuo)

## Graph Tools

This section lists Python libraries for graph manipulation and analysis.

1. [Graph-tool](https://graph-tool.skewed.de/) is an efficient Python module for manipulation and statistical analysis of graphs.
2. [NetworkX](https://github.com/networkx/networkx) is a Python package for the creation, manipulation, and study of the structure, dynamics, and functions of complex networks.
