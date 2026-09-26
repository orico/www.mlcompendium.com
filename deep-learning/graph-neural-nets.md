# GRAPH NEURAL NETWORKS (GNN)

This page collects notes on graph neural networks, courses, graph convolutional networks, and node-embedding methods such as DeepWalk and Node2vec.
Readers use graph structure ideas from above when moving into GCN and walk-based embeddings.

The same notes are in [Graph Theory](../predictive-ml/graph-theory.md).

1. (amazing) [Why i am luke warm about GNN’s](https://www.singlelunch.com/2020/12/28/why-im-lukewarm-on-graph-neural-networks/) - really good insight to what they do (compressing data, vs adjacy graphs, vs graphs, high dim relations, etc.)
- What components are needed for building learning algorithms that leverage the structure and properties of graphs? (amazing) [Graphical intro to GNNs](https://distill.pub/2021/gnn-intro/)
- Learning on Graphs. Learning on Graphs. [Learning on graphs youtube - uriel singer](https://www.youtube.com/watch?v=snLsWos_1WU&feature=youtu.be)
- Benchmarking Graph Neural Networks | NTU Graph Deep Learning Lab, by Chaitanya Joshi. [Benchmarking GNN’s, methodology, git, the works.](https://graphdeeplearning.github.io/post/benchmarking-gnns/)
- A collection of important graph embedding, classification and representation learning papers with implementations. [Awesome graph classification on github](https://github.com/benedekrozemberczki/awesome-graph-classification)
6. Octavian in medium on graphs, [A really good intro to graph networks, too long too summarize](https://medium.com/octavian-ai/deep-learning-with-knowledge-graphs-3df0b469a61a), clever, mcgraph, regression, classification, embedding on graphs.
7. Application of graph networks
8. Recommender systems using GNN, w2v, pytorch w2v, networkx, sparse matrices, matrix factorization, dictionary optimization, part 1 here (how to find product relations, important: creating negative samples)
- Transformers are Graph Neural Networks. Transformers are GNN, original: [Transformers are graphs, not the typical embedding on a graph, but a more holistic approach to understanding text as a graph.](https://thegradient.pub/transformers-are-graph-neural-networks/)
10. Cnn for graphs
- [Staring with gnn](https://medium.com/octavian-ai/how-to-get-started-with-machine-learning-on-graphs-7f0795c83763)
12. Really good - Basics deep walk and graphsage
13. Application of gnn
14. Michael Bronstein’s Central page for Graph deep learning articles on Medium (worth reading)
15. [GAT graphi attention networks](https://petar-v.com/GAT/), paper, examples - The graph attentional layer utilised throughout these networks is computationally efficient (does not require costly matrix operations, and is parallelizable across all nodes in the graph), allows for (implicitly) assigning different importances to different nodes within a neighborhood while dealing with different sized neighborhoods, and does not depend on knowing the entire graph structure upfront—thus addressing many of the theoretical issues with approaches.
16. Medium on Intro, basics, deep walk, graph sage
17. [Struc2vec](https://leoribeiro.github.io/struc2vec.html), [youtube](https://www.youtube.com/watch?v=lu0xMOO48Xo&embeds_euri=https%3A%2F%2Fleoribeiro.github.io%2F&source_ve_path=MjM4NTE&feature=emb_title): Learning Node Representations from Structural Identity- The _struc2vec_ algorithm learns continuous representations for nodes in any graph. struc2vec captures structural equivalence between nodes.

## GNN courses

This section lists GNN courses after the graph-neural-network overview above.

The same notes are in [Graph/GNN courses](../predictive-ml/graph-theory.md#graphgnn-courses).


- CS224W | Home. CS224W | Home. [machine learning with graphs by Stanford](http://web.stanford.edu/class/cs224w/)
- Grids, Groups, Graphs, Geodesics, and Gauges. [Graph deep learning course](https://geometricdeeplearning.com/lectures/)
- ICLR 2021 Keynote - "Geometric Deep Learning: The Erlangen Programme of ML" - M Bronstein, by Michael Bronstein. - graphs, sets, groups, GNNs. [youtube](https://www.youtube.com/watch?app=desktop&v=w6Pw4MOzMuo)

## Graph Convolutional Networks

This section collects notes on graph convolutional networks, after the GNN overview and courses above.

[Explaination here, with some examples](https://tkipf.github.io/graph-convolutional-networks/)

## Deep walk

This section covers DeepWalk embeddings after the GNN and GCN material above.

The same notes are in [Graph Theory](../predictive-ml/graph-theory.md).


- GitHub - phanein/deepwalk: DeepWalk - Deep Learning for Graphs. [Git](https://github.com/phanein/deepwalk)
- Abstract page for arXiv paper 1403.6652: DeepWalk: Online Learning of Social Representations. [Paper](https://arxiv.org/abs/1403.6652)
3. [Medium](https://medium.com/@_init_/an-illustrated-explanation-of-using-skipgram-to-encode-the-structure-of-a-graph-deepwalk-6220e304d71b) and medium on W2v, deep walk, graph2vec, n2v

## Node2vec

This section covers Node2vec after DeepWalk above.

The same notes are in [Graph Topics](../predictive-ml/graph-theory.md#graph-topics).


- Implementation of the node2vec algorithm. Implementation of the node2vec algorithm. [Git](https://github.com/eliorc/node2vec)
- node2vec. node2vec. [Stanford](https://snap.stanford.edu/node2vec/)
- PyData Tel Aviv Meetup: Node2vec - Elior Cohen. Elior on medium [youtube](https://www.youtube.com/watch?v=828rZgV9t1g)
4. [Paper](https://cs.stanford.edu/~jure/pubs/node2vec-kdd16.pdf)

## Graphsage

This section covers GraphSAGE after Node2vec above.


1. medium

## SDNE - structural deep network embedding

This section covers SDNE after GraphSAGE above.


1. medium

## Diff2vec

This section covers Diff2vec after SDNE above.


- Reference implementation of Diffusion2Vec (Complenet 2018) built on Gensim and NetworkX. [Git](https://github.com/benedekrozemberczki/diff2vec)
2. <figure><img src="../.gitbook/assets/gimg-468e43b3cedf.png" alt=""><figcaption><p>Diff2vec</p><p>Credit: <a href="https://lh6.googleusercontent.com/otaXffQv-FribLSm922jhO-904l0ZHD4QcWRJ0dgc7u4vW0HMP1cGP-QU63ohhJSLiUxpz5DTB9L6DsK1ettM0S1MRg76sZZhEjzezQpTDDrrXI6pnh5B-2aRrA8FxJrAJK_fufn">copied from the original hosted image</a>.</p></figcaption></figure>

## Splitter

This section covers Splitter after Diff2vec above.


, [git](https://github.com/benedekrozemberczki/Splitter), [paper](http://epasto.org/papers/www2019splitter.pdf), “Is a Single Embedding Enough? Learning Node Representations that Capture Multiple Social Contexts”

Recent interest in graph embedding methods has focused on learning a single representation for each node in the graph. But can nodes really be best described by a single vector representation? In this work, we propose a method for learning multiple representations of the nodes in a graph (e.g., the users of a social network). Based on a principled decomposition of the ego-network, each representation encodes the role of the node in a different local community in which the nodes participate. These representations allow for improved reconstruction of the nuanced relationships that occur in the graph a phenomenon that we illustrate through state-of-the-art results on link prediction tasks on a variety of graphs, reducing the error by up to 90%. In addition, we show that these embeddings allow for effective visual analysis of the learned community structure.

<figure><img src="../.gitbook/assets/gimg-6b0987b825c9.png" alt=""><figcaption><p>Splitter</p><p>Credit: <a href="https://lh3.googleusercontent.com/ZWvxCQ72uAo6J-nr2uojE4KYzqOvgm3dzzXSuKlP0nbry-qFhEbQVZIG4om_SPLZpWZti3--aG1a6dYmOMnot--vFx0dnimMZDLz4LrjJQkRgAZY8ZospzEPKA9MrW__We61ylD9">copied from the original hosted image</a>.</p></figcaption></figure>

<figure><img src="../.gitbook/assets/gimg-bf3fc09f60ea.png" alt=""><figcaption><p>Splitter</p><p>Credit: <a href="https://lh5.googleusercontent.com/asBPQZ90fcBXUYlz3tT2uV2LbELCjHVm56nhjbvRFuW7UXFBDX8fy353dF_6_OFGHo7ioBmFOl5wwxsyfSJHhA2LIOS0LkOTIdI23WnTjHFIf-PFdr6tp5RG_GaJF7BACv2RrJcK">copied from the original hosted image</a>.</p></figcaption></figure>

- The TensorFlow reference implementation of 'GEMSEC: Graph Embedding with Self Clustering' (ASONAM 2019). [Self clustering graph embeddings](https://github.com/benedekrozemberczki/GEMSEC)

<figure><img src="../.gitbook/assets/gimg-95f08e6a421d.png" alt=""><figcaption><p>Splitter</p><p>Credit: <a href="https://lh5.googleusercontent.com/xLcNkor6PpkcSUl1sW9Ws36NxIrNr9kmdoBuhlPYnfCKlrC7zkaJwNIlSlIBDiXvL9OPi62lQ8q3ZA6oLXr_pJfUJvUTmelHnEy7z2hivhQJxQN4Ppz8ZRCErtlLQzROyIoyZaV-">copied from the original hosted image</a>.</p></figcaption></figure>

17. [Walklets](https://github.com/benedekrozemberczki/walklets), similar to deep walk with node skips. - lots of improvements, works in scale due to lower size representations, improves results, etc.

Nodevectors

[Git](https://github.com/VHRanger/nodevectors), The fastest network node embeddings in the west<figure><img src="../.gitbook/assets/gimg-dd4cbe5489bc.png" alt=""><figcaption><p>Splitter</p><p>Credit: <a href="https://lh3.googleusercontent.com/DwKfPhonL4At5xRePfv77SdSDjSZBYo_Z0Qm1hAFNpLLEYtiGMQhN8QPLO_5tNRr0NYvg3JRyYEECOUhjJkR6sK77k0M-Z1VVYcEwbBLU7cLqjlVN41IV5nGPt1yX8kYP-NlrqO9">copied from the original hosted image</a>.</p></figcaption></figure>

- Towards Data Science: a-gentle-introduction-to-graph-neural-network-basics-deepwalk-and-graphsage-db5d540d50b3. [https://towardsdatascience.com/a-gentle-introduction-to-graph-neural-network-basics-deepwalk-and-graphsage-db5d540d50b3](https://towardsdatascience.com/a-gentle-introduction-to-graph-neural-network-basics-deepwalk-and-graphsage-db5d540d50b3)

## Deprecated links

{% hint style="warning" %}
These links and images no longer work. The original wording is kept here. A same-resource copy, when one was checked, is used above.
{% endhint %}
- Towards Data Science: home. This address no longer opens: https://towardsdatascience.com/graph-deep-learning/home
- Towards Data Science: graph-embeddings-the-summary-cc6075aba007. This address no longer opens: https://towardsdatascience.com/graph-embeddings-the-summary-cc6075aba007
- Towards Data Science: how-to-do-deep-learning-on-graphs-with-graph-convolutional-networks-62acf5b143d0. This address no longer opens: https://towardsdatascience.com/how-to-do-deep-learning-on-graphs-with-graph-convolutional-networks-62acf5b143d0
- Towards Data Science: https-medium-com-aishwaryajadhav-applications-of-graph-neural-networks-1420576be574. This address no longer opens: https://towardsdatascience.com/https-medium-com-aishwaryajadhav-applications-of-graph-neural-networks-1420576be574
- Towards Data Science: node2vec-embeddings-for-graph-data-32a866340fef. This address no longer opens: https://towardsdatascience.com/node2vec-embeddings-for-graph-data-32a866340fef
- Towards Data Science: recommender-systems-applying-graph-and-nlp-techniques-619dbedd9ecc. This address no longer opens: https://towardsdatascience.com/recommender-systems-applying-graph-and-nlp-techniques-619dbedd9ecc
- Towards Data Science: transformers-are-graph-neural-networks-bca9f75412aa. This address no longer opens: https://towardsdatascience.com/transformers-are-graph-neural-networks-bca9f75412aa
