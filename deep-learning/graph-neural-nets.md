# GRAPH NEURAL NETWORKS (GNN)

Some data is not a table or a sequence but a graph, and the question is how a network can learn from the structure and properties of that graph. The page starts with what GNNs actually do and where they fall short, then the courses, then graph convolutional networks, and then the node-embedding methods one after another: DeepWalk, Node2vec, GraphSAGE, SDNE, Diff2vec, and Splitter with its relatives.

The same notes are in [Graph Theory](../predictive-ml/graph-theory.md).

The first thing to read is the skeptical one. (amazing) [Why i am luke warm about GNN’s](https://www.singlelunch.com/2020/12/28/why-im-lukewarm-on-graph-neural-networks/) - really good insight to what they do (compressing data, vs adjacy graphs, vs graphs, high dim relations, etc.). With that caution in mind, the (amazing) [Graphical intro to GNNs](https://distill.pub/2021/gnn-intro/) is Distill's gentle introduction to graph neural networks, built around the question of which components are needed for learning algorithms that leverage the structure and properties of graphs. For a talk instead of an article, [Learning on graphs youtube - uriel singer](https://www.youtube.com/watch?v=snLsWos_1WU&feature=youtu.be) is Uriel Singer's lecture on learning on graphs.

Once the idea is clear, the next question is how to compare models fairly. [Benchmarking GNN’s, methodology, git, the works.](https://graphdeeplearning.github.io/post/benchmarking-gnns/) is the NTU Graph Deep Learning Lab post by Chaitanya Joshi on the Benchmarking Graph Neural Networks paper, written because GNNs are now used across social sciences, knowledge graphs, chemistry, physics, and neuroscience. The papers themselves are gathered in [Awesome graph classification on github](https://github.com/benedekrozemberczki/awesome-graph-classification), a collection of important graph embedding, classification, and representation learning papers with implementations.

Octavian in medium on graphs wrote [A really good intro to graph networks, too long too summarize](https://medium.com/octavian-ai/deep-learning-with-knowledge-graphs-3df0b469a61a); it touches clever, mcgraph, regression, classification, embedding on graphs. The author's list then moved to the application of graph networks, and to recommender systems using GNN, w2v, pytorch w2v, networkx, sparse matrices, matrix factorization, and dictionary optimization, part 1 here (how to find product relations, important: creating negative samples); those sources are kept at the end of the page.

The graph view also reaches text. Transformers are GNN, original: [Transformers are graphs, not the typical embedding on a graph, but a more holistic approach to understanding text as a graph.](https://thegradient.pub/transformers-are-graph-neural-networks/) is The Gradient's piece "Transformers are Graph Neural Networks". The author's next notes were CNN for graphs and, for a practical start, [Staring with gnn](https://medium.com/octavian-ai/how-to-get-started-with-machine-learning-on-graphs-7f0795c83763), David Mack's guide for research teams that have graph data and want to perform machine learning on it but are not sure where to start. The notes that followed are really good - Basics deep walk and graphsage, application of gnn, Michael Bronstein’s Central page for Graph deep learning articles on Medium (worth reading), and Medium on Intro, basics, deep walk, graph sage; their sources are at the end of the page.

Attention also has a graph form. [GAT graphi attention networks](https://petar-v.com/GAT/) is the paper page with examples. The graph attentional layer utilised throughout these networks is computationally efficient (does not require costly matrix operations, and is parallelizable across all nodes in the graph), allows for (implicitly) assigning different importances to different nodes within a neighborhood while dealing with different sized neighborhoods, and does not depend on knowing the entire graph structure upfront—thus addressing many of the theoretical issues with approaches.

Structure can matter more than neighborhood. [Struc2vec](https://leoribeiro.github.io/struc2vec.html) and its KDD 2017 [youtube](https://www.youtube.com/watch?v=lu0xMOO48Xo&embeds_euri=https%3A%2F%2Fleoribeiro.github.io%2F&source_ve_path=MjM4NTE&feature=emb_title) talk are about Learning Node Representations from Structural Identity: the _struc2vec_ algorithm learns continuous representations for nodes in any graph. struc2vec captures structural equivalence between nodes.

## GNN courses

After the overview, the courses give the same material in order, with lectures to follow.

The same notes are in [Graph/GNN courses](../predictive-ml/graph-theory.md#graphgnn-courses).

The [machine learning with graphs by Stanford](http://web.stanford.edu/class/cs224w/) course is CS224W. The [Graph deep learning course](https://geometricdeeplearning.com/lectures/) is the geometric deep learning course on Grids, Groups, Graphs, Geodesics, and Gauges. Its short version is on [youtube](https://www.youtube.com/watch?app=desktop&v=w6Pw4MOzMuo): the ICLR 2021 keynote by Michael Bronstein, "Geometric Deep Learning: The Erlangen Programme of ML", covering graphs, sets, groups, GNNs.

## Graph Convolutional Networks

The courses lead to the most common GNN layer, the graph convolution. The [Explaination here, with some examples](https://tkipf.github.io/graph-convolutional-networks/) is the post "How powerful are Graph Convolutional Networks?", which starts from the fact that many real-world datasets come as graphs or networks (social networks, knowledge graphs, protein-interaction networks, the World Wide Web) while little attention had gone into generalizing neural networks to them.

## Deep walk

Before convolutions on graphs, the simpler route was to turn a graph into walks and embed them like words.

The same notes are in [Graph Theory](../predictive-ml/graph-theory.md).

The code is the [Git](https://github.com/phanein/deepwalk) repo, DeepWalk - Deep Learning for Graphs, and the [Paper](https://arxiv.org/abs/1403.6652) is the arXiv paper "DeepWalk: Online Learning of Social Representations". For the intuition, [Medium](https://medium.com/@_init_/an-illustrated-explanation-of-using-skipgram-to-encode-the-structure-of-a-graph-deepwalk-6220e304d71b) is the illustrated explanation of using skipgram to encode graph structure, and medium on W2v, deep walk, graph2vec, n2v goes wider.

## Node2vec

Node2vec keeps the walk idea from DeepWalk and changes how the walks are drawn.

The same notes are in [Graph Topics](../predictive-ml/graph-theory.md#graph-topics).

The [Git](https://github.com/eliorc/node2vec) repo is an implementation of the node2vec algorithm, and [Stanford](https://snap.stanford.edu/node2vec/) is the node2vec project page. Elior on medium also gave the talk on [youtube](https://www.youtube.com/watch?v=828rZgV9t1g), the PyData Tel Aviv Meetup session on Node2vec by Elior Cohen. The [Paper](https://cs.stanford.edu/~jure/pubs/node2vec-kdd16.pdf) starts from the problem that prediction tasks over nodes and edges need carefully engineered features, and that representation learning can automate that by learning the features themselves.

## Graphsage

After walk-based embeddings, GraphSAGE is the next method on the list. Its only note is a medium post, kept at the end of the page.

## SDNE - structural deep network embedding

SDNE follows GraphSAGE as another way to embed nodes, here with a deep structural model. Its only note is a medium post, kept at the end of the page.

## Diff2vec

Diff2vec replaces random walks with diffusion. The [Git](https://github.com/benedekrozemberczki/diff2vec) repo is the reference implementation of Diffusion2Vec (Complenet 2018) built on Gensim and NetworkX, and the figure below shows the method.

<figure><img src="../.gitbook/assets/gimg-468e43b3cedf.png" alt=""><figcaption><p>Diff2vec</p><p>Credit: <a href="https://lh6.googleusercontent.com/otaXffQv-FribLSm922jhO-904l0ZHD4QcWRJ0dgc7u4vW0HMP1cGP-QU63ohhJSLiUxpz5DTB9L6DsK1ettM0S1MRg76sZZhEjzezQpTDDrrXI6pnh5B-2aRrA8FxJrAJK_fufn">copied from the original hosted image</a>.</p></figcaption></figure>

## Splitter

Every method so far gives each node one vector. Splitter asks whether that is enough. The [git](https://github.com/benedekrozemberczki/Splitter) repo is a PyTorch implementation of the WWW 2019 work, and the [paper](http://epasto.org/papers/www2019splitter.pdf) is “Is a Single Embedding Enough? Learning Node Representations that Capture Multiple Social Contexts”. Its abstract states the problem and the result:

Recent interest in graph embedding methods has focused on learning a single representation for each node in the graph. But can nodes really be best described by a single vector representation? In this work, we propose a method for learning multiple representations of the nodes in a graph (e.g., the users of a social network). Based on a principled decomposition of the ego-network, each representation encodes the role of the node in a different local community in which the nodes participate. These representations allow for improved reconstruction of the nuanced relationships that occur in the graph a phenomenon that we illustrate through state-of-the-art results on link prediction tasks on a variety of graphs, reducing the error by up to 90%. In addition, we show that these embeddings allow for effective visual analysis of the learned community structure.

The two figures below show the Splitter idea.

<figure><img src="../.gitbook/assets/gimg-6b0987b825c9.png" alt=""><figcaption><p>Splitter</p><p>Credit: <a href="https://lh3.googleusercontent.com/ZWvxCQ72uAo6J-nr2uojE4KYzqOvgm3dzzXSuKlP0nbry-qFhEbQVZIG4om_SPLZpWZti3--aG1a6dYmOMnot--vFx0dnimMZDLz4LrjJQkRgAZY8ZospzEPKA9MrW__We61ylD9">copied from the original hosted image</a>.</p></figcaption></figure>

<figure><img src="../.gitbook/assets/gimg-bf3fc09f60ea.png" alt=""><figcaption><p>Splitter</p><p>Credit: <a href="https://lh5.googleusercontent.com/asBPQZ90fcBXUYlz3tT2uV2LbELCjHVm56nhjbvRFuW7UXFBDX8fy353dF_6_OFGHo7ioBmFOl5wwxsyfSJHhA2LIOS0LkOTIdI23WnTjHFIf-PFdr6tp5RG_GaJF7BACv2RrJcK">copied from the original hosted image</a>.</p></figcaption></figure>

Communities can also be learned together with the embedding. [Self clustering graph embeddings](https://github.com/benedekrozemberczki/GEMSEC) is the TensorFlow reference implementation of 'GEMSEC: Graph Embedding with Self Clustering' (ASONAM 2019), shown in the next figure.

<figure><img src="../.gitbook/assets/gimg-95f08e6a421d.png" alt=""><figcaption><p>Splitter</p><p>Credit: <a href="https://lh5.googleusercontent.com/xLcNkor6PpkcSUl1sW9Ws36NxIrNr9kmdoBuhlPYnfCKlrC7zkaJwNIlSlIBDiXvL9OPi62lQ8q3ZA6oLXr_pJfUJvUTmelHnEy7z2hivhQJxQN4Ppz8ZRCErtlLQzROyIoyZaV-">copied from the original hosted image</a>.</p></figcaption></figure>

Back on the walk side, [Walklets](https://github.com/benedekrozemberczki/walklets) is similar to deep walk with node skips - lots of improvements, works in scale due to lower size representations, improves results, etc.

When speed is the constraint, Nodevectors is the library: its [Git](https://github.com/VHRanger/nodevectors) repo calls it The fastest network node embeddings in the west.

<figure><img src="../.gitbook/assets/gimg-dd4cbe5489bc.png" alt=""><figcaption><p>Splitter</p><p>Credit: <a href="https://lh3.googleusercontent.com/DwKfPhonL4At5xRePfv77SdSDjSZBYo_Z0Qm1hAFNpLLEYtiGMQhN8QPLO_5tNRr0NYvg3JRyYEECOUhjJkR6sK77k0M-Z1VVYcEwbBLU7cLqjlVN41IV5nGPt1yX8kYP-NlrqO9">copied from the original hosted image</a>.</p></figcaption></figure>

To close the loop back to the basics, the Towards Data Science gentle introduction to graph neural network basics, DeepWalk and GraphSAGE is at [https://towardsdatascience.com/a-gentle-introduction-to-graph-neural-network-basics-deepwalk-and-graphsage-db5d540d50b3](https://towardsdatascience.com/a-gentle-introduction-to-graph-neural-network-basics-deepwalk-and-graphsage-db5d540d50b3).

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
