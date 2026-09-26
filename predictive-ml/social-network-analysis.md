# Social Network Analysis

Graph theory gives the machinery; social network analysis asks what that machinery says about people who are connected to each other. The page starts with definitions and the people who shaped the field, then the core measures, then how traits spread through networks and why that is hard to prove, and ends with the tools and an applied walkthrough.

The same notes are in [Graph Theory](graph-theory.md).

The definition comes first: the [Wiki](https://en.wikipedia.org/wiki/Social_network) entry is the Wikipedia article on the social network. The algorithmic view of the same object is the [Paper: algorithmic approach to social networks](http://web.archive.org/web/20240712194953/http://www.cs.carleton.edu/faculty/dlibenno/papers/thesis/thesis.pdf), kept here through its archived copy. [Steve Borgatti](https://sites.google.com/site/steveborgatti/home) is the home page of one of the people behind social network analysis methods.

From definitions to measures, [Intro to SNA](http://www.orgnet.com/sna.html) is Orgnet's Social Network Analysis: An Introduction, from a network analysis consulting, training, and software firm. It walks centrality, betweenness centrality, network centralization, network reach, network integration, boundary spanners, and peripheral players. Whether more connections can make up for weaker ones is the question in [Social Network Analysis: Can Quantity Compensate for Quality?](https://33bits.wordpress.com/2009/02/15/social-network-analysis-can-quantity-substitute-for-quality/) on 33 Bits of Entropy.

Measures describe a network; the harder question is why connected people end up alike. [Nicholas Christakis](http://web.archive.org/web/20110603105212/http://www.wjh.harvard.edu/soc/faculty/christakis/) of Harvard and [James Fowler](http://web.archive.org/web/20150810035527/http://jhfowler.ucsd.edu/) of UC San Diego have produced a series of ground-breaking papers analyzing the spread of various traits in social networks: [obesity](http://content.nejm.org/cgi/content/full/357/4/370), [smoking](http://content.nejm.org/cgi/content/full/358/21/2249), [happiness](http://www.bmj.com/cgi/content/full/337/dec04_2/a2338), and most recently, in collaboration with John Cacioppo, [loneliness](http://papers.ssrn.com/sol3/papers.cfm?abstract_id=1319108). The Christakis-Fowler collaboration has now become [well-known](http://web.archive.org/web/20170808104848/http://jhfowler.ucsd.edu/science_friendship_as_a_health_factor.pdf), but from a technical perspective, what was special about their work?

It turns out that they found a way to distinguish between the three reasons why people who are related in a social network are similar to each other. Homophily is the tendency of people to seek others who are alike. For example, most of us restrict our dates to smokers or non-smokers, mirroring our own behavior. Confounding is the phenomenon of related individuals developing a trait because of a (shared) environmental circumstance. For example, people living right next to a McDonald’s might all gradually become obese. Induction is the process of one individual passing a trait or behavior on to their friends, whether by active encouragement or by setting an example.

To compute any of these measures on real data, [Networkx](https://networkx.org/documentation/networkx-1.10/reference/algorithms.html) is the algorithms reference — Centrality is just a fraction of the algorithms contained in networkx, as the figure of its algorithm list shows.

<figure><img src="../.gitbook/assets/gimg-b4faaaed3b79.png" alt=""><figcaption><p>Networkx algorithms.</p><p>Credit: <a href="https://lh3.googleusercontent.com/Z2U_f5O_A407pAxkfZzNLMDjm0LZbFa4bDs2qddvSE2HQ-UbaXHAMRAylOhM7AgblncrxGKHzFvT31O96jKfJ2QgxHK7ntItXsbOxEdlt8eL1HlLUKvvo1tG6kT-txQuxMyAYEif">copied from the original hosted image</a>.</p></figcaption></figure>

The applied walkthrough that ties theory to that code is [Social Network analysis from theory to applications](https://towardsdatascience.medium.com/social-network-analysis-from-theory-to-applications-with-python-d12e9a34c2c7), which introduces the theory of social networks with a short introduction to graph theory and information spread, then dives into Python code with NetworkX that constructs social networks from real datasets. It is by [Dima Goldenberg](https://www.linkedin.com/in/dimgold/), a machine learning group leader at Booking.com.

## Deprecated links

{% hint style="warning" %}
These links and images no longer work. The original wording is kept here. A same-resource copy, when one was checked, is used above.
{% endhint %}

- Paper: algorithmic approach to social networks. This address no longer opens: http://www.cs.carleton.edu/faculty/dlibenno/papers/thesis/thesis.pdf
- Nicholas Christakis. This address no longer opens: http://www.wjh.harvard.edu/soc/faculty/christakis/
- James Fowler. This address no longer opens: http://jhfowler.ucsd.edu/
- well-known (science friendship as a health factor PDF). This address no longer opens: http://jhfowler.ucsd.edu/science_friendship_as_a_health_factor.pdf
