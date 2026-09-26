# Data Engineering

Data engineering is the job that makes data usable, and it is not the same job as writing product software.
This page starts with what the role is, then Job versus Service, then the CAP theorem notes beside the CAP interview questions.
The same notes are in [Data Engineering Questions & Training](data-engineering-questions.md).

The first thing to get right is that the role has its own shape. [Data engineers are not software developers - a comparison](https://betterprogramming.pub/data-engineering-is-not-software-engineering-af81eb8d3949) is the side-by-side of the two jobs. The role also splits further: [Different Types of data engineers](https://medium.com/coriers/different-types-of-data-engineering-teams-6a1056986d3) is Ben Rogojan on how the data engineer role morphed drastically in a decade, from a time when employers expected data scientists to both calculate eigenvectors and write MapReduce jobs for Hadoop. The teams it separates are data infra, data platform, data engineers, data governance, and BI & analytics.

## Job vs Service

Whichever team a data engineer sits on, the first design choice in an ML pipeline is whether a step runs as a job or as a service. The interview question puts it this way:

Explain the difference and the reason to choose using Job over Service and vice versa. Give an example for a project where you had to make this choice, in the context of ML pipelines and walk through your reasoning.


### [Cap theorem](https://medium.com/data-science/cap-theorem-and-distributed-database-management-systems-5c2be977950e)

A service that keeps state across machines runs into the CAP theorem. The heading links Syed Sadat Nazrul's "CAP Theorem and Distributed Database Management Systems", which starts from the shift away from scaling vertically, with more powerful machines, toward expanding horizontally with more machines doing the same task in parallel. The same notes are in [CAP Theorem](data-engineering.md#cap-theorem).

Two sources push the theorem further. [Cap](https://www.confluent.io/blog/turning-the-database-inside-out-with-apache-samza/) is Confluent's "Turning the database inside-out with Apache Samza", from the company building the foundational platform for data in motion. [Cap is changing](https://www.infoq.com/articles/cap-twelve-years-later-how-the-rules-have-changed/) is InfoQ's "CAP Twelve Years Later: How the 'Rules' Have Changed". The figure below is the CAP triangle, copied from the original hosted image.

<figure><img src="../../ops/.gitbook/assets/gimg-54fdea866648.png" alt=""><figcaption><p>Cap theorem</p><p>Credit: <a href="https://lh6.googleusercontent.com/cDV78UprJnSuEkoqVRRzg9K_a8YlvYAQlJ_YDj6CRMqypYp0BwFkHhErzcMtt8h0LWKd4cPk3ftCpRyLTMxLNNxCNJ6nAUZNoEh0umdNzsAdIt0IUMDBJT_uvdWgD9UxHLpHisiS">copied from the original hosted image.</a></p></figcaption></figure>

## CAP Theorem

With the theorem stated, the interview questions check whether you can apply it. Each of the first six is answered in FullStack.Cafe's "15 CAP Theorem Interview Questions (ANSWERED) For System Design Interview":

1. [What Is CAP Theorem?](https://www.fullstack.cafe/blog/cap-theorem-interview-questions)
2. [Can you 'get around' or 'beat' the CAP Theorem?](https://www.fullstack.cafe/blog/cap-theorem-interview-questions)
3. [Name some types of Consistency patterns](https://www.fullstack.cafe/blog/cap-theorem-interview-questions)
4. [What Do You Mean By High Availability (HA)?](https://www.fullstack.cafe/blog/cap-theorem-interview-questions)
5. [What are A and P in CAP and the difference between them?](https://www.fullstack.cafe/blog/cap-theorem-interview-questions)
6. [What does the CAP Theorem actually say?](https://www.fullstack.cafe/blog/cap-theorem-interview-questions)
7. How it affects real world application (latency is availability in real world)

[Another great resource for CAP questions](https://github.com/henryr/cap-faq) is henryr's The CAP FAQ on GitHub. The diagrams below illustrate the theorem.

<figure><img src="../../ops/.gitbook/assets/3" alt=""><figcaption><p>CAP Theorem</p></figcaption></figure>

<figure><img src="../../ops/.gitbook/assets/4" alt=""><figcaption><p>CAP Theorem</p></figcaption></figure>

<figure><img src="../../ops/.gitbook/assets/5" alt=""><figcaption><p>CAP Theorem</p></figcaption></figure>

<figure><img src="../../ops/.gitbook/assets/6" alt=""><figcaption><p>CAP Theorem</p></figcaption></figure>

<figure><img src="../../ops/.gitbook/assets/7" alt=""><figcaption><p>CAP Theorem</p></figcaption></figure>

## Deprecated links

{% hint style="warning" %}
These links and images no longer work. The original wording is kept here. A same-resource copy, when one was checked, is used above.
{% endhint %}

- Cap theorem. This address no longer opens: https://towardsdatascience.com/cap-theorem-and-distributed-database-management-systems-5c2be977950e
