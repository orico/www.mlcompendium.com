# Data Engineering Questions & Training

Interview loops and training prompts need a bank of questions, not a second textbook. The page starts with the general prompts, then points at larger question banks, and ends with the references the answers rest on, from CAP and ACID to join strategies.
The same notes are in [Data Engineering](data-engineering.md).

## General

The general prompts come first, because they test experience before they test theory. The first eight come from LinkedIn's interviewing resources for data engineers, which cover interview strategies, the best questions for assessing a candidate, and how to improve the interview process; the rest are about BigQuery, Spark, and ETL.

1. [What scales of data have you worked with in the past?](https://business.linkedin.com/talent-solutions/resources/interviewing-talent/data-engineer)
2. [How do you generally work with the departments that make use of your data?](https://business.linkedin.com/talent-solutions/resources/interviewing-talent/data-engineer)
3. [Tell me about a time you had performance issues with an ETL. How did you identify this as a performance issue and how did you fix it?](https://business.linkedin.com/talent-solutions/resources/interviewing-talent/data-engineer)
4. [Describe a time when you found a new use case for an existing database.](https://business.linkedin.com/talent-solutions/resources/interviewing-talent/data-engineer)
5. [Describe the most challenging project you’ve worked on. What was your role?](https://business.linkedin.com/talent-solutions/resources/interviewing-talent/data-engineer)
6. [Think back to a project you’re proud of. What was it that gave you that sense of pride and accomplishment?](https://business.linkedin.com/talent-solutions/resources/interviewing-talent/data-engineer)
7. [What do you consider to be one of the biggest mistakes you’ve ever made in your previous job?](https://business.linkedin.com/talent-solutions/resources/interviewing-talent/data-engineer)
8. [Explain the differences between stream processing and data processing, with one caveat: pretend that I’m not familiar with data at all.](https://business.linkedin.com/talent-solutions/resources/interviewing-talent/data-engineer)
9. What are the considerations when choosing methods of ingesting data to BigQuery?
10. Have you worked with data science teams? What were your responsibilities?
11. What are the considerations of choosing Spark vs BigQuery?
12. What are the differences between ETL and ELT?

For the last question, [1](https://www.guru99.com/etl-vs-elt.html) is Guru99's ETL vs ELT comparison, and [2](https://www.xplenty.com/blog/etl-vs-elt/) is Mark Smallcombe's five critical differences, weighing cost, data type and volume, and other factors that shape the integration method. A third source used to sit here and is kept at the end of the page.

The same notes are in [Data Pipelines](../processing/data-pipelines.md).

The three figures below draw the difference; the first is credited to guru99/david taylor and the third to mark smallcombe.

<figure><img src="../../ops/.gitbook/assets/0" alt=""><figcaption><p>ETL and ELT</p><p>Credit: <a href="https://www.guru99.com/etl-vs-elt.html">guru99/david taylor</a></p></figcaption></figure>

<figure><img src="../../ops/.gitbook/assets/1" alt=""><figcaption><p>ETL and ELT</p></figcaption></figure>

<figure><img src="../../ops/.gitbook/assets/2" alt=""><figcaption><p>ETL and ELT</p><p>Credit: <a href="https://www.xplenty.com/blog/etl-vs-elt/">mark smallcombe</a></p></figcaption></figure>

## More question banks

A dozen prompts is enough for one interview; training needs the larger banks.

The biggest is [More than 2000 questions for data engineers](https://github.com/OBenner/data-engineering-interview-questions), a GitHub repo of more than 2000 data engineer interview questions, shown in the figure below.

<figure><img src="../../ops/.gitbook/assets/11" alt=""><figcaption><p>More than 2000 questions for data engineers</p></figcaption></figure>

[More data engineering questions](https://realpython.com/data-engineer-interview-questions-python/) is Real Python's set of data engineer interview questions answered with Python.

<figure><img src="../../ops/.gitbook/assets/12" alt=""><figcaption><p>More data engineering questions</p></figcaption></figure>

[Even more qs](https://www.softwaretestinghelp.com/data-engineer-interview-questions/) is a list of the most frequently asked data engineer interview questions and answers, for preparing for an upcoming interview.

<figure><img src="../../ops/.gitbook/assets/13" alt=""><figcaption><p>Even more qs</p></figcaption></figure>

## References

Questions need answers, and the references are where the answers come from.

[DS leads](https://docs.google.com/document/d/1gdfJce0p7jx0ptHJt3NE3NIhvzZCXJ5O0V-dvJse0HI/edit) is a data engineer interviews document that opens with the CAP theorem and BigQuery questions, such as the difference in the implementation between partitions and clustering in BQ. For system design, the primer that teaches how to design large-scale systems has both [System design interview q’s with solutions](https://github.com/donnemartin/system-design-primer#system-design-interview-questions-with-solutions) and a [Cap theorem](https://github.com/donnemartin/system-design-primer#system-design-interview-questions-with-solutions) section. [2](https://github.com/henryr/cap-faq) is the CAP FAQ on GitHub. The (which is great) [3](https://github.com/kislerdm/data-engineering-interviews) is the data engineering interviews Q&A by and for the data community, which great isn't complete.

The database guarantees are a three-part series. [ACID](https://bardoloi.com/blog/2017/02/26/db-deep-dive/) starts from how an amazing feat of engineering becomes commonplace and taken for granted, and lays out the ACID guarantees. [CAP](https://bardoloi.com/blog/2017/03/06/cap-theorem/) picks up from the ACID guarantees of traditional RDBMSs like MS SQL Server, Oracle, and MySQL, and shows where they run into scale. [PACLEC](https://bardoloi.com/blog/2017/03/06/pacelc-theorem/) takes the CAP theorem as the framework under most modern distributed databases and adds latency, following Daniel Abadi's critique of how CAP had been applied.

The last question is why the field exists at all. [Why do we need Data engineering?](https://podcastaddict.com/episode/116229803) (podcast) is the Data Engineering Podcast episode on proven patterns for building successful data teams, where many stakeholders with competing goals and many roles make data products hard to deliver.

The Spark join strategies article linked on the platforms page, Join strategies #2 from Towards Data Science, is also at its original address: [https://towardsdatascience.com/strategies-of-spark-join-c0e7b4572bcf](https://towardsdatascience.com/strategies-of-spark-join-c0e7b4572bcf).


## Deprecated links

{% hint style="warning" %}
These links and images no longer work. The original wording is kept here. A same-resource copy, when one was checked, is used above.
{% endhint %}

- 3. This address no longer opens: https://blog.panoply.io/etl-vs-elt-the-difference-is-in-the-how
- Can you explain the parquet file format? This address no longer opens: https://parquet.apache.org/documentation/latest/
