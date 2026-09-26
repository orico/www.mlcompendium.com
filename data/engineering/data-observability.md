# Data Observability

Pipelines fail quietly unless someone watches freshness, volume, and schema drift.
This page is the observability idea, then the tools, then outage handling.
The same notes are in [Data Quality](data-quality.md).



## Tools

After the idea, this section lists tools for watching data health and the DataOps cycle.

- Go beyond data quality to unlock true AI observability with the only end-to-end data and AI observability platform for enterprise teams. [Monte Carlo](https://www.montecarlodata.com/)
- Bigeye is the data and AI trust platform for large enterprises. [BigEye](https://www.bigeye.com/)
- Databand is the only proactive data observability platform that catches bad data before it impacts your business. [Databand](http://web.archive.org/web/20231208161414/https://databand.ai/)
 - Data observability covers an umbrella of activities and technologies that allow you to identify, troubleshoot, and resolve data issues in near real-time. [What is data observability](http://web.archive.org/web/20231208161417/https://databand.ai/data-observability/)

> The DataOps cycle outlines the fundamental activities needed to improve how data is managed within the DataOps workflow. This cycle consists of three distinct stages: Detection, Awareness, and Iteration.

<figure><img src="../../ops/.gitbook/assets/image (30).png" alt=""><figcaption><p>The DataOps cycle</p></figcaption></figure>

- Open-source data quality platform for the whole data platform lifecycle, from profiling new data sources to automating data quality monitoring. [DQOps](https://dqops.com)


## Outage handling

After the tools, this section is outage handling notes.

Outage handling and the differences between stream-based processing vs concurrent isolated worker-based processing using

Q: you have a real time stream — what is better? A stream-based processing system, or a worker-based, that can be triggered on different time ranges, in the context of recovery from outage.

<figure><img src="../../ops/.gitbook/assets/8" alt=""><figcaption><p>Outage handling</p><p>Credit: nielsen Ilai Malka</p></figcaption></figure>

## Deprecated links

{% hint style="warning" %}
These links and images no longer work. The original wording is kept here. A same-resource copy, when one was checked, is used above.
{% endhint %}

- Databand. This address no longer opens: https://databand.ai
- What is data observability and what is the dataOps cycle. This address no longer opens: https://databand.ai/data-observability/
