# Data Observability

Pipelines fail quietly unless someone watches freshness, volume, and schema drift. Observability is that watching, done continuously instead of after a dashboard breaks. The page moves from the idea to the tools that do the watching, and then to what to do when an outage happens anyway.
The same notes are in [Data Quality](data-quality.md).

## Tools

Watching data health by hand does not scale, so the idea turns into a set of tools and a cycle for acting on what they find.

[Monte Carlo](https://www.montecarlodata.com/) pitches itself as going beyond data quality to end-to-end data and AI observability for enterprise teams. [BigEye](https://www.bigeye.com/) is the data and AI trust platform for large enterprises, combining data observability, end-to-end lineage, and governance. [Databand](http://web.archive.org/web/20231208161414/https://databand.ai/), read through its archived page, is the proactive observability platform that tries to catch bad data before it impacts the business. Its explainer, [What is data observability](http://web.archive.org/web/20231208161417/https://databand.ai/data-observability/), defines the term as an umbrella of activities and technologies that let you identify, troubleshoot, and resolve data issues in near real-time, and it frames the work as a cycle:

> The DataOps cycle outlines the fundamental activities needed to improve how data is managed within the DataOps workflow. This cycle consists of three distinct stages: Detection, Awareness, and Iteration.

The figure shows those three stages as a loop.

<figure><img src="../../ops/.gitbook/assets/image (30).png" alt=""><figcaption><p>The DataOps cycle</p></figcaption></figure>

For an open-source option, [DQOps](https://dqops.com) is a data quality platform for the whole data platform lifecycle, from profiling new data sources to automating data quality monitoring.


## Outage handling

Detection only helps if the pipeline can recover once something breaks, and the architecture decides how painful that recovery is.

The note is about outage handling and the differences between stream-based processing vs concurrent isolated worker-based processing using

Q: you have a real time stream — what is better? A stream-based processing system, or a worker-based, that can be triggered on different time ranges, in the context of recovery from outage.

The figure below sketches that comparison; the credit is nielsen Ilai Malka.

<figure><img src="../../ops/.gitbook/assets/8" alt=""><figcaption><p>Outage handling</p><p>Credit: nielsen Ilai Malka</p></figcaption></figure>

## Deprecated links

{% hint style="warning" %}
These links and images no longer work. The original wording is kept here. A same-resource copy, when one was checked, is used above.
{% endhint %}

- Databand. This address no longer opens: https://databand.ai
- What is data observability and what is the dataOps cycle. This address no longer opens: https://databand.ai/data-observability/
