# Process Mining

A company writes down a process, but the documented process is often incomplete or ignored, and the real one only shows up in the event logs. Process mining recovers that real process from the logs. The page starts with how the process is modeled and checked for conformance, then moves to healthcare tutorials and a YouTube playlist, and ends with the tools and miners that do the work.

Processes were usually manual, giving trust in people following them, e.g. Process that was defined by the company, and no automation. Process mining identifies the process from the logs (**process discovery**).

The discovered process needs a notation. Modeling is done with [BPMN](https://www.bpmn.org/), the business process model notation language (UML) (DAG), whose official resource covers BPMN 2.0. BPMN is static, boxes & arrows. The alternative is the [Petri net](https://en.wikipedia.org/wiki/Petri_net), and the Cyber Physics [YouTube](https://www.youtube.com/watch?v=EmYVZuczJ6k) video on Petri nets walks through it. A Petri net is dynamic, token based, which allows simulations.

Once there is a model, conformance checking is a comparison between the real process and the discovered one. If you do not do certain parts in the process you are not compliant. For example, it can find out whether people are taking shortcuts: are they optimizing the process without knowing? The same comparison can be used for offline vs real time process bug alerting. The standardized log format behind all of this is XES.

### Tutorials

The ideas above become concrete in a four-part tutorial series, Process Mining with Python tutorial: A healthcare application. [https://medium.com/@c3_62722/process-mining-with-python-tutorial-a-healthcare-application-part-1-ae02027a050](https://medium.com/@c3_62722/process-mining-with-python-tutorial-a-healthcare-application-part-1-ae02027a050) is Part 1, the first article of the series, which lays out the parts that follow. [https://medium.com/@c3_62722/process-mining-with-python-tutorial-a-healthcare-application-part-2-4cf57053421f](https://medium.com/@c3_62722/process-mining-with-python-tutorial-a-healthcare-application-part-2-4cf57053421f) is Part 2, [https://medium.com/@c3_62722/process-mining-with-python-tutorial-a-healthcare-application-part-3-cc9af986c122](https://medium.com/@c3_62722/process-mining-with-python-tutorial-a-healthcare-application-part-3-cc9af986c122) is Part 3, and [https://medium.com/@c3_62722/process-mining-with-python-tutorial-a-healthcare-application-part-4-912286ee51b](https://medium.com/@c3_62722/process-mining-with-python-tutorial-a-healthcare-application-part-4-912286ee51b) is Part 4.

For video, YouTube has the pm4py tutorials playlist by Process Mining for Python, starting with tutorial #1: What is Process Mining? [https://www.youtube.com/watch?v=XLHtvt36g6U&list=PLkWuoFn9UEb5l41T4CMKPYHyRcL5ojI9Z](https://www.youtube.com/watch?v=XLHtvt36g6U&list=PLkWuoFn9UEb5l41T4CMKPYHyRcL5ojI9Z)

### Tools

The tutorials assume the logs already exist, so the first requirement on tooling is that services need to be process-aware, i.e. send standardized logs, the setting [https://www.celonis.com/](https://www.celonis.com/) is built around.

The algorithms that discover the process from those logs can deal with parallelism. The first is the [Alpha miner](http://mlwiki.org/index.php/Alpha_Algorithm), described on the mlwiki page [http://mlwiki.org/index.php/Alpha_Algorithm](http://mlwiki.org/index.php/Alpha_Algorithm). The second is the Inductive miner.

Those miners ship inside products. Process Intelligence Solutions GmbH (P.I.S.) develops process mining software including PM4Py and PMTk, aimed at helping organizations streamline their processes: [https://processintelligence.solutions/pm4py](https://processintelligence.solutions/pm4py). Celonis is the commercial platform: [https://www.celonis.com/?](https://www.celonis.com/?). IBM Process Mining extracts process data from the business, identifies automation opportunities, prioritizes them by impact, and fast-tracks implementation: [https://www.ibm.com/products/process-mining](https://www.ibm.com/products/process-mining). To learn the fundamentals from the same vendor, Celonis has a video series on the fundamentals of process mining: [https://www.celonis.com/wils-process-mining-class/?](https://www.celonis.com/wils-process-mining-class/?)

## Deprecated links

{% hint style="warning" %}
These links and images no longer work. The original wording is kept here. A same-resource copy, when one was checked, is used above.
{% endhint %}

- https://pm4py.fit.fraunhofer.de/static/assets/api/2.7.5.1/getting_started.html#understanding-process-mining. This address no longer opens: https://pm4py.fit.fraunhofer.de/static/assets/api/2.7.5.1/getting_started.html#understanding-process-mining
- https://pm4py.fit.fraunhofer.de/. This address no longer opens: https://pm4py.fit.fraunhofer.de/
