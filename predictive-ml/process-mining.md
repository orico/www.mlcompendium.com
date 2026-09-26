# Process Mining

This page explains process discovery from logs, conformance checking, and related tools. It is for recovering the real process from event logs when the documented process is incomplete or ignored.

Processes were usually manual, giving trust in people following them, e.g. Process that was defined by the company, and no automation. Process mining identifies the process from the logs (**process discovery**).

1. Modeling done with [BPMN](https://www.bpmn.org/) business process model notation language (UML) (DAG), i.e., static, boxes & arrows vs [Petri net](https://en.wikipedia.org/wiki/Petri_net) [YouTube](https://www.youtube.com/watch?v=EmYVZuczJ6k), i.e., dynamic, token based, which allows simulations.
2. Conformance checking — a comparison the real process and the discovered.
 1. if you do not do certain parts in the process you are not compliant.
 2. for example to find out whether people taking shortcuts? optimizing the process without knowing.
3. can be used for offline vs real time process bug alerting
4. XES

### Tutorials

This section lists healthcare process-mining tutorials and a YouTube playlist.

- [https://medium.com/@c3_62722/process-mining-with-python-tutorial-a-healthcare-application-part-1-ae02027a050](https://medium.com/@c3_62722/process-mining-with-python-tutorial-a-healthcare-application-part-1-ae02027a050)
- [https://medium.com/@c3_62722/process-mining-with-python-tutorial-a-healthcare-application-part-2-4cf57053421f](https://medium.com/@c3_62722/process-mining-with-python-tutorial-a-healthcare-application-part-2-4cf57053421f)
- [https://medium.com/@c3_62722/process-mining-with-python-tutorial-a-healthcare-application-part-3-cc9af986c122](https://medium.com/@c3_62722/process-mining-with-python-tutorial-a-healthcare-application-part-3-cc9af986c122)
- [https://medium.com/@c3_62722/process-mining-with-python-tutorial-a-healthcare-application-part-4-912286ee51b](https://medium.com/@c3_62722/process-mining-with-python-tutorial-a-healthcare-application-part-4-912286ee51b)
5. YouTube
 - pm4py tutorials - tutorial #1: What is Process Mining, by Process Mining for Python. [https://www.youtube.com/watch?v=XLHtvt36g6U&list=PLkWuoFn9UEb5l41T4CMKPYHyRcL5ojI9Z](https://www.youtube.com/watch?v=XLHtvt36g6U&list=PLkWuoFn9UEb5l41T4CMKPYHyRcL5ojI9Z)

### Tools

This section lists process-aware services, miners, and pm4py / Celonis / IBM products.

1. services need to be process-aware, i.e. send standardized logs — [https://www.celonis.com/](https://www.celonis.com/)
2. Algorithms — can deal with parallelism
 - Redirecting... Redirecting... [Alpha miner](http://mlwiki.org/index.php/Alpha_Algorithm)
 - Redirecting... Redirecting... [http://mlwiki.org/index.php/Alpha_Algorithm](http://mlwiki.org/index.php/Alpha_Algorithm)
 2. Inductive miner
- Process Intelligence Solutions GmbH (P.I.S.). [https://processintelligence.solutions/pm4py](https://processintelligence.solutions/pm4py)
- The Celonis Platform gives Enterprise AI the operational context it needs to succeed, so you can transform your operations and drive value. [https://www.celonis.com/?](https://www.celonis.com/?)
- IBM Process Mining helps customers to extract process data from business, identify automation opportunities, prioritize by impact, and fast-track implementation. [https://www.ibm.com/products/process-mining](https://www.ibm.com/products/process-mining)
- Video series on the fundamentals of process mining. [https://www.celonis.com/wils-process-mining-class/?](https://www.celonis.com/wils-process-mining-class/?)

## Deprecated links

{% hint style="warning" %}
These links and images no longer work. The original wording is kept here. A same-resource copy, when one was checked, is used above.
{% endhint %}

- https://pm4py.fit.fraunhofer.de/static/assets/api/2.7.5.1/getting_started.html#understanding-process-mining. This address no longer opens: https://pm4py.fit.fraunhofer.de/static/assets/api/2.7.5.1/getting_started.html#understanding-process-mining
- https://pm4py.fit.fraunhofer.de/. This address no longer opens: https://pm4py.fit.fraunhofer.de/
