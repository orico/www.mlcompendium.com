# Multi CPU Processing

One process is not enough once NumPy and Pandas jobs get large, so this page is how those same libraries use more than one CPU.
It moves from NumPy and Pandas on more than one process, through Dask, then async.
The same notes are in [Pandas](data-science-tools.md#pandas).

The first step is to keep the libraries you already use and give them more processes. [Numpy](https://gitlab.com/tenzing/shared-array) on multi process, and [how to use it.](https://medium.com/analytics-vidhya/multiprocessing-for-data-scientists-in-python-427b2ff93af1) are the SharedArray project on GitLab and Sebastian Theiler's multiprocessing for data scientists in Python. For dataframes, [Pandas on multi process](https://github.com/nalepae/pandarallel) is pandarallel, a simple and efficient tool to parallelize Pandas operations on all available CPUs.

When one machine's processes are still not enough, Dask is the library that schedules the work. [Dask](https://docs.dask.org/en/latest/) is the Dask documentation, and the Dask youtube channel has the [intros](https://www.youtube.com/channel/UCj9eavqmvwaCyKhIlu2GaoA). Diagnostic [dashboards](https://www.youtube.com/watch?v=N_GqzcuGLCY) is the Dask Dashboard walkthrough by Matthew Rocklin, and [Distributed sklearn](https://www.youtube.com/watch?v=5Zf6DQaf7jk) is distributed scikit learn by Tom Augspurger.

Dask is not the only way to speed up an apply, and the comparison of Dask vs swifter vs vectorize (its source is kept at the end of the page) comes down to three notes:

 1. Dask is dask
 2. Swifter will attempt to understand if dask or pandas apply should be used, looks like its using multi cpu so it may not be just using dask on the backend?
 3. Vectorize is just another option

Outside dataframes, the same idea applies to any Python job. [Multi process cpu example](https://datascience.blog.wzb.eu/2017/06/19/speeding-up-nltk-with-parallel-processing/) is the WZB Data Science Blog on speeding up NLTK with parallel processing. [Medium on MP, using MP pool, Ray etc.](https://medium.com/distributed-computing-with-ray/how-to-scale-python-multiprocessing-to-a-cluster-with-one-line-of-code-d19f242f60ff) is Edward Oakes showing how `multiprocessing.Pool` can be scaled from a single machine to a cluster.

Processes are one answer; the last one is not to wait at all. [Async (multi process/thread/coroutines/asyncio)](https://medium.com/velotio-perspectives/an-introduction-to-asynchronous-programming-in-python-af0189a88bbb) introduces asynchronous programming, where a unit of work runs separately from the main thread and notifies it when it completes or fails.

## Deprecated links

{% hint style="warning" %}
These links and images no longer work. The original wording is kept here. A same-resource copy, when one was checked, is used above.
{% endhint %}

- Dask vs swifter vs vectorize. This address no longer opens: https://gdcoder.com/speed-up-pandas-apply-function-using-dask-or-swifter-tutorial/
