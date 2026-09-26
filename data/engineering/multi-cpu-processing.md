# Multi CPU Processing

One process is not enough once NumPy and Pandas jobs get large, so this page is how those same libraries use more than one CPU.
The short list moves from NumPy and Pandas on more than one process, through Dask, then async.
The same notes are in [Pandas](data-science-tools.md#pandas).

This group is NumPy and Pandas on more than one process.

[Numpy](https://gitlab.com/tenzing/shared-array) on multi process, and [how to use it.](https://medium.com/analytics-vidhya/multiprocessing-for-data-scientists-in-python-427b2ff93af1)

- A simple and efficient tool to parallelize Pandas operations on all available CPUs - nalepae/pandarallel. [Pandas on multi process](https://github.com/nalepae/pandarallel)

After the process notes, this group is Dask and its walkthroughs.

- Dask — Dask documentation. Dask — Dask documentation. [Dask](https://docs.dask.org/en/latest/)
- - youtube [intros](https://www.youtube.com/channel/UCj9eavqmvwaCyKhIlu2GaoA)


 - Dask Dashboard walkthrough, by Matthew Rocklin. Diagnostic [dashboards](https://www.youtube.com/watch?v=N_GqzcuGLCY)
 - distributed scikit learn, by Tom Augspurger. [Distributed sklearn](https://www.youtube.com/watch?v=5Zf6DQaf7jk)
3. Dask vs swifter vs vectorize
 1. Dask is dask
 2. Swifter will attempt to understand if dask or pandas apply should be used, looks like its using multi cpu so it may not be just using dask on the backend?
 3. Vectorize is just another option
- Speeding up NLTK with parallel processing | WZB Data Science Blog. [Multi process cpu example](https://datascience.blog.wzb.eu/2017/06/19/speeding-up-nltk-with-parallel-processing/)
- [Medium on MP, using MP pool, Ray etc.](https://medium.com/distributed-computing-with-ray/how-to-scale-python-multiprocessing-to-a-cluster-with-one-line-of-code-d19f242f60ff)

Closing the list, this group is async across processes, threads, and coroutines.

- [Async (multi process/thread/coroutines/asyncio)](https://medium.com/velotio-perspectives/an-introduction-to-asynchronous-programming-in-python-af0189a88bbb)

## Deprecated links

{% hint style="warning" %}
These links and images no longer work. The original wording is kept here. A same-resource copy, when one was checked, is used above.
{% endhint %}

- Dask vs swifter vs vectorize. This address no longer opens: https://gdcoder.com/speed-up-pandas-apply-function-using-dask-or-swifter-tutorial/
