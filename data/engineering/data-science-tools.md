# Data Science Tools

The workbench for data science is Python and the libraries around it, plus a notebook and a way to keep environments from colliding.
This page is that short list: Python, async, clean code, virtual environments, Jupyter, SciPy, NumPy, Pandas and EDA, then Git.
## Python

This group is Python OOP, typing, concurrency, and the async, clean-code, and virtual-environment notes under it.

- Improve Your Python: Python Classes and Object Oriented Programming | HackerNoon. [How to use better OOP in python.](https://hackernoon.com/improve-your-python-python-classes-and-object-oriented-programming-d09ff461168d)
- Python's Class Development Toolkit, by Next Day Video. [Best practices programming python classes - a great lecture.](https://www.youtube.com/watch?v=HTLu2DFOdTg)
3. [How to know pip packages size](https://stackoverflow.com/questions/34266159/how-to-see-pip-package-sizes-installed)
- [Python type checking tutorial](https://medium.com/@ageitgey/learn-how-to-use-static-type-checking-in-python-3-6-in-10-minutes-12c86d72677b)
- Python click tutorial shows how to create command 
line interfaces with the click module, by Jan Bodnar. [Import click - command line interface](https://zetcode.com/python/click/)
6. [Concurrency vs Parallelism (great)](https://stackoverflow.com/questions/1050222/what-is-the-difference-between-concurrency-and-parallelism)
- [Async in python](https://medium.com/velotio-perspectives/an-introduction-to-asynchronous-programming-in-python-af0189a88bbb)
8. [Coroutines vs futures](https://stackoverflow.com/questions/34753401/difference-between-coroutine-and-future-task-in-python-3-5)
9. Coroutines generators async wait
10. Intro to concurrent,futures
11. Future task event loop

### Async io

This group under Python is asyncio intros.

1. [Intro](https://realpython.com/lessons/what-asyncio/)
2. [complete](https://realpython.com/async-io-python/)

### Clean code

This group under Python is clean-code-in-Python notes.

- :bathtub: Clean Code concepts adapted for Python. [Clean code in python git](https://github.com/zedr/clean-code-python)
- [About the book](https://medium.com/@m_mcclarty/tech-book-talk-clean-code-in-python-aa2c92c6564f)

### Virtual Environments

This group under Python compares venv, pyenv, pipenv, and Jupyter kernels.

- [stack overflow on pyenv / venv / etc](https://stackoverflow.com/questions/41573587/what-is-the-difference-between-venv-pyvenv-pyenv-virtualenv-virtualenvwrappe)
- [Guide to pyenv & pyenv virtualenv](https://medium.com/swlh/a-guide-to-python-virtual-environments-8af34aa106ac)
- [Managing virtual env with pyenv](https://medium.com/data-science/managing-virtual-environment-with-pyenv-ae6f3fb835f8)
- Just use venv
- [Summary on all the *envs](https://stackoverflow.com/questions/41573587/what-is-the-difference-between-venv-pyvenv-pyenv-virtualenv-virtualenvwrappe)
- [A really good primer on virtual environments](https://realpython.com/python-virtual-environments-a-primer/)
- An Introduction To Venv — Internet Programming with Python. [Introduction to venv](http://cewing.github.io/training.python_web/html/presentations/venv_intro.html)
- Pipenv: Python Development Workflow for Humans — pipenv 2026.8.0 documentation. [Pipenv](https://pipenv.readthedocs.io/en/latest/)
- [A great intro to pipenv](https://realpython.com/pipenv-guide/)
- Use Pipenv to simplify the management of dependencies in your Python projects, by Murtaza Gulamali. [A complementary to pipenv above](https://robots.thoughtbot.com/how-to-manage-your-python-projects-with-pipenv)
- [Comparison between all *env](https://stackoverflow.com/questions/41573587/what-is-the-difference-between-venv-pyvenv-pyenv-virtualenv-virtualenvwrappe)
- pyenv, virtualenv and using them with Jupyter - a make sense tutorial and instructions on how to use all.
- Create isolated Jupyter ipython kernels with pyenv and virtualenv by alfredo motta
- ## Learn how to install a kernelspec to access your Python data science virtual environments within Jupyter Notebook. [Jupyter Notebook in a virtual env](https://medium.com/data-science/jupyter-notebooks-i-getting-started-with-jupyter-notebooks-f529449797d2)

#### PYENV

This part is pyenv install and usage notes.

1. [Installing pyenv](https://bgasparotto.com/install-pyenv-ubuntu-debian)
2. [Intro to pyenv](https://realpython.com/intro-to-pyenv/)
- Pyenv helps us to install, manage and switch between multiple python versions, most commonly done for testing your code across multiple python environments.In this post, we’ll have a look at getting up and running with pyenv. [Pyenv tutorial and finding where it is](https://anil.io/blog/python/pyenv/using-pyenv-to-install-multiple-python-versions-tox/)
- On OSX 10.11.5, that came with Python 2.7.10. [Pyenv override system python on mac](https://github.com/pyenv/pyenv/issues/660)
5. pyenv virtualenv

## Jupyter

After Python, this group is Jupyter, Colab, profiling, and notebook-as-module tooling.

The same notes are in [Docker](../../ai-engineering/devops/full-stack-and-ops/docker.md) and [Docker for data science](../../ai-engineering/mlops/full-stack-and-ops.md#docker-for-data-science).

- Effortlessly build and scale your AI project, backed by world-class GPUs trusted by more than 500K builders. [Cloud GPUS cheap](https://www.paperspace.com/gradient)
- Importing Jupyter Notebooks as Modules — Jupyter Notebook 7.7.0a2 documentation. [Importing a notebook as a module](http://jupyter-notebook.readthedocs.io/en/latest/examples/Notebook/Importing%20Notebooks.html)
- Important [colaboratory commands for jupytr](https://medium.com/deep-learning-turkey/google-colab-free-gpu-tutorial-e113627b9f5d)
- Timing and profiling in Jupyter
- ([Debugging in Jupyter, how?)](https://kawahara.ca/how-to-debug-a-jupyter-ipython-notebook/) - put a one liner before the code and query the variables inside a function.
- Jupyter Notebook is a powerful tool for data analysis, by Celeste Grupman. [28 tips n tricks for jupyter](https://www.dataquest.io/blog/jupyter-notebook-tips-tricks-shortcuts/)
- Jupyter notebooks as a module
 - Create delightful software with Jupyter Notebooks. [Nbdev](https://github.com/fastai/nbdev)
 - Write, test, document, and distribute software packages and technical articles — all in one place, your notebook. on [fast.ai](https://nbdev.fast.ai/)
 - Jupyter Notebooks as Markdown Documents, Julia, Python or R scripts - jupytext/jupytext. [jupytext](https://github.com/mwouts/jupytext)
- Virtual environments in jupyter
 1. Enter your project directory
 2. $ python -m venv projectname
 3. $ source projectname/bin/activate
 4. (venv) $ pip install ipykernel
 5. (venv) $ ipython kernel install --user --name=projectname
 6. Run jupyter notebook * (not entirely sure how this works out when you have multiple notebook processes, can we just reuse the same server?)
 7. Connect to the new server at port 8889
- Redirecting…. Redirecting…. [Virtual env with jupyter](https://janakiev.com/til/jupyter-virtual-envs/)

([how does reshape work?)](http://anie.me/numpy-reshape-transpose-theano-dimshuffle/) - a shape of (2,4,6) is like a tree of 2->4 and each one has more leaves 4->6.

As far as i can tell, reshape effectively flattens the tree and divide it again to a new tree, but the total amount of inputs needs to stay the same. 2*4*6 = 4*2*3*2 for example

code:

```python
import numpy
rng = numpy.random.RandomState(234)
a = rng.randn(2,3,10)
print(a.shape)
print(a)
b = numpy.reshape(a, (3,5,-1))
print(b.shape)
print(b)
```

*** A tutorial for [Google Colaboratory - free Tesla K80 with Jup-notebook](https://www.kdnuggets.com/2018/02/google-colab-free-gpu-tutorial-tensorflow-keras-pytorch.html/2)

[Jupyter on Amazon AWS](https://blog.keras.io/running-jupyter-notebooks-on-gpu-on-aws-a-starter-guide.html)

How to add extensions to jupyter: [extensions](https://codeburst.io/jupyter-notebook-tricks-for-data-science-that-enhance-your-efficiency-95f98d3adee4)

[Connecting from COLAB to MS AZURE](https://medium.com/@d.sakryukin/simple-cryptocurrency-trading-data-preparation-in-15-minutes-using-ms-azure-and-google-colab-44872b023d11)

[Streamlit vs. Dash vs. Shiny vs. Voila vs. Flask vs. Jupyter](https://medium.com/data-science/streamlit-vs-dash-vs-shiny-vs-voila-vs-flask-vs-jupyter-24739ab5d569)

## SciPy

Beside Jupyter, this group points at SciPy.

- 2.7. Mathematical optimization: finding minima of functions — Scipy lecture notes. [Optimization problems, a nice tutorial](http://scipy-lectures.org/advanced/mathematical_optimization/)
2. [Minima / maxima](https://stackoverflow.com/questions/4624970/finding-local-maxima-minima-with-numpy-in-a-1d-numpy-array)

## NumPy

After SciPy, this group points at NumPy.

[Using numpy efficiently](https://speakerdeck.com/cournape/using-numpy-efficiently) - explaining why vectors work faster.
[Fast vector calculation, a benchmark](https://medium.com/data-science/data-science-with-python-turn-your-conditional-loops-to-numpy-vectors-9484ff9c622e) between list, map, vectorize. Vectorize wins. The idea is to use vectorize and a function that does something that may involve if conditions on a vector, and do it as fast as possible.

## Pandas

After NumPy, this group is Pandas and exploratory data analysis.

The same notes are in [Multi CPU Processing](multi-cpu-processing.md).

1. [Great introductory tutorial](http://nikgrozev.com/2015/12/27/pandas-in-jupyter-quickstart-and-useful-snippets/#loading_csv_files) about using pandas, loading, loading from zip, seeing the table’s features, accessing rows & columns, boolean operations, calculating on a whole row/column with a simple function and on two columns even, dealing with time/date parsing.
- Visualizing Pandas. Visualizing Pandas. [Visualizing pandas pivoting and reshaping functions by Jay Alammar](http://jalammar.github.io/visualizing-pandas-pivoting-and-reshaping/)
3. [How to beautify pandas dataframe using html display](https://stackoverflow.com/questions/26873127/show-dataframe-as-table-in-ipython-notebook)
4. [Speeding up pandas](https://realpython.com/fast-flexible-pandas/)
5. [The fastest way to select rows by columns, by using masked values](https://stackoverflow.com/questions/17071871/select-rows-from-a-dataframe-based-on-values-in-a-column-in-pandas) (benchmarked):
6. def mask_with_values(df): mask = df['A'].values == 'foo' return df[mask]
7. Parallelism, pools, threads, dask
8. [Accessing dataframe rows, columns and cells](http://pythonhow.com/accessing-dataframe-columns-rows-and-cells/)- by name, by index, by python methods.
- [Looping through pandas](https://medium.com/swlh/how-to-efficiently-loop-through-pandas-dataframe-660e4660125d)
10. How to inject headers into a headless CSV file
11. [Dealing with time series](http://pandas.pydata.org/pandas-docs/stable/timeseries.html) in pandas,
 1. [Create a new column](https://stackoverflow.com/questions/25570147/add-new-column-based-on-boolean-values-in-a-different-column) based on a (boolean or not) column and calculation:
 2. Using python (map)
 3. Using numpy
 4. using a function (not as pretty)
12. Given a DataFrame, the [shift](http://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/)() function can be used to create copies of columns that are pushed forward (rows of NaN values added to the front) or pulled back (rows of NaN values added to the end).
 1. df['t'] = [x for x in range(10)]
 2. df['t-1'] = df['t'].shift(1)
 3. df['t-1'] = df['t'].shift(-1)
- Column And Row Sums In Pandas And Numpy. [Row and column sum in pandas and numpy](http://blog.mathandpencil.com/column-and-row-sums)
14. [Dataframe Validation In Python](https://www.youtube.com/watch?time_continue=905&v=1fHGXOfiDO0) - A Practical Introduction - Yotam Perkal - PyCon Israel 2018
15. In this talk, I will present the problem and give a practical overview (accompanied by Jupyter Notebook code examples) of three libraries that aim to address it: Voluptuous - Which uses Schema definitions in order to validate data [[https://github.com/alecthomas/voluptuous](https://github.com/alecthomas/voluptuous)] Engarde - A lightweight way to explicitly state your assumptions about the data and check that they're actually true [[https://github.com/TomAugspurger/engarde](https://github.com/TomAugspurger/engarde)] * TDDA - Test Driven Data Analysis [ [https://github.com/tdda/tdda](https://github.com/tdda/tdda)]. By the end of this talk, you will understand the Importance of data validation and get a sense of how to integrate data validation principles as part of the ML pipeline.

The same notes are in [Data & Model Tests](../../evals/data-and-model-tests.md).

- [Stop using itterows](https://medium.com/@rtjeannier/pandas-101-cont-9d061cb73bfc) use apply
- Data scientist and armchair sabermetrician, by Michael Rose
 Copyright. [(great) Group and Aggregate by One or More Columns in Pandas](https://jamesrledoux.com/code/group-by-aggregate-pandas)
- Aggregation and grouping of Dataframes is accomplished in Python Pandas using “groupby()” and “agg()” functions. [Pandas Groupby: Summarising, Aggregating, and Grouping data in Python](https://www.shanelynn.ie/summarising-aggregation-and-grouping-data-in-python-pandas/#applying-multiple-functions-to-columns-in-groups)
19. 25 pandas functions you didnt know about
- [json_normalize()](https://medium.com/data-science/all-pandas-json-normalize-you-should-know-for-flattening-json-13eae1dfb7dd)

### Exploratory Data Analysis (EDA)

This group under Pandas is exploratory data analysis.

- Engine for AI/ML/Data tracking, visualization, explainability, drift detection, and dashboards for Polyaxon. [Pandas summary](https://github.com/mouradmourafiq/pandas-summary)
- 1 Line of code data quality profiling & exploratory data analysis for Pandas and Spark DataFrames. [Pandas html profiling](https://github.com/pandas-profiling/pandas-profiling)
3. [Sweetviz](https://github.com/fbdesignpro/sweetviz) - "Sweetviz is an open-source Python library that generates beautiful, high-density visualizations to kickstart EDA (Exploratory Data Analysis) with just two lines of code. Output is a fully self-contained HTML application.

 The system is built around quickly visualizing target values and comparing datasets. Its goal is to help quick analysis of target characteristics, training vs testing data, and other such data characterization tasks."

<figure><img src="../../.gitbook/assets/image (6).png" alt=""><figcaption><p>Sweetviz EDA output.</p><p>Credit: by Sweetviz.</p></figcaption></figure>

## GIT / Bitbucket

Closing the workbench, this group is Git and Bitbucket notes.

- An interactive Git visualization tool to educate and challenge, by Peter Cottle. [understanding git](https://learngitbranching.js.org/)
- The page covers pre-commit. The page covers pre-commit. [pre-commit](https://pre-commit.com/)
- Learn how to rewrite Git history - Amend, Reword, Delete, Reorder, Squash and Split, by The Modern Coder. [Rewrite git history, all the commands](https://www.youtube.com/watch?v=ElRzTuYln0M)
4. [Installing git LFS](https://askubuntu.com/questions/799341/how-to-install-git-lfs-on-ubuntu-16-04)
- Use Git LFS with Bitbucket | Bitbucket Cloud | Atlassian Support. [Use git lfs](https://confluence.atlassian.com/bitbucket/use-git-lfs-with-bitbucket-828781636.html)
- Git Large File Storage (LFS) replaces large files such as audio samples, videos, datasets, and graphics with text pointers inside Git, while storing the file contents on a remote server like GitHub.com or GitHub Enterprise. [Download git-lfs](https://git-lfs.github.com/)
- Create a git wip command that lists your branches and when you last changed them. [Git wip](https://carolynvanslyck.com/blog/2020/12/git-wip/)

<figure><img src="../../.gitbook/assets/gimg-134013188da4.png" alt=""><figcaption><p>Git WIP illustration.</p>
<p>Credit: by <a href="https://carolynvanslyck.com/">Carolyn Van Slyck</a>.</p>
</figcaption></figure>

by [Carolyn Van Slyck](https://carolynvanslyck.com/)

- Convolutional Neural Networks, Explained, by Mayank Mishra. Managing virtual env with pyenv. [https://towardsdatascience.com/managing-virtual-environment-with-pyenv-ae6f3fb835f8](https://towardsdatascience.com/managing-virtual-environment-with-pyenv-ae6f3fb835f8)
- Jupyter Notebook in a virtual env by Christine Egan. [https://towardsdatascience.com/jupyter-notebooks-i-getting-started-with-jupyter-notebooks-f529449797d2](https://towardsdatascience.com/jupyter-notebooks-i-getting-started-with-jupyter-notebooks-f529449797d2)
- It pays to even vectorize conditional loops for speeding up the overall data transformation, by Tirthajyoti Sarkar. Fast vector calculation, a benchmark. [https://towardsdatascience.com/data-science-with-python-turn-your-conditional-loops-to-numpy-vectors-9484ff9c622e](https://towardsdatascience.com/data-science-with-python-turn-your-conditional-loops-to-numpy-vectors-9484ff9c622e)
- Data analysis and manipulation, plotting, resampling, and rolling. (good) Pandas time series manipulation. [https://towardsdatascience.com/practical-guide-for-time-series-analysis-with-pandas-196b8b46858f](https://towardsdatascience.com/practical-guide-for-time-series-analysis-with-pandas-196b8b46858f)


## Deprecated links

{% hint style="warning" %}
These links and images no longer work. The original wording is kept here. A same-resource copy, when one was checked, is used above.
{% endhint %}

- Coroutines generators async wait. This address no longer opens: https://masnun.com/2015/11/13/python-generators-coroutines-native-coroutines-and-async-await.html
- Intro to concurrent,futures. This address no longer opens: http://masnun.com/2016/03/29/python-a-quick-introduction-to-the-concurrent-futures-module.html
- Future task event loop. This address no longer opens: https://masnun.com/2015/11/20/python-asyncio-future-task-and-the-event-loop.html
- Just use venv. This address no longer opens: https://towardsdatascience.com/all-you-need-to-know-about-python-virtual-environments-9b4aae690f97
- pyenv, virtualenv and using them with Jupyter. This address no longer opens: https://albertauyeung.github.io/2020/08/17/pyenv-jupyter.html/
- Create isolated Jupyter ipython kernels with pyenv and virtualenv by alfredo motta. This address no longer opens: https://www.alfredo.motta.name/create-isolated-jupyter-ipython-kernels-with-pyenv-and-virtualenv/
- Timing and profiling in Jupyter. This address no longer opens: http://pynash.org/2013/03/06/timing-and-profiling/
- Nbdev on fast.ai. This address no longer opens: https://www.fast.ai/2019/12/02/nbdev/
- Virtual environments in jupyter. This address no longer opens: https://anbasile.github.io/programming/2017/06/25/jupyter-venv/
- Streamlit vs. Dash vs. Shiny vs. Voila vs. Flask vs. Jupyter. This address no longer opens: https://towardsdatascience.com/streamlit-vs-dash-vs-shiny-vs-voila-vs-flask-vs-jupyter-24739ab5d569
- Parallelism, pools, threads, dask. This address no longer opens: https://towardsdatascience.com/speed-up-your-algorithms-part-3-parallelization-4d95c0888748#7e6e
- How to inject headers into a headless CSV file. This address no longer opens: http://pythonforengineers.com/introduction-to-pandas/
- pandas function you didnt know about. This address no longer opens: https://towardsdatascience.com/25-pandas-functions-you-didnt-know-existed-p-guarantee-0-8-1a05dcaad5d0
- Using resample. This address no longer opens: https://towardsdatascience.com/using-the-pandas-resample-function-a231144194c4
- Basic TS manipulation. This address no longer opens: https://towardsdatascience.com/basic-time-series-manipulation-with-pandas-4432afee64ea
- Functional api for sk learn. This address no longer opens: https://scikit-lego.readthedocs.io/en/latest/preprocessing.html
