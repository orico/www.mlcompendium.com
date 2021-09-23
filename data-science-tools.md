# Data Science Tools

## **Python**

[**How to use better OOP in python.**](https://hackernoon.com/improve-your-python-python-classes-and-object-oriented-programming-d09ff461168d)

[**Best practices programming python classes - a great lecture.**](https://www.youtube.com/watch?v=HTLu2DFOdTg)

[**How to know pip packages size’**](https://stackoverflow.com/questions/34266159/how-to-see-pip-package-sizes-installed) **good for removal**

[**Python type checking tutorial**](https://medium.com/@ageitgey/learn-how-to-use-static-type-checking-in-python-3-6-in-10-minutes-12c86d72677b)

[**Import click - command line interface**](https://zetcode.com/python/click/)

[**Concurrency vs Parallelism \(great\)**](https://stackoverflow.com/questions/1050222/what-is-the-difference-between-concurrency-and-parallelism#:~:text=Concurrency%20is%20when%20two%20or,e.g.%2C%20on%20a%20multicore%20processor.)

* [**Async in python**](https://medium.com/velotio-perspectives/an-introduction-to-asynchronous-programming-in-python-af0189a88bbb)

[**Coroutines vs futures**](https://stackoverflow.com/questions/34753401/difference-between-coroutine-and-future-task-in-python-3-5)

1. **Coroutines** [**generators async wait**](https://masnun.com/2015/11/13/python-generators-coroutines-native-coroutines-and-async-await.html)
2. [**Intro to concurrent,futures**](http://masnun.com/2016/03/29/python-a-quick-introduction-to-the-concurrent-futures-module.html)
3. [**Future task event loop**](https://masnun.com/2015/11/20/python-asyncio-future-task-and-the-event-loop.html)

**Async io**

1. [**Intro**](https://realpython.com/lessons/what-asyncio/)
2. [**complete**](https://realpython.com/async-io-python/)

**Clean code:**

* [**Clean code in python git**](https://github.com/zedr/clean-code-python)
* [**About the book**](https://medium.com/@m_mcclarty/tech-book-talk-clean-code-in-python-aa2c92c6564f)
* **Virtual Environments**
* [**Guide to pyenv & pyenv virtualenv**](https://medium.com/swlh/a-guide-to-python-virtual-environments-8af34aa106ac)
* [**Managing virtual env with pyenv**](https://towardsdatascience.com/managing-virtual-environment-with-pyenv-ae6f3fb835f8)
* [**Just use venv**](https://towardsdatascience.com/all-you-need-to-know-about-python-virtual-environments-9b4aae690f97)
* [**Summary on all the \*envs**](https://stackoverflow.com/questions/41573587/what-is-the-difference-between-venv-pyvenv-pyenv-virtualenv-virtualenvwrappe)
* [**A really good primer on virtual environments**](https://realpython.com/python-virtual-environments-a-primer/)
* [**Introduction to venv**](http://cewing.github.io/training.python_web/html/presentations/venv_intro.html) **complementary to the above**
* [**Pipenv** ](https://pipenv.readthedocs.io/en/latest/)
* [**A great intro to pipenv**](https://realpython.com/pipenv-guide/)
* [**A complementary to pipenv above**](https://robots.thoughtbot.com/how-to-manage-your-python-projects-with-pipenv)
* [**Comparison between all \*env**](https://stackoverflow.com/questions/41573587/what-is-the-difference-between-venv-pyvenv-pyenv-virtualenv-virtualenvwrappe)

### **PYENV**

[**Installing pyenv**](https://bgasparotto.com/install-pyenv-ubuntu-debian)

[**Intro to pyenv**](https://realpython.com/intro-to-pyenv/)

[**Pyenv tutorial and finding where it is**](https://anil.io/blog/python/pyenv/using-pyenv-to-install-multiple-python-versions-tox/) _\*\*_

[**Pyenv override system python on mac**](https://github.com/pyenv/pyenv/issues/660)

## **JUPYTER**

* [**Cloud GPUS cheap**](https://www.paperspace.com/gradient)
* [**Importing a notebook as a module**](http://jupyter-notebook.readthedocs.io/en/latest/examples/Notebook/Importing%20Notebooks.html)
* **Important** [**colaboratory commands for jupytr** ](https://medium.com/deep-learning-turkey/google-colab-free-gpu-tutorial-e113627b9f5d)
* [**Timing and profiling in Jupyter**](http://pynash.org/2013/03/06/timing-and-profiling/)
* **\(**[**Debugging in Jupyter, how?\)**](https://kawahara.ca/how-to-debug-a-jupyter-ipython-notebook/) **- put a one liner before the code and query the variables inside a function.**
* [**28 tips n tricks for jupyter** ](https://www.dataquest.io/blog/jupyter-notebook-tips-tricks-shortcuts/)
* **Jupyter notebooks as a module**
  1. [**Nbdev**](https://github.com/fastai/nbdev)**, on** [**fast.ai**](https://www.fast.ai/2019/12/02/nbdev/)
  2. [**jupytext**](https://github.com/mwouts/jupytext)
* [**Virtual environments in jupyter**](https://anbasile.github.io/programming/2017/06/25/jupyter-venv/)
  1. **Enter your project directory**
  2. **$ python -m venv projectname**
  3. **$ source projectname/bin/activate**
  4. **\(venv\) $ pip install ipykernel**
  5. **\(venv\) $ ipython kernel install --user --name=projectname**
  6. **Run jupyter notebook \* \(not entirely sure how this works out when you have multiple notebook processes, can we just reuse the same server?\)**
  7. **Connect to the new server at port 8889**
  8. 
* [**Virtual env with jupyter** ](https://janakiev.com/til/jupyter-virtual-envs/)

**\(**[**how does reshape work?\)**](http://anie.me/numpy-reshape-transpose-theano-dimshuffle/) **- a shape of \(2,4,6\) is like a tree of 2-&gt;4 and each one has more leaves 4-&gt;6.**

**As far as i can tell, reshape effectively flattens the tree and divide it again to a new tree, but the total amount of inputs needs to stay the same. 2\*4\*6 = 4\*2\*3\*2 for example**

**code:  
`import numpy    
rng = numpy.random.RandomState(234)    
a = rng.randn(2,3,10)    
print(a.shape)    
print(a)    
b = numpy.reshape(a, (3,5,-1))    
print(b.shape)    
print (b)`**

**\*\*\* A tutorial for** [**Google Colaboratory - free Tesla K80 with Jup-notebook**](https://www.kdnuggets.com/2018/02/google-colab-free-gpu-tutorial-tensorflow-keras-pytorch.html/2)

[**Jupyter on Amazon AWS**](https://blog.keras.io/running-jupyter-notebooks-on-gpu-on-aws-a-starter-guide.html)

**How to add extensions to jupyter:** [**extensions**](https://codeburst.io/jupyter-notebook-tricks-for-data-science-that-enhance-your-efficiency-95f98d3adee4)

[**Connecting from COLAB to MS AZURE**](https://medium.com/@d.sakryukin/simple-cryptocurrency-trading-data-preparation-in-15-minutes-using-ms-azure-and-google-colab-44872b023d11)

[**Streamlit vs. Dash vs. Shiny vs. Voila vs. Flask vs. Jupyter**](https://towardsdatascience.com/streamlit-vs-dash-vs-shiny-vs-voila-vs-flask-vs-jupyter-24739ab5d569)

## **SCIPY**

1. [**Optimization problems, a nice tutorial**](http://scipy-lectures.org/advanced/mathematical_optimization/) **to finding the minima**
2. [**Minima / maxima**](https://stackoverflow.com/questions/4624970/finding-local-maxima-minima-with-numpy-in-a-1d-numpy-array) **finding it in a 1d numpy array**

## **NUMPY**

[**Using numpy efficiently**](https://speakerdeck.com/cournape/using-numpy-efficiently) **- explaining why vectors work faster.**  
[**Fast vector calculation, a benchmark**](https://towardsdatascience.com/data-science-with-python-turn-your-conditional-loops-to-numpy-vectors-9484ff9c622e) **between list, map, vectorize. Vectorize wins. The idea is to use vectorize and a function that does something that may involve if conditions on a vector, and do it as fast as possible.**

## **PANDAS**

1. [**Great introductory tutorial**](http://nikgrozev.com/2015/12/27/pandas-in-jupyter-quickstart-and-useful-snippets/#loading_csv_files) **ab**out using pandas, loading, loading from zip, seeing the table’s features, accessing rows & columns, boolean operations, calculating on a whole row\column with a simple function and on two columns even, dealing with time\date parsing.
2. [Visualizing pandas pivoting and reshaping functions by Jay Alammar](http://jalammar.github.io/visualizing-pandas-pivoting-and-reshaping/) - pivot melt stack unstack
3. [How to beautify pandas dataframe using html display](https://stackoverflow.com/questions/26873127/show-dataframe-as-table-in-ipython-notebook)
4. [Speeding up pandas ](https://realpython.com/fast-flexible-pandas/)
5. [The fastest way to select rows by columns, by using masked values](https://stackoverflow.com/questions/17071871/select-rows-from-a-dataframe-based-on-values-in-a-column-in-pandas) \(benchmarked\):
6. def mask\_with\_values\(df\): mask = df\['A'\].values == 'foo' return df\[mask\]
7. [Parallelism, pools, threads, dask](https://towardsdatascience.com/speed-up-your-algorithms-part-3-parallelization-4d95c0888748#7e6e)
8. [Accessing dataframe rows, columns and cells](http://pythonhow.com/accessing-dataframe-columns-rows-and-cells/)- by name, by index, by python methods.
9. [Looping through pandas](https://medium.com/swlh/how-to-efficiently-loop-through-pandas-dataframe-660e4660125d)
10. [How to inject headers into a headless CSV file](http://pythonforengineers.com/introduction-to-pandas/) - 
11. [Dealing with time series](http://pandas.pydata.org/pandas-docs/stable/timeseries.html) in pandas,
    1. [Create a new column](https://stackoverflow.com/questions/25570147/add-new-column-based-on-boolean-values-in-a-different-column) based on a \(boolean or not\) column and calculation:
    2. Using python \(map\)
    3. Using numpy
    4. using a function \(not as pretty\)
12. Given a DataFrame, the [shift](http://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/)\(\) function can be used to create copies of columns that are pushed forward \(rows of NaN values added to the front\) or pulled back \(rows of NaN values added to the end\).
    1. df\['t'\] = \[x for x in range\(10\)\]
    2. df\['t-1'\] = df\['t'\].shift\(1\)
    3. df\['t-1'\] = df\['t'\].shift\(-1\)
13. [Row and column sum in pandas and numpy](http://blog.mathandpencil.com/column-and-row-sums)
14. [Dataframe Validation In Python](https://www.youtube.com/watch?time_continue=905&v=1fHGXOfiDO0) - A Practical Introduction - Yotam Perkal - PyCon Israel 2018
15. In this talk, I will present the problem and give a practical overview \(accompanied by Jupyter Notebook code examples\) of three libraries that aim to address it: Voluptuous - Which uses Schema definitions in order to validate data \[[https://github.com/alecthomas/voluptuous](https://www.youtube.com/redirect?v=1fHGXOfiDO0&event=video_description&redir_token=jIIzdRAEjZBpzVhRYfFzTcx52358MTU0NzQ0ODY3N0AxNTQ3MzYyMjc3&q=https%3A%2F%2Fgithub.com%2Falecthomas%2Fvoluptuous)\] Engarde - A lightweight way to explicitly state your assumptions about the data and check that they're actually true \[[https://github.com/TomAugspurger/engarde](https://www.youtube.com/redirect?v=1fHGXOfiDO0&event=video_description&redir_token=jIIzdRAEjZBpzVhRYfFzTcx52358MTU0NzQ0ODY3N0AxNTQ3MzYyMjc3&q=https%3A%2F%2Fgithub.com%2FTomAugspurger%2Fengarde)\] \* TDDA - Test Driven Data Analysis \[ [https://github.com/tdda/tdda](https://github.com/tdda/tdda)\]. By the end of this talk, you will understand the Importance of data validation and get a sense of how to integrate data validation principles as part of the ML pipeline.
16. [Stop using itterows](https://medium.com/@rtjeannier/pandas-101-cont-9d061cb73bfc), use apply.
17. [\(great\) Group and Aggregate by One or More Columns in Pandas](https://jamesrledoux.com/code/group-by-aggregate-pandas)
18. [Pandas Groupby: Summarising, Aggregating, and Grouping data in Python](https://www.shanelynn.ie/summarising-aggregation-and-grouping-data-in-python-pandas/#applying-multiple-functions-to-columns-in-groups)
19. [pandas function you didnt know about](https://towardsdatascience.com/25-pandas-functions-you-didnt-know-existed-p-guarantee-0-8-1a05dcaad5d0)

### Exploratory Data Analysis \(EDA\) 

1. [Pandas summary](https://github.com/mouradmourafiq/pandas-summary)
2. [Pandas html profiling](https://github.com/pandas-profiling/pandas-profiling)
3. [Sweetviz](https://github.com/fbdesignpro/sweetviz)  - "Sweetviz is an open-source Python library that generates beautiful, high-density visualizations to kickstart EDA \(Exploratory Data Analysis\) with just two lines of code. Output is a fully self-contained HTML application.

   The system is built around quickly **visualizing target values** and **comparing datasets**. Its goal is to help quick analysis of target characteristics, training vs testing data, and other such data characterization tasks."

### **TIMESERIES**

1. **\(good\)** [**Pandas time series manipulation**](https://towardsdatascience.com/practical-guide-for-time-series-analysis-with-pandas-196b8b46858f)
2. [**Using resample**](https://towardsdatascience.com/using-the-pandas-resample-function-a231144194c4)

![](https://lh6.googleusercontent.com/y9f1kyTrWs6kbOeGZctlWkHXW-LXsWWwtjul9GqSV-xLz3xnIH8PilD2O7jUzA9pqPcvXMgbHDI-GfJqfimxt-gwT9LMBlJCSJqd89htvQ5JsuxttcLRakOFShpyEfbjraDnNgwL)

1. [**Basic TS manipulation**](https://towardsdatascience.com/basic-time-series-manipulation-with-pandas-4432afee64ea)
2. [**Fill missing ts gaps, or how to resample**](https://stackoverflow.com/questions/32241692/fill-missing-timeseries-data-using-pandas-or-numpy)
3. **SCI-KIT LEARN**
4. **Pipeline t**[**o json 1**](https://cmry.github.io/notes/serialize)**,** [**2**](https://cmry.github.io/notes/serialize-sk)
5. [**cuML**](https://github.com/rapidsai/cuml) **- Multi gpu, multi node-gpu alternative for SKLEARN algorithms**
6. [**Gpu TSNE ^**](https://www.reddit.com/r/MachineLearning/comments/e0j9cb/p_2000x_faster_rapids_tsne_3_hours_down_to_5/?utm_source=share&utm_medium=ios_app&utm_name=iossmf)
7. [**Awesome code examples**](http://machinelearningmastery.com/get-your-hands-dirty-with-scikit-learn-now/) **about using svm\knn\naive\log regression in sklearn in python, i.e., “fitting a model onto the data”**
8. [**Parallelism of numpy, pandas and sklearn using dask and clusters**](https://github.com/dask/dask)**.** [**Webpage**](https://dask.pydata.org/en/latest/)**,** [**docs**](http://dask-ml.readthedocs.io/en/latest/index.html)**,** [**example in jupyter**](https://hub.mybinder.org/user/dask-dask-examples-6bi4j3qj/notebooks/machine-learning.ipynb)**.** 

**Also Insanely fast,** [**see here**](https://www.youtube.com/watch?v=5Zf6DQaf7jk)**.**

1. [**Functional api for sk learn**](https://scikit-lego.readthedocs.io/en/latest/preprocessing.html)**, using pipelines. thank you sk-lego.**
2. ![](https://lh4.googleusercontent.com/xKPNwOKjUIG_mFuW3nshvvL7MTmkYk8G5UukjjrLAqEUloehU1YR3WJ9nYI1nkCkM28r7qTdkQlILHNcFtd1lYalKP1lI8tUfw64beU15LiogQi785F9p37GqoA_fKRgMbkNALtP)![](https://lh6.googleusercontent.com/iEIbuEyV0tmZxsBxny_DrtLGtwI36st5NoIZ1OaOqV5HqdPTvuu1cSnIgxDuNcTYVM4V--pLHggZPmt1GohXq1AjFk_Mv4xrSNXka2SmKa6Sfx7r15z2J3Dpre_owNQ0E_BrGfI3)![](https://lh3.googleusercontent.com/yw9Mba28iRLnSNkNJy-oloxeBdRMQ11htLG45Qs8b-vaNtrUk9ecsre36EeS5RxVP5MNnqKLpx7S5qpHTlLCqS9OicYI4QarEc5ewBgdMnzqnZUAXyGdumGwb0lyjP98sM4BAt9c)

## **FAST.AI**

1. [**Medium**](https://medium.com/@hiromi_suenaga/deep-learning-2-part-1-lesson-1-602f73869197) **on all fast.ai courses, 14 posts**

## **PYCARET**

[**1. What is? by vidhaya**](https://www.analyticsvidhya.com/blog/2020/05/pycaret-machine-learning-model-seconds/?utm_source=AVFacebook&utm_medium=post&utm_campaign=19_june_intermediate_article&fbclid=IwAR0NZV5fJgXtpoCBfmauCiGQC_QOK0cbQrpuhhpDBAtEngGG_NBsRlcVRas) **-** [**PyCaret**](https://pycaret.org/) **is an open-source, machine learning library in Python that helps you from data preparation to model deployment. It is easy to use and you can do almost every data science project task with just one line of code.**

## **NVIDIA TF CUDA CUDNN**

* [**Install TF**](https://www.tensorflow.org/install/install_linux#NVIDIARequirements)
* [**Install cuda on ubuntu**](https://devtalk.nvidia.com/default/topic/1030495/cuda-setup-and-installation/install-a-specific-cuda-version-for-ubuntu-16-04/)**,** [**official linux**](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
* [**Replace cuda version**](https://askubuntu.com/questions/959835/how-to-remove-cuda-9-0-and-install-cuda-8-0-instead) _\*\*_
* [**Cuda 9 download**](https://developer.nvidia.com/cuda-90-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1704&target_type=runfilelocal)
* [**Install cudnn**](https://askubuntu.com/questions/1033489/the-easy-way-install-nvidia-drivers-cuda-cudnn-and-tensorflow-gpu-on-ubuntu-1)
* [**Installing everything easily**](https://askubuntu.com/questions/1033489/the-easy-way-install-nvidia-drivers-cuda-cudnn-and-tensorflow-gpu-on-ubuntu-1)
* [**Failed**](https://stackoverflow.com/questions/43022843/nvidia-nvml-driver-library-version-mismatch) **to initialize NVML: Driver/library version mismatch**

## **GCP**

[**Resize google disk size**](https://medium.com/google-cloud/resize-your-persist-disk-on-google-cloud-on-the-fly-b3491277b718)**,** [**1,**](https://cloud.google.com/compute/docs/disks/add-persistent-disk) **\*\*\[**2**\]\(**[https://www.cloudbooklet.com/how-to-resize-disk-of-a-vm-instance-in-google-cloud/](https://www.cloudbooklet.com/how-to-resize-disk-of-a-vm-instance-in-google-cloud/)**\)**,\*\*

## **SQL**

1. [**Introduction, index, keys, joins, aliases etc.**](https://www.youtube.com/watch?v=nWeW3sCmD2k)**,** [**newer**](https://www.youtube.com/watch?v=9ylj9NR0Lcg)
2. [**Sql cheat sheet**](https://gist.github.com/bradtraversy/c831baaad44343cc945e76c2e30927b3)
3. [**Primary key**](https://www.eukhost.com/blog/webhosting/whats-the-purpose-use-primary-foreign-keys/)
4. [**Foreign key, a key constraint that is included in the primary key allowed values**](https://www.1keydata.com/sql/sql-foreign-key.html)
5. [**Index, i.e., book index for fast reading**](https://www.tutorialspoint.com/sql/sql-indexes.htm)

## **GIT / Bitbucket**

1. [**Installing git LFS**](https://askubuntu.com/questions/799341/how-to-install-git-lfs-on-ubuntu-16-04)
2. [**Use git lfs**](https://confluence.atlassian.com/bitbucket/use-git-lfs-with-bitbucket-828781636.html)
3. [**Download git-lfs**](https://git-lfs.github.com/)
4. [**Git wip**](https://carolynvanslyck.com/blog/2020/12/git-wip/) **\(great\)** ![](https://lh5.googleusercontent.com/3pMgGGFXb24nH1jqCLAL9IHp0dYH5H2pp_ZEDDxFvj89nsifmcUH58qHOu0_jTu6ONJsE2cJXW7qT1vnbZ43bWI2iRdUho24yyaOiHtQ5Ygrx0mWA3GhSMOFKsfS0t51SRwda6nB)

