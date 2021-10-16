# Templatization

* [**Awesome log analysis**\
  ****](https://github.com/logpai/awesome-log-analysis)![](https://lh6.googleusercontent.com/PM_BNp146KH_xeEkpCfptSnhvjgluGa9WpxORgpRPqE3CmDMDhGEdRW2ldG1IXV9ZhJXIvJQkEvmNPALe7kw6Xb8JHY-5NRfql27kS2Cf4wgkBKOqDCsmhYhcZolYDy-1ycekXgx)
* **(really good) **[**And list of papers for each field**](https://github.com/logpai/awesome-log-analysis/blob/master/papers.md#anomaly-detection)
* [**How to use log analytics to detect log anomaly**](https://www.msystechnologies.com/blog/how-to-use-log-analytics-to-detect-log-anomaly/)** - more of a survey into the technologies available**\
  ****![](https://lh6.googleusercontent.com/1mjl7BDsTwHKIVLWnlsMffU3S6A4QIKkoL-sMpgEwiYUZyRVHAtY0FI7M2707LvjTHFf3fZ2aiwhzGaCCD2o9nEmfbQIye0cH0HHBy1ZeVPM_X1DhaThvHw82FFnNHC2gfcboIB5)
* [**3 things we learned about applying word vectors to logs**](https://gab41.lab41.org/three-things-we-learned-about-applying-word-vectors-to-computer-logs-c199070f390b#.bk8wnk7pr)
  * **GloVe consistently identified approximately 50 percent or more of the seeded events in the synthetic data as either exact or as valid sub-sequence matches. GloVe tended to nominate a limited number of template sequences that weren’t related to seeded events and many of those were tied to high frequency templates. When we tested GloVe against a generated data set with multiple SSH sessions in an auditd file, GloVe correctly proposed a single event that included all of the auditd record types defined in the SSH user login lifecycle.**
  * **Glove produces sub sequences that needs to be stitched to create a match**![](https://lh4.googleusercontent.com/OtPZY2dZzyVEny4mhyvjzq4ZYfOeoKPq3fGSXm9Mk7aP4eDSHP3G54LrLXEZs67Q8QjXUOKXFs5UHPIwI8LGTMAQ6l5NmR4UjXOegQkCa6CX05ZONxLzWtdYqjw99\_y_CJBlchDj)
  * **Glove is faster than paris and fp growth**
  * **Their clustering method misclassified**\
    ****\
    ****
* [**Towards an NLP based log template generation algorithm for system log analysis**](http://www.3at.work/papers/cfi2014.pdf)** - CRF for templatization, i.e. ner style. “we can see that the more learning data given, the more accurately CRF produces log templates. Clearly a sufficient number of train data enables us to analyze less frequently appearing log templates. Therefore, it is reasonable that a log template can be analyzed correctly if train data include some of similar templates. However, in terms of log template accuracy, CRF requires 10000 train data to achieve same accuracy as Vaarandi’s algorithm”**

![](https://lh5.googleusercontent.com/-\_axdpi4F7bTBQGnRnzf--j4mja6NMbRJfaoLmIQOQJeuF5fBqojXEBDbpzFKkGBK7skRMIQi6AGKCXzWl7PgSnqGe5dekwxRqRtqLxAoGpIBvH0XAlgNVxJJeZTRmnTE2UalNqo)

![](https://lh4.googleusercontent.com/hxCR-hM0aqF8wQBdKwloQtyHrd00MuP3rgfLbKZiiBRv5K06E5y7bsLp9Ye7MPNqztMULM429ZEbmFGX_OGcLjP2TKHLlaa896Etyvj0rkeU-Fb5zoyTrJFON6Fm_RrhGL2by8qV)

![](https://lh5.googleusercontent.com/edSGC4ElX8-mn2pc6yn5WbqzUPYSRxorl1o-Yk9e8w-GBHrKa8234G1glpBpd3NxUdJJpf8Uyij-GSuTWnLYwDGnr7i-z63LtNQixj9a5oYPY4M6DMi3Msif_PSAr41lN7jqc9y8)

### **Logpai**

1. [**Logpai**](https://github.com/logpai)
2. [**Loghub datasets**](https://github.com/logpai/loghub)
3. [**logpaI loglizer:**](https://github.com/PinjiaHe)** **[**An Evaluation Study on Log Parsing and Its Use in Log Mining**](https://pinjiahe.github.io/papers/DSN16.pdf)**, **[**git**](https://github.com/logpai/loglizer)
4. [**log3C**](https://github.com/logpai/Log3C)** - **[**paper**](https://dl.acm.org/citation.cfm?id=3236083)** Log3C is a general framework that identifies service system problems from system logs. It utilizes both system logs and system KPI metrics to promptly and precisely identify impactful system problems. Log3C consists of four steps: Log parsing, Sequence vectorization, Cascading Clustering and Correlation analysis. This is a joint work by CUHK and Microsoft Research. The repository contains the source code of Log3C, including data loading, sequence vectorization, cascading clustering, data saving, etc. The core part is the cascading clustering algorithm, which groups a large number of sequence vectors into clusters by iteratively sampling, clustering, matching. **![](https://lh5.googleusercontent.com/66iv2rGsmWcnFbMZPO2Neg0t9X\_\_mkGI8bOCh1ZAjdIvqqmdov8jiGwWiQANu69PalsDQaDTEbzbu1JezOi_w2Y7z1Ff_do7mwXFFhqY5CUW1CQ3ba19sLMsXP7JpUA375VWxO1H)![](https://lh3.googleusercontent.com/5w3vUKudl-w5oht9i7rw13Wl6DnQaNIPgyaCscyoqBEFZ3r0r7Hz8NonRA6LSQuPxDL--J6O2Rlb1698dsGz_D5NlVn5RBY0tw6FcHgqO3BYLOm_AFzRzxOYzGPp7NIog8Or6-fu)![](https://lh6.googleusercontent.com/\_5BvgUnyNm-3doZBdOzj2fY16UtBVlfp4xc--EU0YVBaHDaOnIsfRPuicMKiEgQOlycYDpYBZrfUGDaJIN0rw4kiAnA0o57HLwUUnHWR0lObuNIjslSqbmf8C_pkbxGudgHnjDdp)****\
   **Selection of KPI: In our experiments, we use failure rate as the KPI for problem identification. failure rate is an important KPI for evaluating system service availability. There are also other KPIs such as mean time between failures, average request latency, throughput, etc. In our future work, we will experiment with problem identification concerning different KPI metrics. Noises in labeling: Our experiments are based on three datasets that are collected as a period of logs on three different days. The engineers manually inspected and labeled the log sequences. (false positives/negatives) may be introduced during the manual labeling process. However, as the engineers are experienced professionals of the product team who maintain the service system, we believe the amount of noise is small (if it exists)**

**Furthermore, we compare our method with two typical methods: PCA \[41] and Invariants Mining \[23]. All these three methods are unsupervised, log-based problem identification methods. PCA projects the log sequence vectors into a subspace. If the projected vector is far from the majority, it is considered as a problem. Invariants Mining extracts the linear relations (invariants) between log event occurrences, which hypothesizes that log events are often pairwise generated. For example, when processing files, "File A is opened" and "File A is closed" should be printed as a pair. Log sequences that violate the invariants are regarded as problematic. Log3C achieves good recalls (similar to those achieved by two comparative methods) and surpasses the comparative methods concerning precision and F1-measure. **

1. [**Logzip**](https://github.com/logpai/logzip)** **[**paper**](https://arxiv.org/abs/1910.00409)**-Logzip is an (personal note seems to be offline) efficient compression tool specific for log files. It compresses log files by utilizing the inherent structures of raw log messages, and thereby achieves a high compression ratio.The results show that logzip can save about half of the storage space on average over traditional compression tools. Meanwhile, the design of logzip is highly parallel and only incurs negligible overhead. In addition, we share our industrial experience of applying logzip to Huawei's real products.**
2. **Logadvisor - **[**paper1**](https://jiemingzhu.github.io/pub/qfu_icse2014.pdf)**, **[**2**](http://jmzhu.logpai.com/pub/jmzhu_icse2015.pdf)** - Our goal, referred to as “learning to log”, is to automatically learn the common logging practice as a machine learning model, and then leverage the model to guide developers to make logging decisions during new development. **
   1. **Labels:  logging method (e.g., Console.Writeline())**
   2. **Features: we need to extract useful features (e.g., exception type) from the collected code snippets for making logging decisions,**
   3. **Train / suggest**
   4. ![](https://lh3.googleusercontent.com/k1bAC6cD6Ut9lBfUfXeqht9j8jzd4OLcLM_as4pJcEhtX2VuCJmFbVRnJAtos5\_lXd8X7ZkFU6WCYmx02bQo0NtWNEZc9J4KgzrwdC7X3uHiDsmbakWbun15SHFiQ_QxNjAyBbpK)![](https://lh6.googleusercontent.com/4LRepv7-CHy91fExDSyk59vmmGXN4yFHayTDe5qmj0u1UXLBrBTmtKKlUwZOWxf-sT-9i0FJ7rs5ZPhQ5koykZgtQhrNJSmGK8T_Fuq49gFqHozzBiubl4bq09qyympjOSK7gcs1)
3. [**Logging descriptions -**](https://github.com/logpai/LoggingDescriptions)** This repository maintains a set of \<code, log> pairs extracted from popular open-source projects, which are amendable to logging description generation research.**
4. **(REALLY GOOD) **[**Loglizer**](https://github.com/logpai/loglizer)** **[**paper**](http://jmzhu.logpai.com/pub/slhe_issre2016.pdf)** **[**git demo**](https://github.com/logpai/loglizer/tree/master/demo)**- Loglizer is a machine learning-based log analysis toolkit for automated anomaly detection.**

![](https://lh4.googleusercontent.com/TtxjVZA8y03fapSbEa0-9m5qD6nZEl1sUShed_UmBXaKcoRjqov5SOLCM4uWW6U9dOG\_9nmYNOBqTUDnYDtUAY06XVQUsc7oJSQdvLbOCEh4\_0Tsaih_ucswOYmm5hVmINkwj99l)

* **Feature extraction using fixed window, sliding window and session window**
  * **Fixed window: Both fixed windows and sliding windows are based on timestamp, which records the occurrence time of each log. Each fixed window has its size, which means the time span or time duration. As shown in Figure 1, the window size is Δt, which is a constant value, such as one hour or one day. Thus, the number of fixed windows depends on the predefined window size. Logs that happened in the same window are regarded as a log sequence. **
  * **Sliding window: Different from fixed windows, sliding windows consist of two attributes: window size and step size, e.g., hourly windows sliding every five minutes. In general, step size is smaller than window size, therefore causing the overlap of different windows. Figure 1 shows that the window size is ΔT , while the step size is the forwarding distance. The number of sliding windows, which is often larger than fixed windows, mainly depends on both window size and step size. Logs that occurred in the same sliding window are also grouped as a log sequence, though logs may duplicate in multiple sliding windows due to the overlap. **
  * **Session window: Compared with the above two windowing types, session windows are based on identifiers instead of the timestamp. Identifiers are utilized to mark different execution paths in some log data. For instance, HDFS logs with block_id record the allocation, writing, replication, deletion of certain block. Thus, we can group logs according to the identifiers, where each session window has a unique identifier**
* **Many Supervised methods and most importantly a cool unsupervised method - > PCA for anomaly based on the length of the projected transformed sample vector by dividing the first and last PC vectors:**
* **PCA was first applied in log-based anomaly detection by Xu et al. \[47]. In their anomaly detection method, each log sequence is vectorized as an event count vector. After that, PCA is employed to find patterns between the dimensions of event count vectors. Employing PCA, two subspace are generated, namely normal space Sn and anomaly space Sa. Sn is constructed by the first k principal components and Sn is constructed by the remaining (n−k), where n is the original dimension. Then, the projection ya = (1−P P T )y of an event count vector y to Sa is calculated, where P = \[v1,v2, ...,vk,] is the first k principal components. If the length of ya is larger**
*

1. [**LogParser**](https://github.com/logpai/logparser)** - a benchmark for log parsers using 13 models on 16 datasets**\
   **Important insights:**
2. **Drain is fastest, most performing on most datasets (9/16)**
3. **Fitting parameters should be adapted, which what makes drain the most performing**
4. **More demanding metrics.**
5. **Papers: **

* **\[ICSE'19] Jieming Zhu, Shilin He, Jinyang Liu, Pinjia He, Qi Xie, Zibin Zheng, Michael R. Lyu. **[**Tools and Benchmarks for Automated Log Parsing**](https://arxiv.org/pdf/1811.03509.pdf)**. International Conference on Software Engineering (ICSE), 2019.**
* **\[TDSC'18] Pinjia He, Jieming Zhu, Shilin He, Jian Li, Michael R. Lyu. **[**Towards Automated Log Parsing for Large-Scale Log Data Analysis**](https://jiemingzhu.github.io/pub/pjhe_tdsc2017.pdf)**. IEEE Transactions on Dependable and Secure Computing (TDSC), 2018.**
* **\[ICWS'17] Pinjia He, Jieming Zhu, Zibin Zheng, Michael R. Lyu. **[**Drain: An Online Log Parsing Approach with Fixed Depth Tree**](https://jiemingzhu.github.io/pub/pjhe_icws2017.pdf)**. IEEE International Conference on Web Services (ICWS), 2017.**
* **\[DSN'16] Pinjia He, Jieming Zhu, Shilin He, Jian Li, Michael R. Lyu. **[**An Evaluation Study on Log Parsing and Its Use in Log Mining**](https://jiemingzhu.github.io/pub/pjhe_dsn2016.pdf)**. IEEE/IFIP International Conference on Dependable Systems and Networks (DSN), 2016.**

1.

![](https://lh5.googleusercontent.com/61Q9N3ArWIwYdnQpUiTHMWCc5C_gnGeYkLZ9uv0GhNorh4tRQ-x9YReH0JZkSsLEYooAqVHWhzavf9ejTiHxDkmoSVpplEpbxMwXJ2EGx0xB3Xb08eDaz1qoVUNWtj-zupggmzOu)

1. [**Gpt3 with logs**](https://www.zebrium.com/blog/using-gpt-3-with-zebrium-for-plain-language-incident-root-cause-from-logs)
2. [**DeepLog**](https://www.cs.utah.edu/\~lifeifei/papers/deeplog.pdf)** (**[**git**](https://github.com/wuyifan18/DeepLog)**)**

[**Log2vec**](https://netman.aiops.org/wp-content/uploads/2020/05/Log2Vec-icccn20.pdf)** (**[**git**](https://github.com/NetManAIOps/Log2Vec)**)**
