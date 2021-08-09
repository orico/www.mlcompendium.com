# Meta Learning

[**What is?**](https://www.automl.org/) **Automated Machine Learning provides methods and processes to make Machine Learning available for non-Machine Learning experts, to improve efficiency of Machine Learning and to accelerate research on Machine Learning.**  


**Personal note: automl algorithms in this field will bridge the gap and automate several key processes, but it will not allow a practitioner to do serious research or solve business or product problems easily. The importance of this field is to advance each subfield, whether HPO, NAS, etc. these selective novelties can help us solve specific issues, i.e, lets take HPO, we can use it to save time and money on redundant parameter searches, especially when it comes to resource heavy algorithms such as Deep learning \(think GPU costs\).**  


**Personal thoughts on optimizations: be advised that optimizing problems will not guarantee a good result, you may over fit your problem in ways you are not aware of, beyond traditional overfitting and better accuracy doesn't guarantee a better result \(for example if your dataset is unbalanced, needs imputing, cleaning, etc.\).**   
  


**Always examine the data and results in order to see if they are correct.**  


[**Automl.org’s github - it has a backup for the following projects.**](https://github.com/automl)  
****

[**Automl.org**](https://www.automl.org/) **is a joint effort between two universitie, freiburg and hannover, their website curates information regarding:**

1. **HPO - hyper parameter optimization**
2. **NAS - neural architecture search**
3. **Meta Learning - learning across datasets, warmstarting of HPO and NAS etc.**

**Automl aims to automate these processes:**

* **Preprocess and clean the data.**
* **Select and construct appropriate features.**
* **Select an appropriate model family.**
* **Optimize model hyperparameters.**
* **Postprocess machine learning models.**
* **Critically analyze the results obtained.**

**Historically, AFAIK AutoML’s birth started with several methods to optimize each one of the previous processes in ml. IINM,** [**weka’s paper \(2012**](https://arxiv.org/abs/1208.3719)**\) was the first step in aggregating these ideas into a first public working solution.**  


**The following is referenced from AutoML.org:**  


### **ML Systems**

* [**AutoWEKA**](http://www.cs.ubc.ca/labs/beta/Projects/autoweka/) **is an approach for the simultaneous selection of a machine learning algorithm and its hyperparameters; combined with the** [**WEKA**](http://www.cs.waikato.ac.nz/ml/weka/) **package it automatically yields good models for a wide variety of data sets.**
* [**Auto-sklearn**](http://automl.github.io/auto-sklearn/stable/) **is an extension of AutoWEKA using the Python library** [**scikit-learn**](http://scikit-learn.org/stable/) **which is a drop-in replacement for regular scikit-learn classifiers and regressors.**
* [**TPOT**](http://epistasislab.github.io/tpot/) **is a data-science assistant which optimizes machine learning pipelines using genetic programming.**
* **\(google\)** [**H2O AutoML**](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html) **provides automated model selection and ensembling for the** [**H2O machine learning and data analytics platform**](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/welcome.html)**. \(**[**git**](https://github.com/google/automl)**\)**
* [**TransmogrifAI**](https://github.com/salesforce/TransmogrifAI) **is an AutoML library running on top of Spark.**
* [**MLBoX**](https://github.com/AxeldeRomblay/MLBox) **is an AutoML  library with three components: preprocessing, optimisation and prediction**
* [**MLJar**](https://mljar.com/) **\(**[**git**](https://github.com/mljar/mljar-supervised)**\)** [**medium**](https://medium.com/@MLJARofficial/mljar-supervised-automl-with-explanations-and-markdown-reports-36d5104e117)**,** [**2**](https://towardsdatascience.com/automating-eda-machine-learning-6ddb76c1eb4d) **- Automated Machine Learning for tabular data mljar builds a complete Machine Learning Pipeline. Perform exploratory analysis, search for a signal in the data, and discover relationships between features in your data with AutoML.  Train top ML models with advanced feature engineering, many algorithms, hyper-parameters tuning, Ensembling, and Stacking. Stay ahead of competitors and predict the future with advanced ML. Deploy your models in the cloud or use them locally**
  * **+ advanced feature engineering**
  * **+ algorithms selection and tuning**
  * **+ automatic documentation**
  * **+ ML explanations**  

![](https://lh3.googleusercontent.com/duUZ_u8kLJ9fhJ1AtGodADX6n3aV4CB9hsLhCV4yANEA0_Rui8yQBAtBe_DxHsJP0s-I8mCCRlyMgvZwJFkc0hy0TtejPLqq_AYmOMXyE73xph8YhEjVQnYeR0lDqI0LTf5YnSOG)

### **Hyper param optimization** 

* [**Hyperopt**](http://jaberg.github.io/hyperopt/)**, including the TPE algorithm**
* [**Sequential Model-based Algorithm Configuration \(SMAC\)**](http://aclib.net/SMAC/)
* [**Spearmint**](https://github.com/JasperSnoek/spearmint)
* [**BOHB**](https://www.automl.org/automl/bohb/)**: Bayesian Optimization combined with HyperBand**
* [**RoBO – Robust Bayesian Optimization framework**](http://www.automl.org/automl/robo/)
* [**SMAC3**](https://github.com/automl/SMAC3) **– a python re-implementation of the SMAC algorithm**

### **Architecture Search** 

* [**Auto-PyTorch**](https://github.com/automl/Auto-PyTorch)
* [**AutoKeras**](https://autokeras.com/)
* [**DEvol**](https://github.com/joeddav/devol)
* [**HyperAS**](https://github.com/maxpumperla/hyperas)**: a combination of Keras and Hyperopt**
* [**talos**](https://github.com/autonomio/talos)**: Hyperparameter Scanning and Optimization for Keras**

