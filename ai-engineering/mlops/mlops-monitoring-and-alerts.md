# ML Model Monitoring & Alerts

A model that ships without eyes goes blind. The page first covers why and how to watch models, dependencies, and features in production, then names drift, the change those alerts are for, and ends with the landscapes that compare monitoring tools.

## Monitoring & alerts

Before drift has a name, the first job is to watch models, dependencies, features, and production performance at all.

The case for it is my own series. [Monitor! Stop being a blind DS](https://cohenori.medium.com/monitor-stop-being-a-blind-data-scientist-ac915286075f) explains the many use cases and the monumental importance of monitoring and alerts from a data-science-researcher point of view, and reviews companies that try to solve this huge but strangely undiscussed problem. A model also breaks when what it depends on breaks, which is the point of [Monitor your dependencies! Stop being a blind DS](https://cohenori.medium.com/monitor-your-dependencies-stop-being-a-blind-data-scientist-a3150bd64594), with dominos as the analogy. The same argument aimed at leadership is [Data science observability for executives](https://cohenori.medium.com/data-science-observability-for-executives-a054411faecc).

Once you agree to watch, the question is what to watch. [Production Machine Learning Monitoring: Outliers, Drift, Explainers & Statistical Performance](https://medium.com/data-science/production-machine-learning-monitoring-outliers-drift-explainers-statistical-performance-d9b1d02ac158) starts from the line "The lifecycle of a machine learning model only begins once it’s in production". Alejandro Saucedo's talk on the same subject, Production Machine Learning Monitoring: Principles, Patterns and Techniques, is on [youtube](https://www.youtube.com/watch?v=QcevzK9ZuDg); it uses [alibi-explain](https://docs.google.com/document/d/1dXELAcJn9KCPSRMDvZoumUyHx8K8Yn7wfFxesSpbNCM/edit#heading=h.xs1o8m3ro5iy) uses see compendium and Ali detect see compendium.

The logging side of the same problem is [MLflow, HyperparameterHunter, Hyperopt, concept drift, unit tests](https://medium.com/data-science/putting-ml-in-production-ii-logging-and-monitoring-algorithms-91f174044e4e) Javier Rodriguez Zaurin. When there are many models, Yaron Gueta's [Meta anomaly over multiple models, aggregate](https://www.anodot.com/blog/monitoring-machine-learning/) answers how to track machine learning algorithms with machine learning algorithms. What happens after your machine learning model is deployed? [Vidhya on monitoring data & models](https://www.analyticsvidhya.com/blog/2019/10/deployed-machine-learning-model-post-production-monitoring/) is a framework for planning post-deployment monitoring. Features drift too: production feature distributions can move away from the baseline because the world changes, and [Monitor ML features using Amazon SageMaker Feature Store and AWS Glue DataBrew](https://medium.com/data-science/monitor-ml-features-using-amazon-sagemaker-feature-store-and-aws-glue-databrew-c530abcc479a) is how to catch that when features are shared across teams.

The same notes are in [Feature Stores & Feature Pipelines](feature-stores-and-feature-pipelines.md).

### Drift

Once monitoring is on, drift is the change in data or labels that those alerts are for.

The same notes are in [Anomaly Detection](../../predictive-ml/anomaly-detection.md), [Comparing distributions (distance methods)](../../data/distribution.md#comparing-distributions-distance-methods), [General patterns](mlops.md#general-patterns), [Hyper Parameter Optimization](../../evals/hyper-parameter-optimization.md), and [Training Strategies](../../evals/training-strategies.md).

Drift comes in two kinds, and the first two readings separate them. Philip Tannor's [Data & concept drifts](https://deepchecks.com/how-to-monitor-ml-models-in-production/) focuses on monitoring models once they are fully deployed and running in production. [2](https://www.explorium.ai/blog/understanding-and-handling-data-and-concept-drift/) is Explorium on understanding and handling data drift and concept drift. The hard case is when labels are missing: (good) [Inferring Concept Drift Without Labeled Data](https://concept-drift.fastforwardlabs.com/). Also talks about stream-based drift by Cloudera — Fast Forward Labs.

Measuring drift is where Arize.ai's articles come in. Data, concept, [feature drifts](https://medium.com/data-science/using-statistical-distance-metrics-for-machine-learning-observability-4c874cded78) — various comparisons between train/prod/validation time windows, diff models, A/B testing etc., and how to measure drifts, using statistical distances on inputs, outputs, and actuals.

The same notes are in [A/B Testing](../../decision-intelligence/a-b-testing.md).

The same authors argue that teams rushing ML into production lack the right tools, and name the three they need: [Model store, Feature store, evaluation store](https://medium.com/data-science/the-only-3-ml-tools-you-need-1aa750778d33). [Monitor model performance in production](https://medium.com/data-science/the-playbook-to-monitor-your-models-performance-in-production-ec06c1cc3245) — real-time, biased, delayed, and no ground truth. The statistical-distance article also works as a list of [use cases — i.e., how to use statistical differences/distances](https://medium.com/data-science/using-statistical-distance-metrics-for-machine-learning-observability-4c874cded78).

Detecting drift raises the question of what to do next. [Some advice on Medium](https://medium.com/data-science/concept-drift-and-model-decay-in-machine-learning-a98a809ea8d4), relabel using latest model (can we even trust it?) retrain after. Uber argues that waiting is the mistake, in [Adversarial Validation Approach to Concept Drift Problem in User Targeting Automation Systems at Uber](https://arxiv.org/abs/2004.03045) — Previous research on concept drift mostly proposed model retraining after observing performance decreases. However, this approach is suboptimal because the system fixes the problem only after suffering from poor performance on new data. Here, we introduce an adversarial validation approach to concept drift problems in user targeting automation systems. With our approach, the system detects concept drift in new data before making inference, trains a model, and produces predictions adapted to the new data.

The same idea can be coded directly: a drift estimator between data sets using random forest; the formula is in the Medium article above; code here at [MLBox](https://github.com/AxeldeRomblay/MLBox/blob/811dbcb04fc7f5501e82f3e78aa6c119f426ee78/python-package/mlbox/preprocessing/drift/drift_estimator.py). For a maintained library, [Alibi-detect](https://docs.google.com/document/d/1dXELAcJn9KCPSRMDvZoumUyHx8K8Yn7wfFxesSpbNCM/edit#heading=h.y6mpsp4co5t9) — is an open-source Python library focused on outlier, adversarial, and drift detection, by Seldon. And for why drift is missed in the first place, [What is concept drift and why does it go undetected](http://web.archive.org/web/20240416043659/https://censius.ai/blogs/what-is-concept-drift-and-why-does-it-go-undetected). Breaking down concept drit and explaining the best methods to avoid it.

The figure below, copied from the original hosted image, lists Alibi Detect's drift features.

<figure><img src="../../ops/.gitbook/assets/gimg-0cd60b66f9db.png" alt=""><figcaption><p>Alibi Detection Drift Features</p><p>Credit: <a href="https://lh4.googleusercontent.com/sASV5qq3CTmv0gx6Tl3DiwACMnwsW9wj1yNHF5sFIFbQr4BFFgAVgfcWsnrHxNnQtQKa-b5-IdbC-OElnQIr117lxaH3TGCuz1CmpgU6mof3i9VkPR3LyzdD9S0ujTmWj7o88Iep">copied from the original hosted image.</a></p></figcaption></figure>

### Tool comparisons

With drift named, the remaining question is which tool does the watching, so these lists compare the monitoring landscapes and tool shelves.

The market map I built is at https://www.stateofmlops.com, the [State of MLOps](https://www.stateofmlops.com) card view, and (by me) [Medium](https://cohenori.medium.com/mlops-monitoring-market-review-66904f0863bb) is the MLOps Monitoring Market Review 2021 that goes with it. The same review has an article, open-source [Airtable](https://airtable.com/shr4rfiuOIVjMhvhL) version on Airtable's low-code app platform. A curated list of MLOps projects that used to sit here now points to an unrelated site and is kept at the end of the page.

Other vendors and communities drew their own maps. [Aporia](http://web.archive.org/web/20260725060446/https://www.aporia.com/) kept one; the archived address now announces that Aporia has been acquired by Coralogix. [Neptune.AI](http://web.archive.org/web/20230610222659/https://mlops.neptune.ai/) is the MLOPS Tools Landscape by Neptune. [Ambiata](https://www.ambiata.com/blog/2020-12-07-mlops-tools/) wrote an MLOps tools post; the address now shows Ambiata's AI staff augmentation services. [LakeFS](https://lakefs.io/the-state-of-data-engineering-in-2021/) on the state of data engineering — has monitoring and observability inside. [The NLP Pandect](https://github.com/ivan-bilan/The-NLP-Pandect#mlops-for-nlp), a comprehensive reference for all topics related to Natural Language Processing, has a section on MLOps for NLP. [ml-ops.org](https://ml-ops.org/) is the community site, and [Awesome production ML](https://github.com/EthicalML/awesome-production-machine-learning/) is the curated list of open source libraries to deploy, monitor, version, and scale machine learning, shown in the figure below.

<figure><img src="../../ops/.gitbook/assets/image (33).png" alt=""><figcaption><p>Awesome production ML</p></figcaption></figure>

Two of the articles above also live at their original addresses. Monitor ML features using Amazon SageMaker Feature Store and AWS Glue DataBrew. [https://towardsdatascience.com/monitor-ml-features-using-amazon-sagemaker-feature-store-and-aws-glue-databrew-c530abcc479a](https://towardsdatascience.com/monitor-ml-features-using-amazon-sagemaker-feature-store-and-aws-glue-databrew-c530abcc479a). Concept drift is a drift of labels with time for the essentially the same data, by Ashok Chilakapati, and that is the some advice on medium article. [https://towardsdatascience.com/concept-drift-and-model-decay-in-machine-learning-a98a809ea8d4](https://towardsdatascience.com/concept-drift-and-model-decay-in-machine-learning-a98a809ea8d4)

Monitoring does not stop at alerts. Modern distributed, event-driven, multi-cloud systems generate more signals and failure modes than human operators can manage in real time, so [Self-Healing Agentic Systems](https://cohenori.medium.com/the-rise-of-self-healing-systems-fe653869b7fc) (January 2026) is the monitoring note for a system that repairs itself, and the agent side is on the agents page.

For a concrete starting point, [Monitor! Easy MLOps Model Monitoring With New Relic](https://cohenori.medium.com/monitor-easy-mlops-model-monitoring-with-new-relic-ef2a9b611bd1) (April 2022) is a monitoring setup on this page.

## Deprecated links

{% hint style="warning" %}
These links and images no longer work. The original wording is kept here. A same-resource copy, when one was checked, is used above.
{% endhint %}


- Production Machine Learning Monitoring: Outliers, Drift, Explainers & Statistical Performance. This address no longer opens: https://towardsdatascience.com/production-machine-learning-monitoring-outliers-drift-explainers-statistical-performance-d9b1d02ac158
- Mlflow, Hyperparameterhunter,hyperopt, concept drift, unit tests. This address no longer opens: https://towardsdatascience.com/putting-ml-in-production-ii-logging-and-monitoring-algorithms-91f174044e4e
- feature drifts. This address no longer opens: https://towardsdatascience.com/using-statistical-distance-metrics-for-machine-learning-observability-4c874cded78
- Model store, Feature store, evaluation store. This address no longer opens: https://towardsdatascience.com/the-only-3-ml-tools-you-need-1aa750778d33
- Monitor model performance in production. This address no longer opens: https://towardsdatascience.com/the-playbook-to-monitor-your-models-performance-in-production-ec06c1cc3245
- use cases - i.e., how to use statistical differences/distances. This address no longer opens: https://towardsdatascience.com/using-statistical-distance-metrics-for-machine-learning-observability-4c874cded78
- What is concept drift and why does it go undetected. Breaking down concept drit and explaining the best methods to avoid it. This address no longer opens: https://censius.ai/blogs/what-is-concept-drift-and-why-does-it-go-undetected
- How does data drift hamper AI performance. Understand how data drift affect peak AI performance and how you can detect it. This address no longer opens: https://censius.ai/blogs/data-drift-barrier-to-ai-performance
- by Aporia. This address no longer opens: https://aporia.com
- Neptune.AI MLOPS tools landscape. This address no longer opens: https://mlops.neptune.ai/
- Twimlai ML AI solutions. This address no longer opens: https://twimlai.com/solutions/
- MLOps.toys. This address now points to an unrelated site: https://mlops.toys/
