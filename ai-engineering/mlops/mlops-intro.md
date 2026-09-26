# MLOps Intro

ML in a notebook is not ML in production. The page closes that gap in four steps: the basic building blocks and a curated reading list, then the Analytics Vidhya guides from beginner to end-to-end architecture, and finally the MLOps without Ops series for companies that are not Big Tech.

The same notes are in [Definitions](../definitions.md) and [Life cycle](../../business-problems/data-science.md#life-cycle).

Before any tool, you need the vocabulary. A [great intro about MLOps](https://betterprogramming.pub/mlops-and-mlflops-795781d17989) lays out what are the basic building blocks that you need to understand, in comparison to Data Engineering & DevOps — by Andrew Blance. Once those blocks have names, [Awesome MLOps on GitHub](https://github.com/visenger/awesome-mlops) is the curated list of references for MLOps to branch out from.

A reference list is wide but not ordered, so the next step is one publication that walks the topic from the start. Analytics Vidhya has run enough MLOps pieces that [A list of MLOps articles based on the "MLOps" label](https://medium.com/analytics-vidhya/tagged/mlops) is a reading queue on its own. The entry point there is ASHWINI's beginners guide to machine learning operations, [A beginner guide](https://www.analyticsvidhya.com/blog/2021/06/mlops-a-beginners-guide-to-machine-learning-operations/), which shows how to set a model up in production and then maintain and monitor it. The same article is also [A comprehensive guide](https://www.analyticsvidhya.com/blog/2021/06/mlops-a-beginners-guide-to-machine-learning-operations/) — there are quite a lot of details in this article that you should know only if you truly know the basics of ML lifecycle, feature engineering, deployment strategies etc.

With the basics in hand, Illiyas's two-part series follows a generic MLOps workflow for building, deploying, and monitoring ML applications. Part 1 is about [connecting agile, data, devops, and technology](https://www.analyticsvidhya.com/blog/2022/02/mlops-part-1-revealing-the-approach-behind-mlops/), the fundamentals behind the approach. Part 2 walks the workflow from start to end, step by step, [going deeper into architecture, deployment, training](https://www.analyticsvidhya.com/blog/2022/02/workflow-of-mlops-part-2-model-building/).

That workflow raises the obvious question of MLOps vs DevOps. [The tip of the iceberg](https://www.analyticsvidhya.com/blog/2022/09/how-is-mlops-different-from-devops/) is the short answer, i.e., adding model + data to DevOps methodologies. [Another comparison, has some more details](https://www.analyticsvidhya.com/blog/2020/11/mlops-the-why-and-the-what/): the ML model represents a small fraction of the components that comprise an enterprise production deployment workflow, and the rest is why MLOps exists. Those components have to run somewhere, and Gitesh Dhore's [MLOps & Kubernetes](https://www.analyticsvidhya.com/blog/2022/09/mlops-and-use-of-kubernetes/) frames MLOps as collaboration between data scientists and the operations unit within an organization, run on Kubernetes.

The same notes are in [Kubeflow](../devops/full-stack-and-ops/kubernetes.md#kubeflow).

Kubernetes is one piece; the last two Analytics Vidhya guides put all the pieces on one diagram. Sarvagya Agrawal's [High level E2E architecture and explanations](https://www.analyticsvidhya.com/blog/2023/02/mlops-end-to-end-mlops-architecture-and-workflow/) explains the complete MLOps pipeline that automates data science projects. Mukul Kirti's [High level E2E concepts](https://www.analyticsvidhya.com/blog/2021/07/deepdive-into-the-emerging-concpet-of-machine-learning-operations-or-mlops/) describes MLOps as what minimizes the gap between data scientists and production teams, cutting waste and making ML systems more scalable.

Those architectures assume a platform team. If you do not work for Big Tech, the Googles, Facebooks, and Amazons of this world, chances are you work for a "reasonable scale" company, and that is the premise of the MLOps without Ops series. [Part 1](https://medium.com/data-science/mlops-without-much-ops-d17f502f76e8) sets out that premise. [Part 2](https://medium.com/data-science/ml-and-mlops-at-a-reasonable-scale-31d2c0782d9c) is ML and MLOps at a reasonable scale. [Part 3](https://medium.com/data-science/hagakure-for-mlops-the-four-pillars-of-ml-at-reasonable-scale-5a09bd073da) is by Ciro Greco. [Part 4](https://medium.com/data-science/the-modern-data-pattern-d34d42216c81) is Luca Bigon's modern data pattern. Beside the series, Cristiano Breuel's [As an engineering discipline](https://medium.com/data-science/ml-ops-machine-learning-as-an-engineering-discipline-b86ca4874a3f) starts where many companies stand: a talented team of data scientists, great metrics, jaw-dropping demos, and executives asking how soon the model can be in production.

The same series also lives at its original Towards Data Science addresses, each episode with its own title:

- ML and MLOps at a Reasonable Scale, by Ciro Greco, episode 2 of MLOps without much Ops. Part 2. [https://towardsdatascience.com/ml-and-mlops-at-a-reasonable-scale-31d2c0782d9c](https://towardsdatascience.com/ml-and-mlops-at-a-reasonable-scale-31d2c0782d9c)
- Hagakure for MLOps: the four pillars of ML at Reasonable Scale, by Ciro Greco, episode 3. Part 3. [https://towardsdatascience.com/hagakure-for-mlops-the-four-pillars-of-ml-at-reasonable-scale-5a09bd073da](https://towardsdatascience.com/hagakure-for-mlops-the-four-pillars-of-ml-at-reasonable-scale-5a09bd073da)
- Replyable data processing and ingestion at scale with serverless, Snowflake and dot, by Luca Bigon. Part 4. [https://towardsdatascience.com/the-modern-data-pattern-d34d42216c81](https://towardsdatascience.com/the-modern-data-pattern-d34d42216c81)
- Towards Data Science. As an engineering discipline. [https://towardsdatascience.com/ml-ops-machine-learning-as-an-engineering-discipline-b86ca4874a3f](https://towardsdatascience.com/ml-ops-machine-learning-as-an-engineering-discipline-b86ca4874a3f)

## Deprecated links

{% hint style="warning" %}
These links and images no longer work. The original wording is kept here. A same-resource copy, when one was checked, is used above.
{% endhint %}

- MLOps without Ops series Part 1. This address no longer opens: https://towardsdatascience.com/mlops-without-much-ops-d17f502f76e8
