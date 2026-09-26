# ML Architecture

An ML system is more than a model file. The page walks end-to-end architecture examples in order of how much infrastructure you own: lambda serving, then no-ops PaaS setups, then a post-modern stack.

The same notes are in [Data Architecture](../../data/engineering/data-architecture.md), [Data Platforms](../../data/engineering/data-platforms.md), [MetaFlow](full-stack-and-ops.md#metaflow), and [Prefect](full-stack-and-ops.md#prefect).

The classic split between batch and real-time paths is the starting point. [Lambda architecture for ML serving / Training](https://www.youtube.com/watch?v=fPlgoTLJh38) is Mike Bernico's Lambda Architecture in 10 minutes or less.

Owning both paths is a lot of operations, so the next examples remove as much of it as they can. A PaaS [End-to-End ML Setup](https://github.com/jacopotagliabue/no-ops-machine-learning) with Metaflow, Serverless and SageMaker is the repo, and [No Ops ML](https://medium.com/data-science/noops-machine-learning-3893a42e32a4) is the article that goes with it. The same approach scales to a real task in an [end-to-end implementation](https://github.com/jacopotagliabue/you-dont-need-a-bigger-boat) of intent prediction with Metaflow and other cool tools.

The series ends by joining the modern data stack with the modern ML stack. [A post-modern stack](https://github.com/jacopotagliabue/post-modern-stack) is that repo, with an [article](https://medium.com/data-science/the-post-modern-stack-993ec3b044c1) — by Jacopo Tagliabue, speaks about DBT, Snowflake, S3, Comet, SageMaker, the last episode that recaps the themes of the earlier ones.

## Deprecated links

{% hint style="warning" %}
These links and images no longer work. The original wording is kept here. A same-resource copy, when one was checked, is used above.
{% endhint %}

- No Ops ML Article - by Jacopo Tagliabue. This address no longer opens: https://towardsdatascience.com/noops-machine-learning-3893a42e32a4
- Article - by Jacopo Tagliabue, speaks about DBT, Snowflake, S3, Comet, Sagemaker. This address no longer opens: https://towardsdatascience.com/the-post-modern-stack-993ec3b044c1
