# Definitions

Ops names get mixed: DataOps, MLOps, GitOps, DevOps, and DevSecOps. The page first lets a ChatGPT answer separate those professions, then a comparison block places DevOps next to SRE, CloudOps, ITOps, and AIOps.

The same notes are in [Definitions](devsecops/tbd.md), [DevOps Strategy](devops/devops-strategy.md), [MLOps Intro](mlops/mlops-intro.md), and [Ops Definition Comparisons](definitions.md).

The quickest way to see the five names side by side was to ask one question and keep the whole reply. I asked ChatGPT: "what are DataOps, MLOps, GitOps, DevOps and the DevSecOps professions?"

The answer:

DataOps, MLOps, DevOps and GitOps are specialized areas within the field of operations management. DevSecOps is a combination of these areas with a focus on security.

DataOps is a set of practices that combines data management (data) and information technology operations (Ops). It aims to improve the speed and reliability of data analytics by automating data pipelines and promoting collaboration between data engineers, data scientists, and IT operations professionals.

MLOps, or machine learning operations, is a set of practices that combines machine learning (ML) and information technology operations (Ops). It aims to improve the speed and reliability of deploying machine learning models into production environments by automating the build, test, and release process, and promoting collaboration between data scientists, machine learning engineers, and IT operations professionals.

GitOps is a set of practices that uses Git as a single source of truth for declarative infrastructure and application code. It aims to improve the speed and reliability of deploying applications by using Git as the central source of truth for the entire delivery process, from development to production.

DevOps is a set of practices that combines software development (Dev) and information technology operations (Ops). It aims to shorten the development lifecycle and provide continuous delivery and integration of software.

DevSecOps is a set of practices that combines DevOps with a focus on security (Sec). It aims to improve the security of the software development and delivery process by integrating security practices into the development lifecycle and promoting collaboration between development, security, and operations teams.

Generally, individuals working in these specialized areas of operations management may be referred to as DevOps professionals, DataOps professionals, MLOps professionals, GitOps professionals, or DevSecOps professionals, depending on their specific area of expertise.
# Ops Definition Comparisons

With the five names defined, the harder question is where DevOps ends and each neighboring operations role begins, and what AIOps adds on top.

The same notes are in [Definitions](definitions.md).

Google's own history is the place to start: [Dev vs ops, vs devops vs sre: history and details by Google.](https://www.youtube.com/watch?v=tEylFyxbDLE&list=PLIivdWyY5sqJrKl7D2u-gmis8h9K66qoj&index=2) is the Google Cloud Tech video "SLIs, SLOs, SLAs, oh my!", framed as class SRE implements DevOps. Because SRE and DevOps both seem to rule software development and overlap enough to confuse people, [SRE vs DevOps](https://medium.com/hackernoon/sre-vs-devops-the-dilemma-f7054714525c) is Pavan Belagatti's attempt to find the differentiating factors between the two. [CloudOps vs DevOps](https://victorops.com/blog/what-is-cloudops-vs-devops) was a VictorOps post; the address now lands on Splunk's note that VictorOps is now Splunk On-Call. [ITOps vs DevOps](https://www.graylog.org/post/itops-vs-devops-what-is-the-difference) is The Graylog Team asking where one role ends and the other begins, and what each encompasses. [AIOps](https://www.appdynamics.com/what-is-ai-ops/) was an AppDynamics page, which now sits in Splunk's observability portfolio and pitches unified observability across any environment and stack, finding root causes and resolving problems proactively. For a simple visual answer, the DZone [Definition](http://web.archive.org/web/20231001113410/https://dzone.com/articles/dev-vs-ops-and-devops) notes that the buzz about DevOps is still dominated by conversations describing what it is, and draws its own description as an image.

The AIOps link above needs a sharper definition than a product page gives, so here it is in full:

> AIOps platforms utilize big data, modern machine learning and other advanced analytics technologies to directly and indirectly enhance IT operations (monitoring, automation and service desk) functions with proactive, personal and dynamic insight. AIOps platforms enable the concurrent use of multiple data sources, data collection methods, analytical (real-time and deep) technologies, and presentation technologies.

The figure below puts the Ops definitions side by side in one picture; its credit says it was copied from the original hosted image.

<figure><img src="../ops/.gitbook/assets/gimg-8690ceb743f8.png" alt=""><figcaption><p>Ops definition comparisons</p><p>Credit: <a href="https://lh3.googleusercontent.com/-q3xPZ_ASRimnV37VYLqPZxoKFSQPKSrkIQdnBHaxCPOkP9rTZT7t-6n98Zp4NKwG8QuuFNlk4omZv234Dx8QrBohcVzh7kLoiOwYmfHF5skCBKt6q8zRaHZrn2r481i3QXzr7hH">copied from the original hosted image.</a></p></figcaption></figure>

## Deprecated links

{% hint style="warning" %}
These links and images no longer work. The original wording is kept here. A same-resource copy, when one was checked, is used above.
{% endhint %}

- Definition. This address no longer opens: https://dzone.com/articles/dev-vs-ops-and-devops
