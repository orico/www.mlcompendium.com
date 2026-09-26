# MLOps Literature

Operations work has a shelf of books behind it, and reading them in order turns a pile of tools into a way of thinking. The list starts with the twelve-factor app, then data-intensive systems, SRE, and distributed systems, moves to Jez Humble's delivery books, and ends with code craft: clean code, domain-driven design, a Python course, and a talk archive.

## Twelve-factor

The first idea is how a single service should be built so that it can be operated at all. [12 Factors](https://12factor.net/). The Twelve-Factor App is a methodology for building scalable and portable SaaS applications. It emphasizes best practices like maintaining a single codebase, isolating dependencies, storing configuration in the environment, treating services as replaceable resources, and enabling fast, stateless, and resilient deployments across consistent environments.

## Data-intensive applications

Once one service is well built, the hard part moves to the data it reads and writes. [Designing Data-Intensive Applications: reliable, maintainable](https://www.amazon.com/Designing-Data-Intensive-Applications-Reliable-Maintainable/dp/1449373321) is the book, and the [Medium](https://medium.com/@m_mcclarty/tech-book-talk-designing-data-intensive-applications-eb4908f2f6d6) post is a Tech Book Talk review of it, part of a monthly series reviewing a technical book finished in the last month.

## SRE

Data systems that people depend on have to stay up, which is what site reliability engineering is about. [Google SRE](https://landing.google.com/sre/books/) is Google's page for its SRE books, a guide to SRE principles, best practices, and the role of a reliability engineer. The two books behind it are [Site Reliability Engineering](https://www.amazon.com/dp/149192912X?psc=1&pf_rd_p=0c07d3ef-dd9a-4ce4-8daa-7b9b90db3048&pf_rd_r=F4QGSBSPA6DJJCX10WXK&pd_rd_wg=RfKM8&pd_rd_i=149192912X&pd_rd_w=khBYM&pd_rd_r=5bd80e38-7a30-41bb-ae0e-25a91dd1cb3d&ref_=pd_luc_rh_crh_rh_sbs_sem_01_03_t_ttl_lh), the principles, and the [Site Reliability Workbook](https://www.amazon.com/Site-Reliability-Workbook-Practical-Implement/dp/1492029505/ref=sr_1_1?dchild=1&keywords=The+Site+Reliability+Workbook&link_code=qs&qid=1598257953&sr=8-1&tag=amznsearchff-20), the practical companion.

## Distributed systems and microservices

Reliability gets harder when the system is split across many machines and services. [Designing Distributed Systems: patterns and paradigms](https://www.amazon.com/Designing-Distributed-Systems-Patterns-Paradigms/dp/1491983647) is the book of distributed-system patterns, and [Building Microservices: Designing Fine-Grained Systems](https://www.amazon.com/dp/1491950358/?coliid=I1H3OSVXC7XRBL&colid=300M9JC4311P3&psc=0&ref_=lv_ov_lig_dp_it) is the book on cutting a system into fine-grained services.

## Jez Humble

With the architecture in place, the next question is how changes get delivered through it, and Jez Humble's books are the delivery shelf. [Jez Humble](https://www.amazon.com/Jez-Humble/e/B003SNGS8E/ref=dp_byline_cont_pop_book_2) is his Amazon author page with the full bibliography. From it, [Accelerate: software and high-performing organizations](https://www.amazon.com/Accelerate-Software-Performing-Technology-Organizations/dp/1942788339/ref=tmm_pap_swatch_0?_encoding=UTF8&qid=&sr=) is about software and high-performing organizations, the [DevOps Handbook](https://www.amazon.com/DevOps-Handbook-World-Class-Reliability-Organizations/dp/1942788002/ref=tmm_pap_swatch_0?_encoding=UTF8&qid=&sr=) is the DevOps book, and [Continuous Delivery: deployment](https://www.amazon.com/Continuous-Delivery-Deployment-Automation-Addison-Wesley-dp-0321601912/dp/0321601912/ref=mt_other?_encoding=UTF8&me=&qid=) is the deployment book. Lean Enterprise completes the list, without a link.

## Clean code

Delivery pipelines only help if the code moving through them is readable. [Clean Code](https://www.amazon.com/Clean-Code-Handbook-Software-Craftsmanship/dp/0132350882) (Java). The first chapters are good. For Python, [Clean Code in Python](https://www.packtpub.com/product/clean-code-in-python/9781788835831) starts from the point that there is no strict definition or formal measure of clean code: checkers, linters, and static analyzers are necessary, but not sufficient. The [Medium](https://medium.com/@m_mcclarty/tech-book-talk-clean-code-in-python-aa2c92c6564f) post is a book talk on that same Python book, and the [Git](https://github.com/zedr/clean-code-python) repo is zedr's :bathtub: Clean Code concepts adapted for Python.

## Domain-driven design

Clean functions still need a model of the business they serve. [Domain-Driven Design](https://www.amazon.com/Domain-Driven-Design-Reference-Definitions-Summaries/dp/1457501198/ref=pd_cart_crc_cko_mrai_1_1/146-7136232-7217867?_encoding=UTF8&pd_rd_i=1457501198&pd_rd_r=e0b19e54-c0c3-4ad1-abd0-eff99f815aee&pd_rd_w=TP8KJ&pd_rd_wg=ompBT&pf_rd_p=77f3805b-bff9-40ee-9688-bcdb2cd9e197&pf_rd_r=1WF281MJHVQYG506PTRX&psc=1&refRID=1WF281MJHVQYG506PTRX) is the reference, with its definitions and summaries.

## The Python course

For the language underneath most of this work, a single video course covers it end to end. [The Complete Python Course](https://www.packtpub.com/product/the-complete-python-course-video/9781839217289) is that video course.

## pyVideo

After the course, talks keep the learning going. [pyVideo](https://pyvideo.org/) is PyVideo.org, the archive of Python talks.
