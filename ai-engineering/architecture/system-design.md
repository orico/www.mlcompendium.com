# System Design

A model service still has to scale, and the system design interview is where that is tested. The page starts with the reading that teaches the vocabulary, moves to interview walkthroughs and scalability write-ups, then guided practice, and ends with the question notes themselves: the rate limiter and Twitter.

The same notes are in [System design](system-design.md#system-design).

Two starting points are marked as amazing. [System Design in a Hurry](https://www.hellointerview.com/learn/system-design/in-a-hurry/introduction) is Hello Interview's fast pass through the subject, and [System Design and architecture](https://github.com/puncsky/system-design-and-architecture) is puncsky's repo for learning how to design large-scale systems and preparing for the system design interview. The figure below comes from that repo.

<figure><img src="../../ops/.gitbook/assets/gimg-3e29c94f1bbb.png" alt=""><figcaption><p>Puncsky</p><p>Credit: <a href="https://lh4.googleusercontent.com/UpUWT6Jx5fE_l9YMZQuCa4c_-OlpXYYZldQzF3l1A2H-B2Sd8uMQAdekbeeIXNdKdooKShlFSN6bncvUjvOUB3F7-_drbrFx9ZO6kQ4rrcRoWqyNJjZ33X_3ewgs0leEndte3Xhw">copied from the original hosted image.</a></p></figcaption></figure>

Once the vocabulary is there, the interview format itself needs explaining. [Hired in tech](https://www.hiredintech.com/system-design/) has a System Design section, and [Episode 06: Intro to Architecture and Systems Design Interviews](https://www.youtube.com/watch?v=ZgdS0EUmn70) is Jackson Gabbard's video introduction to that kind of interview. A [System design template](https://leetcode.com/discuss/career/229177/My-System-Design-Template) gives the answer a fixed shape. For an ML service the design often includes retrieval: [RAG SD](https://blogs.nvidia.com/blog/what-is-retrieval-augmented-generation/) is Rick Merritt's NVIDIA explainer of retrieval-augmented generation, a technique for making generative AI models more accurate and reliable with facts fetched from external sources.

Jordan's interview series then works single problems end to end, each as a Systems Design Interview Questions With Ex-Google SWE episode by Jordan has no life: [Distributed Job Scheduler](https://www.youtube.com/watch?v=pzDwYHRzEnk) is episode 20, [Distributed Locking](https://www.youtube.com/watch?v=Lp8oITg0MiI) is episode 21, and [Rate Limiter](https://www.youtube.com/watch?v=VzW41m4USGs) is episode 7, the same rate limiter question that closes this page.

The broadest single reference is [(great) System Design Primer](https://github.com/donnemartin/system-design-primer/tree/master) — Learn how to design large-scale systems. Prep for the system design interview. Includes Anki flashcards.

Every design question comes back to scalability, so the next block is about that one word. [CS75 (Summer 2012) Lecture 9 Scalability Harvard Web Development David Malan](https://www.youtube.com/watch?v=-W9F__D3oY4) is the lecture, posted by George Schott. [a word about](https://www.allthingsdistributed.com/2006/03/a_word_on_scalability.html) is "A Word on Scalability", which points out that "that doesn't scale" is often used as a magic word to end an argument, when it usually means the architecture limits how far the service can grow. [scalability for dummies](https://web.archive.org/web/20221030091841/http://www.lecloud.net/tagged/scalability/chrono) is the archived lecloud series on making a web service massively scalable, and its parts build on each other: [1](https://web.archive.org/web/20220530193911/https://www.lecloud.net/post/7295452622/scalability-for-dummies-part-1-clones) is clones, so servers scale horizontally; [2](https://web.archive.org/web/20220602114024/https://www.lecloud.net/post/7994751381/scalability-for-dummies-part-2-database) is the database, for when the application gets slower even after that; [3](https://web.archive.org/web/20230126233752/https://www.lecloud.net/post/9246290032/scalability-for-dummies-part-3-cache) is the cache, for when users still wait on a scalable database; and [4](https://web.archive.org/web/20220926171507/https://www.lecloud.net/post/9699762917/scalability-for-dummies-part-4-asynchronism) is asynchronism, told through a bakery that bakes the bread at night. [Scalability, Availability & Stability Patterns](https://www.slideshare.net/slideshow/scalability-availability-stability-patterns/4062682) (good for topics) is a slide deck that names the trade-offs, performance vs scalability, latency vs throughput, and availability vs consistency, and then the patterns for each. To pick between designs, [Use Back-of-the-envelope-calculations to Choose the Best Design](https://highscalability.com/google-pro-tip-use-back-of-the-envelope-calculations-to-choo/) (good) is the High Scalability answer to how you know which is the "best" design for a given problem.

Reading only goes so far, so the last step is practice. [System Design Guided Practice](https://bugfree.ai/) — AI-Powered platform provided guided practice on system design problem and behavior questions like the way you do at Leetcode. [PracHub System Design Practice](https://prachub.com/categories/system-design) — Company-tagged interview prompts for timed, answer-first practice.


## System design

After the reading list, these notes are the interview questions themselves: rate limiter and Twitter.

The same notes are in [System Design](system-design.md).

The worked solutions come from the primer's section on [How to design a](https://github.com/donnemartin/system-design-primer#system-design-interview-questions-with-solutions) system, which pairs each interview question with a solution.

<figure><img src="../../ops/.gitbook/assets/9" alt=""><figcaption><p>System design</p></figcaption></figure>

The first question is the one from Jordan's episode 7: how would you design and implement an API rate limiter? The second is [The twitter question](https://github.com/donnemartin/system-design-primer/blob/master/solutions/system_design/twitter/README.md), the primer's worked solution for designing Twitter, drawn below.

<figure><img src="../../ops/.gitbook/assets/10" alt=""><figcaption><p>The twitter question</p></figcaption></figure>
