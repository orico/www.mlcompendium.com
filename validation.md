# Data & Model Validation

### DATASETS RELIABILITY & CORRECTNESS&#x20;

1\. [Clever Hans effect](https://thegradient.pub/nlps-clever-hans-moment-has-arrived/?fbclid=IwAR3vSx9EjXcSPhXU3Jyf7aWpTqhbVDARnh3qGpSw0rysv9rLeGyZFFCPnJA) - in relations to cues left in the dataset that models find, instead of actually solving the defined task!

* Ablating, i.e. removing, part of a model and observing the impact this has on performance is a common method for verifying that the part in question is useful. If performance doesn't go down, then the part is useless and should be removed. Carrying this method over to datasets, it should become common practice to perform dataset ablations, as well, for example:
* Provide only incomplete input (as done in the reviewed paper): This verifies that the complete input is required. If not, the dataset contains cues that allow taking shortcuts.
* Shuffle the input: This verifies the importance of word (or sentence) order. If a bag-of-words/sentences gives similar results, even though the task requires sequential reasoning, then the model has not learned sequential reasoning and the dataset contains cues that allow the model to "solve" the task without it.
* Assign random labels: How much does performance drop if ten percent of instances are relabeled randomly? How much with all random labels? If scores don't change much, the model probably didn't learning anything interesting about the task.
* Randomly replace content words: How much does performance drop if all noun phrases and/or verb phrases are replaced with random noun phrases and verbs? If not much, the dataset may provide unintended non-content cues, such as sentence length or distribution of function words.

[2. Paper](https://arxiv.org/abs/1908.05267?fbclid=IwAR1xOHxCF3gewyijMYAfZJSysu88Y8lgRIT2OiG-jWQav4zcbPqGYSoFFkk)\


### UNIT / DATA TESTS

1. A great :P [unit test and logging](https://towardsdatascience.com/unit-testing-and-logging-for-data-science-d7fb8fd5d217?fbclid=IwAR3pze0DtV-2Q4L4ysPyjrInk7LB89mdiodxlEUTv4rv37ZoDzl\_2I4ZbgA) post on medium - it’s actually mine :)
2. A mind blowing [lecture](https://www.youtube.com/watch?v=1fHGXOfiDO0\&feature=youtu.be\&fbclid=IwAR1bKByLgdYBDoBEr-e6Pw0Un5o0wvOg1yp4C-q4AoWZ1QuBEopTFFn0Gdw) about unit testing your data using Voluptuous & engrade & TDDA lecture
3. [Great expectations](https://greatexpectations.io/), [article](https://github.blog/2020-10-01-keeping-your-data-pipelines-healthy-with-the-great-expectations-github-action/), “TDDA” for Unit tests and CI, [Youtube](https://www.youtube.com/watch?v=uM9DB2ca8T8)
4. [DataProfiler git](https://github.com/capitalone/DataProfiler)
5. [Unit tests in python](https://jeffknupp.com/blog/2013/12/09/improve-your-python-understanding-unit-testing/)
6. [Unit tests in python - youtube](https://www.youtube.com/watch?v=6tNS--WetLI)
7. [Unit tests asserts](https://docs.python.org/3/library/unittest.html#unittest.TestCase.debug)
8. [Auger - automatic unit tests, has a blog post inside](https://github.com/laffra/auger), doesn't work with py 3+
9. [A rather naive unit tests article aimed for DS](https://medium.com/@danielhen/unit-tests-for-data-science-the-main-use-cases-1928d9e7a4d4)
10. A good pytest [tutorial](https://www.tutorialspoint.com/pytest/index.htm)
11. [Mock](https://medium.com/@yasufumy/python-mock-basics-674c33de1ced), [mock 2](https://medium.com/python-pandemonium/python-mocking-you-are-a-tricksy-beast-6c4a1f8d19b2)

### WHY WE SHOULDN’T TRUST MODELS

1. [Clever Hans effect for NLP](https://thegradient.pub/nlps-clever-hans-moment-has-arrived/)
   1. Datasets need more love
   2. Datasets ablation and public beta
   3. Inter-prediction agreement
2. Behavioral testing and CHECKLIST
   1. [Blog](https://amitness.com/2020/07/checklist/), [Youtube](https://www.youtube.com/watch?v=L3gaWctPg6E), [paper](https://arxiv.org/pdf/2005.04118.pdf), [git](https://github.com/marcotcr/checklist)
   2. [Yonatan hadar on the subject in hebrew](https://www.facebook.com/groups/MDLI1/permalink/1627671704063538/)

