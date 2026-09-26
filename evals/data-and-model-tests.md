# Data & Model Tests

A score is only trustworthy if the data feeding it and the code producing it are checked every time the pipeline runs. The page starts with data testing, Great Expectations-style checks and data profiling, and then moves to model testing with unit tests, pytest, and mocks.

The same notes are in [Lakes and warehouses](../data/engineering/lakes-and-warehouses.md#lakes-and-warehouses) and [Pandas](../data/engineering/data-science-tools.md#pandas).

## Data Testing

The data comes first, because a broken table breaks every model trained on it. The same notes are in [Data Quality](../data/engineering/data-quality.md) and [Data Validation](../data/engineering/data-quality.md#data-validation).

[Great expectations](https://greatexpectations.io/) is GX Core, an open source framework for testing, validating, and documenting data quality across modern data pipelines, workflows, and teams. The GitHub [article](https://github.blog/2020-10-01-keeping-your-data-pipelines-healthy-with-the-great-expectations-github-action/) by Hamel Husain, the second in a series on using GitHub for MLOps and data science, is about keeping your data pipelines healthy with the Great Expectations GitHub Action, the point where those checks join continuous integration. The [Youtube](https://www.youtube.com/watch?v=uM9DB2ca8T8) video is the Great Expectations 101 getting-started webinar, and it is the “TDDA” for Unit tests and CI note on this page. Before writing expectations, it helps to know what is in the table: the [DataProfiler git](https://github.com/capitalone/DataProfiler) repo from capitalone asks "What's in your data?" and extracts schema, statistics and entities from datasets.

## Model Testing

Once the data is checked, the code that trains and serves the model needs the same discipline. The same notes are in [Continuous Integration](../ai-engineering/devops/full-stack-and-ops/continuous-integration.md).

The starting point is A great :P [unit test and logging](https://cohenori.medium.com/unit-testing-and-logging-for-data-science-d7fb8fd5d217) post on medium - it's actually mine :) After that, a mind blowing [lecture](https://www.youtube.com/watch?v=1fHGXOfiDO0&feature=youtu.be) about unit testing your data using Voluptuous & engrade & TDDA lecture ties the model tests back to the data tests above.

The Python tooling follows. [Unit tests in python](https://jeffknupp.com/blog/2013/12/09/improve-your-python-understanding-unit-testing/) is the written introduction. [Unit tests in python - youtube](https://www.youtube.com/watch?v=6tNS--WetLI) is Corey Schafer's Python Tutorial: Unit Testing Your Code with the unittest Module. [Unit tests asserts](https://docs.python.org/3/library/unittest.html#unittest.TestCase.debug) is the unittest documentation, the unit testing framework whose list of assert methods is the part to skip to if you already know the basic concepts of testing. [Auger - automatic unit tests, has a blog post inside](https://github.com/laffra/auger) is Automated Unittest Generation for Python, but it doesn't work with py 3+.

For data science specifically, [A rather naive unit tests article aimed for DS](https://medium.com/@danielhen/unit-tests-for-data-science-the-main-use-cases-1928d9e7a4d4) is Daniel Hen on when we should use unit testing for data science: the question is not if a DS pipeline needs unit tests, but what to look out for when writing them. A good pytest [tutorial](https://www.tutorialspoint.com/pytest/index.htm) covers Pytest, a testing framework for Python that makes it easy to write simple and scalable test cases, with fixtures, parameterized tests, and detailed error reporting.

Tests that touch outside services need mocks. [Mock](https://medium.com/@yasufumy/python-mock-basics-674c33de1ced) is Let’s be Friends with Mock in Python, on `unittest.mock` and why a library that talks to Google Spreadsheet should not connect to it in every test. [mock 2](https://medium.com/python-pandemonium/python-mocking-you-are-a-tricksy-beast-6c4a1f8d19b2) is Matt Pease's Python Mocking, You Are A Tricksy Beast: mocking is unintuitive and takes several “Ah, now I get it!” moments, but it is a powerful tool for enabling automated unit tests.
