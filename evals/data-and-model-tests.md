# Data & Model Tests

This page collects resources for validating data in ML pipelines and unit testing models. It covers Great Expectations-style data checks alongside pytest, mocks, and related model-test notes.

The same notes are in [Lakes and warehouses](../data/engineering/lakes-and-warehouses.md#lakes-and-warehouses) and [Pandas](../data/engineering/data-science-tools.md#pandas).

## Data Testing

This section points to Great Expectations, related articles, and DataProfiler for pipeline checks.

The same notes are in [Data Quality](../data/engineering/data-quality.md) and [Data Validation](../data/engineering/data-quality.md#data-validation).

- GX Core is an open source framework for testing, validating, and documenting data quality across modern data pipelines, workflows, and teams. [Great expectations](https://greatexpectations.io/)
- This post is the second in our series on using GitHub for MLOps and data science, by Hamel Husain. [article](https://github.blog/2020-10-01-keeping-your-data-pipelines-healthy-with-the-great-expectations-github-action/)
- “TDDA” for Unit tests and CI. [Youtube](https://www.youtube.com/watch?v=uM9DB2ca8T8)
- GitHub - capitalone/DataProfiler: What's in your data? Extract schema, statistics and entities from datasets. [DataProfiler git](https://github.com/capitalone/DataProfiler)

## Model Testing

This section lists posts and tutorials on unit tests, mocks, and pytest for data science code.

The same notes are in [Continuous Integration](../ai-engineering/devops/full-stack-and-ops/continuous-integration.md).

1. A great :P [unit test and logging](https://cohenori.medium.com/unit-testing-and-logging-for-data-science-d7fb8fd5d217) post on medium - it's actually mine :)
2. A mind blowing [lecture](https://www.youtube.com/watch?v=1fHGXOfiDO0&feature=youtu.be) about unit testing your data using Voluptuous & engrade & TDDA lecture
3. [Unit tests in python](https://jeffknupp.com/blog/2013/12/09/improve-your-python-understanding-unit-testing/)
- Python Tutorial: Unit Testing Your Code with the unittest Module, by Corey Schafer. [Unit tests in python - youtube](https://www.youtube.com/watch?v=6tNS--WetLI)
- Source code: Lib/unittest/__init__.py(If you are already familiar with the basic concepts of testing, you might want to skip to the list of assert methods.) The unittest unit testing framework was ... [Unit tests asserts](https://docs.python.org/3/library/unittest.html#unittest.TestCase.debug)
- Automated Unittest Generation for Python. doesn't work with py 3+ [Auger - automatic unit tests, has a blog post inside](https://github.com/laffra/auger)
- [A rather naive unit tests article aimed for DS](https://medium.com/@danielhen/unit-tests-for-data-science-the-main-use-cases-1928d9e7a4d4)
- Pytest is a testing framework for Python that makes it easy to write simple and scalable test cases. A good pytest [tutorial](https://www.tutorialspoint.com/pytest/index.htm)
- [Mock](https://medium.com/@yasufumy/python-mock-basics-674c33de1ced)
- [mock 2](https://medium.com/python-pandemonium/python-mocking-you-are-a-tricksy-beast-6c4a1f8d19b2)
