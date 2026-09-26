# Analytics

Probability and dependence decide what a later score means.
After this chapter the reader can read a distribution, a dependence, and an information measure, and can choose a feature for a reason.

The chapter starts with the shapes under the data. [Probability](probability.md) gives the basic probability pictures and the PDF they become, then kernel density estimation for when a histogram is sparse. A dataset always sits under some probability shape, so [Distribution](distribution.md) names that shape, looks at the Gaussian that so many tests assume, and then compares two distributions, including by distance. [Probability & Statistics](probability-and-statistics.md) steps back to the difference between the two fields: probability asks what happens next under a random process, and statistics asks what process would explain what already happened.

With the shapes named, the chapter turns to measures. A model that ranks classes or splits trees needs a measure of surprise and a measure of how wrong a predicted distribution is, and [Information Theory](information-theory.md) supplies entropy, information gain, cross-entropy, the divergence family, and softmax. [Data Analytics](analytics/data-analytics.md) is the practical path from raw tables to questions a team can answer without a model yet, laid out as a free course. The chapter ends with [Dependence and Selection](dependence-and-selection.md): features that move together waste capacity and features that do not predict the target waste the model, so that page measures dependence and then selects.

Analysis does not have to go through code alone. [Understanding PandasAI](https://cohenori.medium.com/understanding-pandasai-fc135b871e84) (May 2023) shows analysis through a generative interface: Pandas-AI is a Python library that adds generative AI to Pandas and makes data frames conversational, and the piece looks at how an LLM can be steered to answer data questions.
