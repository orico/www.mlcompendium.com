# Feature Types

A model needs to know whether a column is a label, an ordered rank, or a real number, because encoding and regression treat those differently.
This page separates variable types, then discrete features, then continuous ones, and how categorical predictors become indicators.
The same notes are in [Regression](../predictive-ml/regression.md).

Feature Types - no permission doc

## Variable types

This section names nominal, ordinal, and interval variable types before discrete and continuous are separated.

[Variable types:](http://www.socialresearchmethods.net/kb/measlevl.php) Nominal (weather), ordinal (order var 1,2,3), interval (range)

## Discrete

After the measurement levels, this section covers discrete features: numbers and categorical labels.

- Numbers
- Categorical
- Categorical data are variables that contain label values rather than numeric values.

The number of possible values is often limited to a fixed set.

- Nominal category - Wikipedia. Categorical variables are often called [nominal](https://en.wikipedia.org/wiki/Nominal_category)
- Labels, usually discrete values such as gender, country of origin, marital status, high-school graduate

## Continuous

With discrete named, this section is continuous features and how categorical predictors become numeric for regression.

Continuous (the opposite of discrete): real-number values, measured on a continuous scale: height, weight.

In order to compute a regression, categorical predictors must be re-expressed as numeric: some form of indicator variables (0/1) with a separate indicator for each level of the factor.

Discrete with many values are often treated as continuous, i.e. zone numbers -> binary

## Deprecated links

{% hint style="warning" %}
These links and images no longer work. The original wording is kept here. A same-resource copy, when one was checked, is used above.
{% endhint %}

- Feature Types - no permission doc. This address no longer opens: http://www.biostat.umn.edu/~will/6470stuff/Class09-12/Handout09.pdf
