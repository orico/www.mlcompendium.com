# Feature Types

A model needs to know whether a column is a label, an ordered rank, or a real number, because encoding and regression treat those differently.
This page separates variable types, then discrete features, then continuous ones, and how categorical predictors become indicators.
The same notes are in [Regression](../predictive-ml/regression.md).

The original Feature Types handout, a no permission doc, used to sit here; it no longer opens and is kept at the end of the page.

## Variable types

Before a column can be called discrete or continuous, it needs a measurement level. The [Variable types:](http://www.socialresearchmethods.net/kb/measlevl.php) note on levels of measurement is the reference for the three used here: nominal (weather), ordinal (order var 1,2,3), and interval (range).

## Discrete

The measurement levels split first into discrete features, which come as numbers or as categories. Categorical data are variables that contain label values rather than numeric values, and the number of possible values is often limited to a fixed set. Categorical variables are often called [nominal](https://en.wikipedia.org/wiki/Nominal_category), which is the Wikipedia entry on the nominal category. Those labels are usually discrete values such as gender, country of origin, marital status, or high-school graduate.

## Continuous

Continuous is the opposite of discrete: real-number values, measured on a continuous scale, such as height and weight. The two meet at the regression step. In order to compute a regression, categorical predictors must be re-expressed as numeric: some form of indicator variables (0/1) with a separate indicator for each level of the factor. The boundary also runs the other way, since discrete features with many values are often treated as continuous, i.e. zone numbers -> binary.

## Deprecated links

{% hint style="warning" %}
These links and images no longer work. The original wording is kept here. A same-resource copy, when one was checked, is used above.
{% endhint %}

- Feature Types - no permission doc. This address no longer opens: http://www.biostat.umn.edu/~will/6470stuff/Class09-12/Handout09.pdf
