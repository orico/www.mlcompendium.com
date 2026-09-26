# NYC TAXI

Regression needs a concrete city problem to be more than a formula. The page works through the NYC taxi pickup problem, then the error measures used to judge it, and ends with a practical regression tip and two related worked cases.

The same notes are in [Regression](../predictive-ml/regression.md).

### [NYC taxi pickup problem](http://www.vivekchoksi.com/papers/taxi_pickups.pdf)

The problem is predicting how many taxi pickups happen where and when, and the paper is the base for the rest of the page.

[The taxi problem](http://www.vivekchoksi.com/papers/taxi_pickups.pdf) is an intro to a well-known machine learning problem: "Predicting Taxi Pickups in New York City", the final paper for CS221 in Autumn 2014 by Josh Grinberg, Arzav Jain, and Vivek Choksi. The paper explains feature engineering, analysis, and using various regression algorithms to solve the problem. You can use this as a base for many regression and classification problems.

The same problem was attacked again with different models. A [second study](http://blog.nycdatascience.com/student-works/predict-new-york-city-taxi-demand/) from the NYC Data Science Academy student works predicts New York City taxi demand (regression, random forest, [xgboost](https://xgboost.readthedocs.io/en/latest/tutorials/model.html) (extreme gradient boosting tree)); the xgboost link is its introduction to boosted trees.

Once models exist, they need an error measure. [Standard error estimate](https://www.youtube.com/watch?v=r-txC-dpI-E&index=4&list=PLF596A4043DBEAE9C) — measures the distance from the estimated value to the real value.

R^2 error estimate — measures the distance of the estimated to the mean against the real to the mean; 1 means no error, 0 means lots.

The last lesson is about features, not errors. With regression prediction it is best to create dummy variables (i.e., binary variables — exist or doesn't exist) from numeric variables, such as grid_number to grid_1, grid_2, etc.

See also [Breast Augmentation Using Gen-AI](https://cohenori.medium.com/breast-augmentation-using-gen-ai-15492ab71f8b) (October 2024), a guide to using AI for virtual breast augmentation in photographs as an example of realistically altering human features in digital image editing.
[Optimizing University Course Scheduling](https://cohenori.medium.com/optimizing-university-course-scheduling-a-constraint-programming-approach-a4f1533037a3) (August 2025) is a worked scheduling case: a constraint programming approach to scheduling courses efficiently while maximizing revenue and satisfying numerous operational constraints.

## Deprecated links

{% hint style="warning" %}
These links and images no longer work. The original wording is kept here. A same-resource copy, when one was checked, is used above.
{% endhint %}

- xgboost (extreme gradient boosting tree). This address no longer opens: http://xgboost.readthedocs.io/en/latest/model.html
