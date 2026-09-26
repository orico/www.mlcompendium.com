# NYC TAXI

This page is an intro to the well-known NYC taxi pickup problem for regression and related evaluation notes.

### [NYC taxi pickup problem](http://www.vivekchoksi.com/papers/taxi_pickups.pdf)

This section points at the pickup paper and a second study, plus error measures and a regression tip.

[The taxi problem](http://www.vivekchoksi.com/papers/taxi_pickups.pdf) is an intro to a well-known machine learning problem. The paper explains feature engineering, analysis, and using various regression algorithms to solve the problem. You can use this as a base for many regression and classification problems.

A [second study](http://blog.nycdatascience.com/student-works/predict-new-york-city-taxi-demand/) (regression, random forest, [xgboost](https://xgboost.readthedocs.io/en/latest/tutorials/model.html) (extreme gradient boosting tree)).

[Standard error estimate](https://www.youtube.com/watch?v=r-txC-dpI-E&index=4&list=PLF596A4043DBEAE9C) — measures the distance from the estimated value to the real value.

R^2 error estimate — measures the distance of the estimated to the mean against the real to the mean; 1 means no error, 0 means lots.

With regression prediction it is best to create dummy variables (i.e., binary variables — exist or doesn't exist) from numeric variables, such as grid_number to grid_1, grid_2, etc.

## Deprecated links

{% hint style="warning" %}
These links and images no longer work. The original wording is kept here. A same-resource copy, when one was checked, is used above.
{% endhint %}

- xgboost (extreme gradient boosting tree). This address no longer opens: http://xgboost.readthedocs.io/en/latest/model.html
