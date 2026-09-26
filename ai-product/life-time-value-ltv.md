# Life Time Value (LTV)

A customer is worth more than one purchase. The page starts with probabilistic and Bayesian lifetime-value models, then the libraries that implement them, and ends with the classic Pareto/NBD model explained conceptually.

The same notes are in [Survival Analysis](survival-analysis.md).

The modern starting point is probabilistic LTV using churn and neural nets, and Google's code for it is [GitHub Google LTV](https://github.com/google/lifetime_value), the google/lifetime_value repository. The older statistical line has its own library: [lifetimes](https://github.com/CamDavidsonPilon/lifetimes) is lifetime value in Python. Between the two sits Bayesian Customer Lifetime Values Modeling using PyMC3.

Those libraries rest on the Pareto/NBD model, which the literature explains mostly in mathematics. ![Image 1: site logo](https://stats.stackexchange.com/Content/Sites/stats/Img/icon-48.png?v=6b892b6c384d)**Join Cross Validated**: [Understand Pareto and NBD models](https://stats.stackexchange.com/questions/251506/is-it-possible-to-understand-pareto-nbd-model-conceptually) is the Cross Validated question asking for a simple, conceptual explanation of Pareto/NBD, from someone using the BTYD package to predict when a customer is expected to be back.

The two modeling approaches from the start of the page each have a write-up. Probabilistic LTV using churn and neural nets is [https://towardsdatascience.com/the-paper-a-deep-probabilistic-model-for-customer-lifetime-value-prediction-eb5d61a83ecd](https://towardsdatascience.com/the-paper-a-deep-probabilistic-model-for-customer-lifetime-value-prediction-eb5d61a83ecd), Maja Pavlovic's run through the neural network architecture and loss function of "A Deep Probabilistic Model for Customer Lifetime Value Prediction". Bayesian Customer Lifetime Values Modeling using PyMC3 is [https://towardsdatascience.com/bayesian-customer-lifetime-values-modeling-using-pymc3-d770676f5c06](https://towardsdatascience.com/bayesian-customer-lifetime-values-modeling-using-pymc3-d770676f5c06), Meraldo Antonio's implementation of BG-NBD, a probabilistic hierarchical model, in PyMC3 to analyze customer purchase behavior.
