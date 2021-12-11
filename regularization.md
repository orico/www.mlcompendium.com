# Regularization

****[**Watch this.**](https://www.youtube.com/watch?v=sO4ZirJh9ds) **Also explains about ISO surfaces, lp norm, sparseness.**

**(what is?)** [**Regularization (in linear regression**](https://datanice.github.io/machine-learning-101-what-is-regularization-interactive.html)**) - to find the best model we define a loss or cost function that describes how well the model fits the data, and try minimize it. For a complex model that fits even the noise, i.e., over fitted, we penalize it by adding a complexity term that would add BIGGER LOSS for more complex models.**

* **Bigger lambda -> high complexity models (deg 3) are ruled out, more punishment.**
* **Smaller lambda -> models with high training error are rules out. I.e.,  linear model on non linear data?, i.e., deg 1.**
* **Optimal is in between (deg 2)**

[**L1 - for sparse models,** ](https://stats.stackexchange.com/questions/45643/why-l1-norm-for-sparse-models)

[**L1 vs L2, some formula**](https://towardsdatascience.com/l1-and-l2-regularization-methods-ce25e7fc831c)

[**Rehearsal on vector normalization**](http://mathworld.wolfram.com/VectorNorm.html) **- for l1,l2,l3,l4 etc, what is the norm? (absolute value in certain cases)**![](https://lh4.googleusercontent.com/5hTo0rvgBumQGTtucuYoXqXdL3Le2hDfKmqy6JfLwzWXFGn-SjWXcT34vc04uM6SJAuixyRkxPIUr3Fyv-3CrJ1SdqWjGll\_hvy3p9rMjY-ZT0bV07Y2fvzBNgCG1-xbhlLdOxaJ)

**(Difference between? And features of)** [**L1 vs L2**](http://www.chioka.in/differences-between-l1-and-l2-as-loss-function-and-regularization/) **as loss function and regularization.**

* **L1 - moves the regressor faster, feature selection by sparsing coefficients (zeroing them), with sparse algorithms it is computationally efficient, with others no, so use L2.**
* **L2 - moves slower, doesn't sparse, computationally efficient.**

**Why does L1 lead to sparity?**

* [**Intuition**](https://www.quora.com/Why-is-L1-regularization-supposed-to-lead-to-sparsity-than-L2) **+** [**some mathematical info**](https://www.quora.com/What-is-the-difference-between-L1-and-L2-regularization)![](https://lh6.googleusercontent.com/WOFPU50nTvEN0O6HdQZ8ZEyJQ3lAETvDEF\_gyPWkauv7OG13X31ac51\_iSTVHvejv34i4DVhQ67W2NgGh5i9Z90iZ3ojhtoLJVWVqo2nmPPb6Rla\_eb21CoAI7uT-bjBvaWTYZ3J)
* **L1 & L2 regularization add constraints to the optimization problem. The curve H0 is the hypothesis. The solution is a set of points where the H0 meets the constraints.**&#x20;
* **In L2 the the hypothesis is tangential to the ||w||\_2. The point of intersection has both x1 and x2 components. On the other hand, in L1, due to the nature of ||w||\_1, the viable solutions are limited to the corners of the axis, i.e.,  x1. So that the value of x2 = 0. This means that the solution has eliminated the role of x2 leading to sparsity.**&#x20;
* **This can be extended to a higher dimensions and you can see why L1 regularization leads to solutions to the optimization problem where many of the variables have value 0.** &#x20;
* **In other words, L1 regularization leads to sparsity.**
* **Also considered feature selection - although with LibSVM the recommendation is to feature select prior to using the SVM and use L2 instead.**

[**L1 sparsity - intuition #2**](https://www.quora.com/What-is-the-difference-between-L1-and-L2-regularization)

* **For simplicity, let's just consider the 1-dimensional case.**
* **L2:**
* **L2-regularized loss function F(x)=f(x)+λ∥x∥^2 is smooth.**&#x20;
* **This means that the optimum is the stationary point (0-derivative point).**&#x20;
* **The stationary point of F can get very small when you increase λ, but it will still won't be 0 unless f′(0)=0.**
* **L1:**
  * **regularized loss function F(x)=f(x)+λ∥x∥ is non-smooth, i.e., a min knee of 0.**
  * **It's not differentiable at 0.**&#x20;
  * **Optimization theory says that the optimum of a function is either the point with 0-derivative or one of the irregularities (corners, kinks, etc.). So, it's possible that the optimal point of F is 0 even if 0 isn't the stationary point of f.**
  * **In fact, it would be 0 if λ is large enough (stronger regularization effect). Below is a graphical illustration.**

![](https://lh4.googleusercontent.com/stbOxAhMUFmtwSCdHHFFRdw-A3ngyZzVZHmEvezUHb5dkQrF4KQVs27I3euth9gUng3nkx4g7H2Gn2cx7\_R0lzO-14sGhr9Yz8OiLYZ1gRoWIV8b5tl3pVI7z9uvRMI6IXhEpn9k)

**In multi-dimensional settings: if a feature is not important, the loss contributed by it is small and hence the (non-differentiable) regularization effect would turn it off.**

[**Intuition + formulation, which is pretty good:**](https://stats.stackexchange.com/questions/45643/why-l1-norm-for-sparse-models)

![](https://lh5.googleusercontent.com/BJ\_dZzNlDQLh23d5OvjEJV-IYcBRjw57fZbWcuxO9bmpxpXIV1kKrZ3rIR4b\_eKU4dx7tiFFCCd-VD2KYEcG9Yj5PqvpLzcUcj163WfrtaiC5b6JmgoOtZbJCE7j8VyOQpcOiSPc)

**(**[**did not watch**](https://www.coursera.org/learn/machine-learning/lecture/db3jS/model-representation)**) but here is andrew ng talks about cost functions.**

**L2 regularization** [**equivalent to Gaussian prior**](https://stats.stackexchange.com/questions/163388/l2-regularization-is-equivalent-to-gaussian-prior)

![](https://lh6.googleusercontent.com/IKbhIIL-8B\_VML7\_gaPwgW70A9suIWqR2iELzjKTD\_ABm9vruQUc5RSs83vYK8ujWb-q16gL2W4hzMT3f9FBCTsQQxH2\_U-r24zXIva3FnllHjYc-VfA1qQEMyUu76ncSrI8ovri)

**L1 regularization** [**equivalent to a Laplacean Prior**](https://stats.stackexchange.com/questions/163388/l2-regularization-is-equivalent-to-gaussian-prior)**(same link as above) - “Similarly the relationship between L1 norm and the Laplace prior can be undestood in the same fashion. Take instead of a Gaussian prior, a Laplace prior combine it with your likelihood and take the logarithm.“**\
****[**How does regularization look like in SVM**](https://datascience.stackexchange.com/questions/4943/intuition-for-the-regularization-parameter-in-svm) **- controlling ‘C’**
