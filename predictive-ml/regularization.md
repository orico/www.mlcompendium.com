# Regularization

A model that fits noise looks great on training data and fails on anything new; regularization answers that with a penalty added to the loss so such a model is discouraged. The page starts with the definition and lambda, then the vector norms the penalties are built from, then L1 against L2 and why L1 leads to sparsity, then the priors behind both penalties, and ends with how regularization shows up in an SVM.

The same notes are in [DROPOUT LAYERS IN KERAS AND GENERAL](../deep-learning/deep-neural-nets.md#dropout-layers-in-keras-and-general), [Follow the regularized leader](../decision-intelligence/follow-the-regularized-leader.md), [Intuition for regularization in SVM](linear-separator-algorithms.md#intuition-for-regularization-in-svm), and [Regularization and influence](linear-separator-algorithms.md#regularization-and-influence).

## What regularization is

The definition starts from the loss a model already minimizes. [Regularization in linear regression](https://datanice.github.io/machine-learning-101-what-is-regularization-interactive.html) is the interactive explanation behind this section: to find the best model, we define a loss or cost function that describes how well the model fits the data, and we try to minimize it. A complex model that also fits the noise is overfitted, so we penalize it by adding a complexity term. That term adds a bigger loss for more complex models.

Lambda sets how strong that penalty is. A bigger lambda rules out high-complexity models (degree 3), because the penalty is stronger. A smaller lambda rules out models with high training error, such as a linear model on non-linear data? (degree 1). The optimum is in between (degree 2).

### Cost functions

Since the penalty is added to a cost function, it helps to know where that cost function comes from. ([I did not watch this](http://web.archive.org/web/20180309202629/https://www.coursera.org/learn/machine-learning/lecture/db3jS/model-representation), but here Andrew Ng talks about cost functions.)

## Vector norms

The complexity term is usually a norm of the weights, so the next step is to recall what a vector norm is, because L1 and L2 are norms. [A rehearsal on vector normalization](http://mathworld.wolfram.com/VectorNorm.html) answers it: for L1, L2, L3, L4, and so on, what is the norm? In some cases it is the absolute value. The figure below draws those norms side by side, copied from the original hosted image.

<figure><img src="../.gitbook/assets/gimg-a495d70a0f5b.png" alt=""><figcaption><p>Vector norms for L1, L2, L3, L4, and so on.</p><p>Credit: <a href="https://lh4.googleusercontent.com/5hTo0rvgBumQGTtucuYoXqXdL3Le2hDfKmqy6JfLwzWXFGn-SjWXcT34vc04uM6SJAuixyRkxPIUr3Fyv-3CrJ1SdqWjGll_hvy3p9rMjY-ZT0bV07Y2fvzBNgCG1-xbhlLdOxaJ">copied from the original hosted image</a>.</p></figcaption></figure>

### ISO surfaces, the Lp norm, and sparseness

The shapes in that figure are ISO surfaces, and their corners are where sparseness will come from later. Watch this. It also explains ISO surfaces, the Lp norm, and sparseness.

{% embed url="https://www.youtube.com/watch?v=sO4ZirJh9ds" %}

{% endembed %}

## L1 and L2

With the norms in hand, the two penalties people actually use are L1 and L2. The same notes are in [FEATURE SELECTION](../data/dependence-and-selection.md#feature-selection).

[L1, for sparse models](https://stats.stackexchange.com/questions/45643/why-l1-norm-for-sparse-models) is the question every reader of linear regression books eventually asks: the formulas are known, but why does the L1 norm enforce sparsity in models?

[L1 versus L2, with some formulas](https://towardsdatascience.com/l1-and-l2-regularization-methods-ce25e7fc831c) is an article on understanding L1 and L2 regularization.

### As a loss and as regularization

The same two norms can be the loss itself or the penalty on top of it, and they behave differently in each role. (What is the difference, and what are the features?) [L1 versus L2](http://web.archive.org/web/20260102124606/http://www.chioka.in/differences-between-l1-and-l2-as-loss-function-and-regularization/) as a loss function and as regularization.

{% columns %}
{% column %}
L1 moves the regressor faster. It selects features by driving coefficients to zero. With sparse algorithms it is computationally efficient. With others it is not, so use L2.
{% endcolumn %}

{% column %}
L2 moves more slowly. It does not make the coefficients sparse, and it is computationally efficient.
{% endcolumn %}
{% endcolumns %}

## Why L1 leads to sparsity

The columns claim that L1 drives coefficients to zero; this section shows why. Why does L1 lead to sparsity? There is an [Intuition](https://www.quora.com/Why-is-L1-regularization-supposed-to-lead-to-sparsity-than-L2) for it, and [some of the math](https://www.quora.com/What-is-the-difference-between-L1-and-L2-regularization) is in the Quora question on the difference between L1 and L2 regularization, how it solves overfitting, and which regularizer to use when. The figure shows where the hypothesis meets the L1 and L2 constraints, copied from the original hosted image.

<figure><img src="../.gitbook/assets/gimg-547e6ce30a8a.png" alt=""><figcaption><p>Where the hypothesis meets the L1 and L2 constraints.</p><p>Credit: <a href="https://lh6.googleusercontent.com/WOFPU50nTvEN0O6HdQZ8ZEyJQ3lAETvDEF_gyPWkauv7OG13X31ac51_iSTVHvejv34i4DVhQ67W2NgGh5i9Z90iZ3ojhtoLJVWVqo2nmPPb6Rla_eb21CoAI7uT-bjBvaWTYZ3J">copied from the original hosted image</a>.</p></figcaption></figure>

L1 and L2 regularization add constraints to the optimization problem. The curve H0 is the hypothesis. The solution is the set of points where H0 meets the constraints. In L2 the hypothesis is tangential to $$||w||_2$$. The point of intersection has both an x1 component and an x2 component. In L1, because of the shape of $$||w||_1$$, the viable solutions are limited to the corners of the axis, for example x1, so x2 = 0. The solution has eliminated the role of x2, which leads to sparsity.

This extends to higher dimensions, which is why L1 regularization leads to solutions in which many of the variables are 0. In other words, L1 regularization leads to sparsity. It is also treated as feature selection. With LibSVM, the recommendation is to select features before using the SVM, and to use L2 instead.

### A one-dimensional intuition

The picture above is geometric; a second intuition, in one dimension, shows why L1 can land exactly on zero. [L1 sparsity, intuition 2](https://www.quora.com/What-is-the-difference-between-L1-and-L2-regularization) comes from the same Quora question. For simplicity, consider the one-dimensional case.

With L2, the regularized loss $$F(x) = f(x) + \lambda ||x||^2$$ is smooth. The optimum is the stationary point, where the derivative is 0. That stationary point of F can become very small as λ increases, but it still will not be 0 unless $$f'(0) = 0$$.

With L1, the regularized loss $$F(x) = f(x) + \lambda ||x||$$ is not smooth. It has a kink, a minimum corner, at 0, and it is not differentiable there. Optimization theory says the optimum of a function is either a point where the derivative is 0 or one of the irregularities, such as a corner or a kink. So the optimal point of F can be 0 even when 0 is not the stationary point of f. If λ is large enough it is 0, which is a stronger regularization effect. Below is a graphical illustration, copied from the original hosted image.

<figure><img src="../.gitbook/assets/gimg-a48abb650e77.png" alt=""><figcaption><p>A graphical illustration of the one-dimensional L1 and L2 losses.</p><p>Credit: <a href="https://lh4.googleusercontent.com/stbOxAhMUFmtwSCdHHFFRdw-A3ngyZzVZHmEvezUHb5dkQrF4KQVs27I3euth9gUng3nkx4g7H2Gn2cx7_R0lzO-14sGhr9Yz8OiLYZ1gRoWIV8b5tl3pVI7z9uvRMI6IXhEpn9k">copied from the original hosted image</a>.</p></figcaption></figure>

#### In more than one dimension

The kink argument carries over feature by feature. In more than one dimension: if a feature is not important, the loss it contributes is small, and the non-differentiable regularization effect turns it off.

### Another intuition and formulation

For readers who want the result once more in a different form, the Stack Exchange question on why L1 norm gives sparse models has [An intuition and a formulation, which is pretty good:](https://stats.stackexchange.com/questions/45643/why-l1-norm-for-sparse-models)

<figure><img src="../.gitbook/assets/gimg-5f784d0a3b5c.png" alt=""><figcaption><p>An intuition and a formulation.</p><p>Credit: <a href="https://lh5.googleusercontent.com/BJ_dZzNlDQLh23d5OvjEJV-IYcBRjw57fZbWcuxO9bmpxpXIV1kKrZ3rIR4b_eKU4dx7tiFFCCd-VD2KYEcG9Yj5PqvpLzcUcj163WfrtaiC5b6JmgoOtZbJCE7j8VyOQpcOiSPc">copied from the original hosted image</a>.</p></figcaption></figure>

## Priors

Penalties can also be read as beliefs about the weights before seeing data: L2 as a Gaussian prior, and L1 as a Laplacean prior. The Stack Exchange question asks how one goes analytically from L2 regularization to a Gaussian prior: L2 regularization is [equivalent to a Gaussian prior](https://stats.stackexchange.com/questions/163388/l2-regularization-is-equivalent-to-gaussian-prior). The figure is that derivation, copied from the original hosted image.

<figure><img src="../.gitbook/assets/gimg-cb9d4cd1a22d.png" alt=""><figcaption><p>L2 regularization as a Gaussian prior.</p><p>Credit: <a href="https://lh6.googleusercontent.com/IKbhIIL-8B_VML7_gaPwgW70A9suIWqR2iELzjKTD_ABm9vruQUc5RSs83vYK8ujWb-q16gL2W4hzMT3f9FBCTsQQxH2_U-r24zXIva3FnllHjYc-VfA1qQEMyUu76ncSrI8ovri">copied from the original hosted image</a>.</p></figcaption></figure>

The same question covers the other penalty: L1 regularization is [equivalent to a Laplacean prior](https://stats.stackexchange.com/questions/163388/l2-regularization-is-equivalent-to-gaussian-prior) (same link as above).

> Similarly the relationship between the L1 norm and the Laplace prior can be understood in the same fashion. Take a Laplace prior instead of a Gaussian prior, combine it with your likelihood, and take the logarithm.

## Regularization in an SVM

The last stop is a model where the penalty hides under another name, the parameter C. [How does regularization look in an SVM?](https://datascience.stackexchange.com/questions/4943/intuition-for-the-regularization-parameter-in-svm) asks how varying it changes the decision boundary for a non-separable dataset. It comes down to controlling `C`.

## Deprecated links

{% hint style="warning" %}
These links and images no longer work. The original wording is kept here. A same-resource copy, when one was checked, is used above.
{% endhint %}

- (Difference between? And features of) L1 vs L2 as loss function and regularization. This address no longer opens: http://www.chioka.in/differences-between-l1-and-l2-as-loss-function-and-regularization/
- (did not watch) but here is andrew ng talks about cost functions. This address no longer opens: https://www.coursera.org/learn/machine-learning/lecture/db3jS/model-representation
