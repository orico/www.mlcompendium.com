# Distribution Transformation

[**Top 3 methods for handling skewed data**](https://towardsdatascience.com/top-3-methods-for-handling-skewed-data-1334e0debf45)**. Log, square root, box cox transformations**

### **BOX COX**

[**Power transformations**](https://machinelearningmastery.com/power-transforms-with-scikit-learn/?fbclid=IwAR37SGKEXWQ\_39qZLKAQ5WunSECo0JXsd3qgz3dPGITTGcVwHJla-\_7GLKg)****\
****

**(What is the Box-Cox Power Transformation?) **

* **a procedure to identify an appropriate exponent (Lambda = l) to use to transform data into a “normal shape.”**
* **The Lambda value indicates the power to which all data should be raised.**

![](https://lh5.googleusercontent.com/3OZx1GhRUjnDqpD91pEYoXMCSq9aYtf\_6IIBgMJRj680OYddZlNachWfRfTVyB1TJlhzwQ_m6iAINfTU2VSn4QoXwPbZPBNoQm7SQ4ijWw2001kCNKAVvKhpLpotNL_btUEo8cui)

[**The Box-Cox transformation is a useful family of transformations. **](http://www.itl.nist.gov/div898/handbook/eda/section3/eda336.htm)****\
****

* **Many statistical tests and intervals are based on the assumption of normality. **
* **The assumption of normality often leads to tests that are simple, mathematically tractable, and powerful compared to tests that do not make the normality assumption. **
* **Unfortunately, many real data sets are in fact not approximately normal. **
* **However, an appropriate transformation of a data set can often yield a data set that does follow approximately a normal distribution.**
* ** This increases the applicability and usefulness of statistical techniques based on the normality assumption.**![](https://lh6.googleusercontent.com/zPpR_hjhoZZkL5BjkI1n20Lu2AQW4PaY9sGgUDXr9dptmTHx4wK1n_WpeTc5ACkr7LaQ\_38xHyl9KGO012SdHGpSg1lDmVd4GGgi7R195KEnxJHIMklq-tDcGRsRjj2T4Gs2ezSk)

**IMPORTANT:!! After a transformation (c), we need to measure the normality of the resulting transformation (d) . **

* **One measure is to compute the correlation coefficient of a **[**normal probability plot**](http://www.itl.nist.gov/div898/handbook/eda/section3/normprpl.htm)** => (d). **
* **The correlation is computed between the vertical and horizontal axis variables of the probability plot and is a convenient measure of the linearity of the probability plot **
* **In other words: the more linear the probability plot, the better a normal distribution fits the data!**

[**\*NOTE: another useful link that explains it with figures, but i did not read it.**](http://blog.minitab.com/blog/applying-statistics-in-quality-projects/how-could-you-benefit-from-a-box-cox-transformation)

**GUARANTEED NORMALITY?**

* **NO!**
* **This is because it actually does not really check for normality;**
* **the method checks for the smallest standard deviation.**
* **The assumption is that among all transformations with Lambda values between -5 and +5, transformed data has the highest likelihood – but not a guarantee – to be normally distributed when standard deviation is the smallest. **
* **it is absolutely necessary to always check the transformed data for normality using a probability plot. (d)**

**+ Additionally, the Box-Cox Power transformation only works if all the data is positive and greater than 0.**

**+ achieved easily by adding a constant ‘c’ to all data such that it all becomes positive before it is transformed. The transformation equation is then:**\
****

[**COMMON TRANSFORMATION FORMULAS (based on the actual formula)**](http://www.statisticshowto.com/box-cox-transformation/)

![](https://lh4.googleusercontent.com/Vw2mhxsDDXw5qnI-WbQ7cCdeLW7TKQ_A4KL95c6UhkvyCsOC4vO7AfqsvN1Uw32Mz1cR8bAtxUld4ui-v1mq74ICcPfQiSe1w1o5JTvhgox3urLj9t9ATAz_d1RGQv94\_cO_Ye3b)

**Finally: An awesome **[**tutorial (dead),**](http://www.kmdatascience.com/2017/07/box-cox-transformations-in-python.html)** **[**here is a new one**](https://towardsdatascience.com/box-cox-transformation-explained-51d745e34203#:\~:text=scipy.stats.boxcox\(\),the%2095%25%20confidence%20interval\).)** in python with **[**code examples**](https://github.com/kentmacdonald2/Box-Cox-Transformation-Python-Example)**, there is also another code example **[**here**\
****](https://stackoverflow.com/questions/33944129/python-library-for-data-scaling-centering-and-box-cox-transformation)**“Simply pass a 1-D array into the function and it will return the Box-Cox transformed array and the optimal value for lambda. You can also specify a number, alpha, which calculates the confidence interval for that value. (For example, alpha = 0.05 gives the 95% confidence interval).” **\
****

![](https://lh6.googleusercontent.com/kbGUwNoKCtOEvSu02zfiJMmEScrFGSW5iuwzvNOm6V4t3OigHiTHtJLqKVzchyVe2MPH3LpsvywhFW3v3-j16dgRHb_o73rBPk264Z9HSXsCRTZodB\_41YQukSjMVtZ6IQecd2Rk)

**\* Maybe there is a slight problem in the python vs R code, **[**details here**](http://shahramabyari.com/2015/12/21/data-preparation-for-predictive-modeling-resolving-skewness/)**, but needs investigating.**

### **MANN-WHITNEY U TEST**

**(**[**what is?**](https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test)**) - the Mann–Whitney U test  is a **[**nonparametric**](https://en.wikipedia.org/wiki/Nonparametric_statistics)** **[**test**](https://en.wikipedia.org/wiki/Statistical_hypothesis_test)** of the **[**null hypothesis**](https://en.wikipedia.org/wiki/Null_hypothesis)** that it is equally likely that a randomly selected value from one sample will be less than or greater than a randomly selected value from a second sample.**\
****

**In other words: This test can be used to determine whether two independent samples were selected from populations having the same distribution. **

**Unlike the **[**t-test**](https://en.wikipedia.org/wiki/T-test)** it does not require the assumption of **[**normal distributions**](https://en.wikipedia.org/wiki/Normal_distribution)**. It is nearly as efficient as the t-test on normal distributions.**

### **NULL HYPOTHESIS**

1. [**What is chi-square and what is a null hypothesis, and how do we calculate observed vs expected and check if we can reject the null and get significant difference.**](https://medium.com/greyatom/goodness-of-fit-using-chi-square-be5bba375caf)
2. **Analytics vidhya**
   1. [**What is hypothesis testing **](https://www.analyticsvidhya.com/blog/2015/09/hypothesis-testing-explained/)
   2. [**Intro to t-tests analytics vidhya**](https://www.analyticsvidhya.com/blog/2019/05/statistics-t-test-introduction-r-implementation/)** - always good**
   3. [**Anova analysis of variance**](https://www.analyticsvidhya.com/blog/2018/01/anova-analysis-of-variance/?fbclid=IwAR1lMhaoKevShaIDpNoRNPL-V7y_LMscZSPG\_0Dp1qvCkhDoJgzyt4fMDKM)**, one way, two way, manova**
      1. ** if the means of two or more groups are significantly different from each other. ANOVA checks the impact of one or more factors by comparing the means of different samples.**
      2. **A one-way ANOVA tells us that at least two groups are different from each other. But it won’t tell us which groups are different.**
      3. **For such cases, when the outcome or dependent variable (in our case the test scores) is affected by two independent variables/factors we use a slightly modified technique called two-way ANOVA.**
3. **multivariate case and the technique we will use to solve it is known as MANOVA.**
