# Normalization & Scaling

This page compares normalization and scaling choices, including when to standardize versus normalize.

1. [A comparison of normalization / scaling techniques in sklearn](http://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#sphx-glr-auto-examples-preprocessing-plot-all-scaling-py)
2. [Another great explanation on sklearn and (general) scaling](http://benalexkeen.com/feature-scaling-with-scikit-learn/) - normal, min max, etc.
3. [Normalization/standardize features](http://machinelearningmastery.com/normalize-standardize-machine-learning-data-weka/)

- Data has varying scales.
- Normalize between range 0 to 1.
   - When the algorithm you are using does not make assumptions about the distribution of your data, such as k-nearest neighbors and artificial neural networks.
- Standardize, mean of 0 and a std of 1:
   - When the algorithm assumes a Gaussian dist, such as linear regression, logistic regression and linear discriminant analysis. LR, LogR, LDA

Generally, it is a good idea to standardize data that has a Gaussian (bell curve) distribution and normalize otherwise. In general terms, we should test 0,1 or -1,1 empirically and possibly match the range to the NN gates/activation function etc.
