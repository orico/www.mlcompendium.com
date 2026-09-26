# Model Formats

A trained model has to leave the notebook. This page is how you save it so another application can read it, instead of pickling a Python object.

[Stop using pickle, use JSON or PMML](https://faoel.medium.com/stop-using-pickle-to-save-machine-learning-models-eaa46e8e561a).

## PMML

Once pickle is off the table, PMML is an XML description of a model that other applications can read.

[PMML](http://www.kdnuggets.com/faq/pmml.html): an XML file that describes a machine learning model that is transferable between applications.

- PMML 4.3 - General Structure. PMML 4.3 - General Structure. [PMML](http://dmg.org/pmml/v4-3/GeneralStructure.html)
- The structure of the models is described by an XML Schema.
- One or more mining models can be contained in a PMML document.
