# Model Formats

A trained model has to leave the notebook. The page first says why pickling a Python object is the wrong way to save it, then shows PMML as a format another application can read.

The default habit is pickle, and the case against it is [Stop using pickle, use JSON or PMML](https://faoel.medium.com/stop-using-pickle-to-save-machine-learning-models-eaa46e8e561a), Elfao's argument to stop using pickle to save machine learning models.

## PMML

Once pickle is off the table, PMML is an XML description of a model that other applications can read.

[PMML](http://www.kdnuggets.com/faq/pmml.html): an XML file that describes a machine learning model that is transferable between applications. The specification's [PMML](http://dmg.org/pmml/v4-3/GeneralStructure.html) page, PMML 4.3 - General Structure, sets out two rules. The structure of the models is described by an XML Schema, and one or more mining models can be contained in a PMML document.
