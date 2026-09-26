# Data

The business problem is now a question about records, and the next work is to get those records into a form a model can use.
After this part the reader can walk datasets, processing, analytics, and data engineering in that order.

The walk starts with the rows themselves, because a model is only as defined as the rows it is trained on: [Datasets](datasets-intro.md) names feature types, finds a dataset, and judges whether the split and the label balance are fit to use. Those raw values are rarely on the scale or in the shape a learner expects, so [Data Processing](processing-intro.md) scales, transforms, imputes, and annotates features and points at a pipeline that does it. Once the table is in shape, probability and dependence decide what a later score means, and [Analytics](analytics-intro.md) reads distributions, dependence, and information measures so a feature is chosen for a reason. Finally the dataset has to live in a system other people can run, and [Data Engineering](engineering-intro.md) places SQL, storage, quality, lineage, and ownership on one path from source to table.
