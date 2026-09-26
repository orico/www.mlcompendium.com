# Datasets

A model is only as defined as the rows it is trained on.
After this chapter the reader can name feature types, find a dataset, and judge whether the split and the label balance are fit to use.

The first question about a row is what each column is, and [Feature Types](feature-types.md) separates a label from an ordered rank from a real number, because encoding and regression treat those differently. Knowing the types is useless if nobody can find the data, so [Data Catalogs](datasets/data-catalogs.md) is the shelf that names what exists and the tools that implement one. A found dataset can still look good in training while its labels are noisy or its easy examples dominate, and [Dataset Confidence](dataset-confidence.md) uses dataset cartography to decide how much to trust it. With trust settled, [Datasets](datasets.md) is the work of typing the data, splitting it, and deciding when the sample is fair, from structured versus unstructured data through imbalance and transfer learning.
