# Data Patterns

A row that changes over time needs a rule for what the history looks like. The pattern for that is slowly changing dimensions: what to do when a row changes, from keeping nothing to keeping every version.

The rule belongs to the warehouse design, not to one pipeline. The same notes are in [Data Warehouse](database-architecture-and-modeling.md#data-warehouse).

The reference is Adatis. [Slowly changing dimensions (SCD)](https://adatis.co.uk/introduction-to-slowly-changing-dimensions-scd-types/) by Adatis: what you can do when the information in your table changes, types 0 through 6. The figure below lays those types out side by side, with the credit to Adatis.

<figure><img src="../../ops/.gitbook/assets/image (12).png" alt=""><figcaption><p>Slowly changing dimensions</p><p>Credit: <a href="https://adatis.co.uk/introduction-to-slowly-changing-dimensions-scd-types/">Adatis</a></p></figcaption></figure>
