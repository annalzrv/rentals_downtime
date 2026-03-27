# Outliers, Validation, And Submission Hygiene

Rental data often contains extreme nightly prices, suspicious coordinates, and stale listings with no reviews. Tree models can tolerate some noise, but upstream validation still matters. Check ranges for `sum`, `lat`, `lon`, `min_days`, and target predictions before generating the final submission.

Outlier handling does not always require dropping rows. Winsorizing or log-transforming skewed monetary features can preserve signal while reducing fold-to-fold variance. Missing values should be paired with indicator features whenever the missingness itself reflects listing behavior.

Submission files should contain only numeric predictions, no NaN values, and exactly the expected row count. If a validation script catches bad predictions early, route the pipeline back to the coder rather than letting broken outputs reach the final stage.
