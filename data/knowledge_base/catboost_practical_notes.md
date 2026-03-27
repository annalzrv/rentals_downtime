# CatBoost Practical Notes

CatBoost is a strong first choice for heterogeneous rental tables because it handles mixed numeric and categorical features with limited preprocessing. It is especially convenient when the dataset contains text-like neighborhood labels, room types, and missing values.

Use a clean validation loop and keep categorical handling consistent across folds. If feature engineering creates dense numeric features from coordinates, dates, and counts, CatBoost can combine them with raw categorical columns such as `location_cluster`, `location`, and `type_house`.

Watch out for unstable targets caused by extreme price outliers or data-entry mistakes. Log transforms on skewed inputs, clipping extreme numeric features, and explicit missing-value indicators can improve generalization more reliably than aggressive manual cleaning.

For a competition baseline, prioritize a reproducible cross-validation setup, sensible feature construction, and compact hyperparameter search before attempting complex ensembles.
