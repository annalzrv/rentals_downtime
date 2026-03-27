# Kaggle Baseline For Rental Listings

For tabular rental competitions, the strongest early baseline usually combines robust numeric transforms with a tree model such as CatBoost. Price-like columns are heavily skewed, so `log1p(sum)` is often more stable than the raw value. Host portfolio size (`total_host`) is also long-tailed and benefits from `log1p`.

Location is usually the dominant signal. Start with borough-level features from `location_cluster`, then add neighborhood or coordinate-derived features. For New York style listings, distance to Midtown or lower Manhattan often behaves like a smooth premium feature even when neighborhood labels are noisy.

Review sparsity matters. If `avg_reviews` is missing exactly when `amt_reviews == 0`, treat the missingness as information rather than noise. A binary `has_reviews` feature plus a filled score column is often better than mean imputation.

When the task resembles demand or availability over time, combine structural listing features with lightweight temporal features extracted from the last known activity date. Even if `last_dt` is incomplete, `days_since_last_review` and calendar buckets can help the model separate active listings from stale ones.
