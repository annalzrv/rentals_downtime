# GitHub NYC Airbnb Baseline

An accessible public baseline for Airbnb-style price prediction is the repository "Predicting Airbnb Prices in New York City". It uses listing metadata such as location, room type, review counts, host behavior, and cross-validation to compare multiple regressors. Even when the exact target differs from our Kaggle task, the repository is useful as a feature-engineering reference because it highlights location, room type, reviews, and host-level patterns as the strongest drivers.

The practical lesson for our pipeline is to keep the RAG output concrete and tabular. Retrieved ideas should map cleanly to columns already present in the dataframe, and the coder should receive a compact specification such as "log-transform price", "encode borough and room type", "derive recency from last review date", and "model host portfolio size".

Another useful point from this baseline is that gradient-boosting families tend to outperform simpler linear baselines once categorical and skewed numeric signals are cleaned up. That aligns with our choice to steer the coder toward CatBoost or LightGBM.
