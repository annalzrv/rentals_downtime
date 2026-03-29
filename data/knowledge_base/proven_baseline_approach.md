# Proven Baseline Approach For NYC Rental Availability Prediction

This document describes a feature set and model configuration that has been validated
on the competition dataset and achieves strong results. Use it as a starting point and
build on top of it.

## Feature families that are confirmed to help

### Date decomposition from last_dt
Parse last_dt as a datetime and extract four components: year, month, day of month, and
day of week. Leave NaN as NaN — CatBoost handles missing numerics natively. These
features capture seasonality and recency patterns in listing activity.

### Text length features from name and host_name
For the listing name and host name columns, compute character length and word count
after filling NaN with an empty string. Even simple text length signals correlate with
listing quality tier and host professionalism. Use str.len() and str.split().str.len().

### avg_reviews NaN handling
avg_reviews is NaN exactly when amt_reviews is zero — this is semantic, not random
missingness. Fill with 0 to encode "no reviews yet" correctly.

### Keeping identity columns as CatBoost categoricals
name, _id, host_name, location_cluster, location, and type_house should all be kept
as string categorical features and passed to CatBoost via cat_features. Do not drop
them. CatBoost learns listing-level and host-level patterns directly from these
identifiers, which is a strong signal for availability prediction. Fill NaN with
"unknown" before passing to CatBoost.

## Model configuration that works well
- CatBoostRegressor with loss_function RMSE
- depth 8, learning_rate 0.03, l2_leaf_reg 5
- 2500 iterations for the final submission model (trained on all data)
- 500 iterations for cross-validation folds (faster, still directionally correct)
- random_seed 42 for reproducibility
- TimeSeriesSplit with 5 folds on data sorted by last_dt

## Features to try on top of the baseline
The baseline above is a solid floor. To improve further, consider:
- Haversine distance from lat/lon to Manhattan centre (40.758, -73.985)
- log1p(sum) to reduce listed price skew
- log1p(total_host) — professional hosts behave differently
- has_reviews flag from amt_reviews
- days_since_last_review from last_dt with a large sentinel for NaN
- Interaction of location_cluster and type_house as a new string column
- log1p(amt_reviews) to capture review volume
- Neighbourhood rarity flag: locations with very few listings behave differently
