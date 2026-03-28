# Time Series Feature Engineering For Daily Rental Demand

If the dataset contains a timestamp like `last_dt`, random train/validation splits are misleading. Sort by time and validate with `TimeSeriesSplit` so that later observations are predicted from earlier ones. This matters even if `last_dt` is sparse, because the model should not learn future review activity patterns during validation.

The safest temporal features in this dataset are recency and calendar buckets derived from `last_dt`. Useful examples:
- `days_since_last_review`
- `weeks_since_last_review`
- review month
- review quarter
- review weekday
- weekend indicator
- missing-date flag for listings with no reviews yet

Do not treat missing `last_dt` as generic noise. In rental data, a missing review date often means the listing is new, inactive, or review-free. Encode this state explicitly with:
- `has_last_dt`
- `no_reviews_yet`
- a large sentinel recency value

Temporal features become stronger when paired with listing structure. Good interactions include:
- `location_cluster x review_month`
- `type_house x review_weekday`
- `log_sum x recency`
- `min_days x recency_bucket`

Because this dataset does not provide a full time series per listing, classical target lags are usually impossible or unsafe. Focus instead on proxy temporal signals:
- freshness of listing activity
- seasonality bucket from the last known review
- whether the listing looks recently active versus stale

For evaluation, keep the ordering rule simple and reproducible. Parse `last_dt` with `errors="coerce"`, sort on it after filling missing values with a sentinel date, and use the same ordering rule across folds. In short-term rental problems, a careful split strategy is often as important as the feature formulas themselves.
