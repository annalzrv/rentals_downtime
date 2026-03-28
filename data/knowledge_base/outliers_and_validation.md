# Outliers, Validation, And Submission Hygiene

Rental datasets are usually noisy. Extreme nightly prices, suspicious coordinates, stale listings, and malformed review fields can all hurt model stability. The best approach is usually not to drop everything suspicious, but to encode and cap problematic signals in a controlled way.

For numeric cleanup, start with simple robust operations:
- `log1p(sum)` for price skew
- clipping or winsorizing extreme `sum`
- clipping obviously unreasonable `min_days`
- checking coordinate ranges for `lat` and `lon`
- flagging impossible or suspicious values instead of silently discarding rows

Missing values should often become features. Examples:
- `avg_reviews_is_missing`
- `last_dt_is_missing`
- `name_is_missing`
- `host_name_is_missing`

Validation hygiene matters as much as feature engineering. Common failure modes to avoid:
- random split when time information exists
- target-derived encodings fitted on the full dataset
- fitting preprocessing differently on train and validation
- allowing generated code to assume that `test.csv` contains `target`

Useful validation-oriented features can still be built safely:
- frequency counts fitted on train only
- neighborhood rarity flags
- coarse geo bins
- price buckets
- review-recency buckets

Submission checks should be strict and boring. A valid pipeline should guarantee:
- numeric predictions only
- no NaN in `prediction`
- exact row count match with `test.csv`
- correct columns `index,prediction`

If a run fails one of these checks, route it back to the coder rather than pretending it is a modeling problem. Broken artifacts waste iterations and make the RAG agent learn from noise.
