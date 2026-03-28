# Kaggle Baseline For Rental Listings

For Airbnb-style tabular competitions, a strong first baseline is usually a gradient-boosting model with careful feature cleanup rather than a very complex architecture. The columns already available in this dataset are rich enough to produce a useful baseline if they are transformed consistently and validated without leakage.

Start with price-aware but leakage-safe features derived from the observed listing price `sum`. Good baseline transformations include `log1p(sum)`, clipped price, price buckets, and interactions such as `log_sum x type_house` or `log_sum x location_cluster`. On rental data, raw price is often heavy-tailed, so models usually behave more stably after log scaling and basic clipping.

Location should be treated as the strongest structural signal. Do not stop at a simple borough encoding. Useful first-wave features are:
- categorical `location_cluster`
- categorical `location`
- rare-location flag for low-frequency neighborhoods
- lat/lon interactions
- distance to Midtown or another Manhattan anchor
- coarse geo cells or coordinate bins

Review sparsity is informative by itself. If `avg_reviews` is missing when `amt_reviews == 0`, treat that as a semantic state rather than ordinary missingness. A good baseline often includes:
- `has_reviews`
- filled `avg_reviews`
- `log1p(amt_reviews)`
- `days_since_last_review`
- recency buckets such as `recent / stale / never reviewed`

Host-level structure is also useful even without an explicit host id column. The column `total_host` already tells us whether a listing belongs to a likely professional host. Suggested baseline features:
- `log1p(total_host)`
- bins such as `1`, `2-3`, `4+`
- interaction `total_host x type_house`
- flag `is_professional_host`

Text columns can help if handled lightly. Even simple features from `name` are often worth trying:
- title length
- word count
- presence of tokens such as `luxury`, `cozy`, `private`, `modern`, `central`
- missing-title indicator

When the target behaves like demand or availability, combine these structural features with date-derived recency features from `last_dt`. The goal of a baseline is not to invent exotic features first, but to cover location, price, reviews, host behavior, and time recency in a leakage-safe way.
