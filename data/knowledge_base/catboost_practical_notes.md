# CatBoost Practical Notes

CatBoost is a strong default model for this dataset because the schema mixes numeric columns, medium-cardinality categorical columns, sparse review signals, and missing values. It is especially useful when we want a baseline that can ingest raw categorical features such as `location_cluster`, `location`, and `type_house` without fragile preprocessing.

When using CatBoost, do not rush to one-hot encode everything. Keep natural categorical columns as categorical whenever possible:
- `location_cluster`
- `location`
- `type_house`
- possibly cleaned `host_name`

CatBoost already handles missing numeric values and can exploit category combinations internally, so a good feature plan should focus on meaningful derived variables instead of redundant preprocessing. High-value engineered inputs for CatBoost in this dataset are:
- `log1p(sum)`
- clipped `min_days`
- recency features from `last_dt`
- missing-value indicators
- frequency or rarity flags for neighborhoods

Text-like columns can be useful if handled carefully. The listing `name` may encode price tier, room style, or tourist appeal. Lightweight text features are often safer than full NLP:
- word count
- title length
- keyword flags such as `luxury`, `private`, `spacious`, `central`
- missing-title flag

For high-cardinality location fields, the model often benefits from combining raw categorical information with spatial numeric proxies. In practice this means:
- keep `location` as categorical
- also provide `lat`, `lon`
- add one or more distance-based numeric features
- add geo-cell or cluster identifiers

CatBoost will not save a weak validation protocol. If cross-validation is wrong, the model will still overestimate quality. Pair CatBoost with time-aware validation, stable feature generation, and a compact hyperparameter search before trying ensembles.
