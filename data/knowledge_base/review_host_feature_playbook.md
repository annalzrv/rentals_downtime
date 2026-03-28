# Review And Host Feature Playbook

Reviews and host structure often encode listing maturity, trust, and market segment. The current baseline features cover only part of that story. A stronger feature plan should treat review sparsity and host professionalism as separate axes.

Review-derived features worth trying:
- `has_reviews`
- `log1p(amt_reviews)`
- `avg_reviews_filled`
- `amt_reviews_is_zero`
- `days_since_last_review`
- recency bucket such as `recent / medium / stale / never reviewed`
- interaction `amt_reviews x type_house`
- interaction `avg_reviews_filled x log_sum`

Missing review fields are usually meaningful:
- `avg_reviews` missing often means `amt_reviews == 0`
- `last_dt` missing often means no recent activity
- the pair `(has_reviews, has_last_dt)` can separate several listing states

Host-derived features worth trying:
- `log1p(total_host)`
- `is_professional_host`
- bins for `total_host`
- interaction `total_host x location_cluster`
- interaction `total_host x type_house`

Host behavior can also help explain price and demand positioning:
- a host with many listings may optimize occupancy differently
- a single-listing host may be more idiosyncratic
- room type and host portfolio size together often separate private occasional hosts from professional operators

These feature families are attractive for RAG because they are easy to compute from existing columns and often complement spatial features well.
