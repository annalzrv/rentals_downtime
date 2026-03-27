# Time Series Feature Engineering For Daily Rental Demand

If a dataset includes a review timestamp such as `last_dt`, do not randomize validation. Sort by time and use `TimeSeriesSplit` so later periods are validated on earlier training windows. Random splits leak future behavior patterns and produce unrealistically low error.

Useful time-derived features include year, month, week of year, weekday, weekend flag, quarter, and elapsed days from a reference point. For sparse review data, missing dates can be mapped to a sentinel and paired with a `has_recent_review` or `has_reviews` indicator.

Temporal signals become more useful when combined with listing metadata. Interactions such as `borough x weekday`, `room_type x month`, or `price x recency` can approximate seasonality without requiring sequence models.

For competitions with daily or near-daily booking behavior, check whether very old listings or listings with no recent reviews behave differently from recently active ones. A monotonic recency feature often helps boosted trees.
