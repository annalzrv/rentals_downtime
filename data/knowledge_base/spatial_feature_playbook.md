# Spatial Feature Playbook For NYC Rentals

Spatial structure is probably underused if the feature plan only suggests a single Manhattan-distance feature. With `lat`, `lon`, `location_cluster`, and `location`, the agent can propose a much richer family of geography-aware features.

Strong spatial ideas for this dataset:
- haversine distance to Midtown Manhattan
- haversine distance to Lower Manhattan / Financial District
- borough centroid distance
- coordinate bins from rounded `lat` / `lon`
- geo clusters from KMeans on coordinates
- interaction `geo_cluster x type_house`
- interaction `geo_cluster x log_sum`

Neighborhood information should be treated at multiple granularities:
- raw `location_cluster`
- raw `location`
- rare-neighborhood flag
- neighborhood frequency count
- borough-neighborhood consistency checks

Useful price-aware spatial features that do not use the target:
- `sum_minus_borough_median`
- `sum_ratio_to_borough_median`
- `sum_quantile_within_borough`
- `sum_quantile_within_room_type`

Spatial outlier handling can also become a feature:
- latitude/longitude out-of-range flag
- listings far from borough centroid
- isolated geo cells with very low frequency

If only one spatial feature is allowed, distance to Midtown is a baseline. If we want a second wave of ideas that can move MSE, geo bins, location rarity, and price-relative-to-area features are better candidates than repeating the same single-anchor distance formula.
