# GitHub NYC Airbnb Baseline

Public Airbnb baselines on GitHub usually agree on one thing: location, room type, reviews, and host behavior dominate early model quality. Even when the exact target differs from our Kaggle task, these projects are useful because they show which feature families are consistently strong on New York rental data.

The most reusable patterns from public baselines are:
- treat `type_house` as a major price-tier signal
- encode borough and neighborhood information
- derive spatial features from `lat` and `lon`
- use review counts and review freshness
- separate casual hosts from multi-listing hosts

What is often missing from lightweight public baselines is a better second wave of features. For our RAG system, we should go beyond the usual `log_sum + distance + has_reviews` trio and try richer but still implementable ideas:
- neighborhood rarity indicator
- coordinate bins or clusters
- price bucket within borough
- `type_house x borough`
- `type_house x log_sum`
- `recency_bucket x borough`
- `professional_host x room_type`

Another lesson from public baselines is that small text signals can help. The listing title `name` is often ignored, but simple text-derived features may add useful texture:
- presence of marketing keywords
- mention of room type in title
- title length and token count
- missing-title indicator

The practical takeaway for our pipeline is that retrieved ideas should stay concrete and column-grounded. A useful RAG answer is not “improve spatial modeling”, but something implementable like “bucket lat/lon into geo cells and add a rare-neighborhood flag from location frequency”.
