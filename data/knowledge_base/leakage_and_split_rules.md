# Leakage And Split Rules For The Agent

The agent should never suggest or implement features derived from `target`. This includes direct target transforms, neighborhood target means fitted on the full dataset, or any feature that would require access to the true label at prediction time.

Safe versus unsafe patterns:
- safe: `log1p(sum)`
- safe: `days_since_last_review`
- safe: location frequency fitted on train
- unsafe: `target_mean_by_location` fitted on all rows
- unsafe: any feature computed from `test["target"]`
- unsafe: random split when `last_dt` is the time proxy

Split rules:
- parse `last_dt`
- sort by time proxy
- validate with `TimeSeriesSplit`
- keep preprocessing consistent across folds

Encoding rules:
- use raw categorical features with CatBoost when possible
- if computing frequency or aggregated statistics, fit them on training folds only
- avoid leaking validation or test information into train-derived mappings

Test-time rules:
- `test.csv` has no `target`
- generated code must not assume otherwise
- any transformation that needs target can only exist inside fold-local training logic

This playbook is included to stop the RAG agent from repeatedly proposing superficially strong but invalid feature ideas.
