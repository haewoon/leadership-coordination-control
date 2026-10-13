# Final Challenge 10

This shortlist is a manually curated subset of the stable challenge pool for the foundational shared-information study.

Selection principles:
- Keep items where the gold label is defensible to a human reader without specialist lookup.
- Prefer items that can still trigger different decision rules across `counterexample_first`, `surface_match`, and `plausibility_first`.
- Exclude annotation-artifact items where the benchmark gold answer strongly conflicts with ordinary reading.
- Exclude obviously pathological duplicate-answer items used only for stress testing.

Excluded examples:
- `661919f362d276f8196a590cbf40edf1`: ordinary reading strongly favors `wake up`, while the gold label is `ask spouse to wake`.
- `b65bee87291f9bb63f972a720e34e0e5`: duplicated `ability to fly` options make the label artifact too dominant.
- `8669c9d0666951d25ef0`: phrasing around `level with` remains too metaphor-sensitive for a clean foundational set.
- Several StrategyQA items comparing unusual institutions or media titles were excluded because they are too niche for a compact foundational benchmark.
