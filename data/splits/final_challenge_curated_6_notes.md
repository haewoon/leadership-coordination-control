# Final Challenge Curated 6

This is the last pilot-scale challenge subset before scaling up to a larger curated benchmark.

Retained items:
- have benchmark gold labels that remain defensible under ordinary reading
- still allow different decision rules to surface in traces
- avoid obvious duplicate-answer artifacts
- avoid highly metaphor-sensitive or annotation-fragile wording

Removed from `final_challenge_10`:
- `4c1cb0e95b99f72d55c068ba0255c54d`: many readers prefer `neck`, creating a benchmark-versus-ordinary-reading mismatch.
- `f375ae73d22eb099f403128708e21962`: relatively straightforward household default, lower disagreement value.
- `e239a7c6e9a2efc98ccb`: medically quirky and potentially too niche for the pilot subset.
- `f8cc260e3863b0be4b81`: geographic risk item is valid but less diagnostic than the retained set for profile disagreement.

Purpose:
- verify that the manually curated challenge regime produces interpretable disagreement
- confirm that `flat` and `transactional` differ on coordination cost before scaling to a larger challenge pool
