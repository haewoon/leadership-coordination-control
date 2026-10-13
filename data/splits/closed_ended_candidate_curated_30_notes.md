# Closed-Ended Candidate Curated 30

This file is a conference-scale candidate set for the convergence-heavy closed-ended regime.
Construction rule:
- keep the existing defended curated 6;
- extend with high-scoring stable reference-pool items from CommonsenseQA and StrategyQA;
- maintain a balanced mix of approximately 15 CommonsenseQA and 15 StrategyQA items;
- exclude previously flagged benchmark-fragile or lower-diagnostic items;
- require final human review before the set becomes the paper's final curated closed-ended benchmark.

Final candidate size: 30
CommonsenseQA items: 15
StrategyQA items: 15

## Added beyond the original curated 6
- `09b5f02f1ce6d2a28d70383a239cd1b8`
  - source: `commonsenseqa`; gold: `A`; challenge score: `7.2`
  - details: `{'top_distractor_similarity': 0.4, 'plausible_choice_count': 4, 'duplicate_penalty': 0, 'cue_count': 0}`
- `82bdf702116bca7605cde3954569b095`
  - source: `commonsenseqa`; gold: `C`; challenge score: `6.5`
  - details: `{'top_distractor_similarity': 0.3333, 'plausible_choice_count': 3, 'duplicate_penalty': 0, 'cue_count': 2}`
- `9453b0f2bd3a9ff93c948dde4a7a9bb3`
  - source: `commonsenseqa`; gold: `B`; challenge score: `6.5`
  - details: `{'top_distractor_similarity': 0.0, 'plausible_choice_count': 4, 'duplicate_penalty': 0, 'cue_count': 1}`
- `11486f615f20f2975079df3165eef20e`
  - source: `commonsenseqa`; gold: `C`; challenge score: `6.2`
  - details: `{'top_distractor_similarity': 0.4, 'plausible_choice_count': 3, 'duplicate_penalty': 0, 'cue_count': 1}`
- `0188cfd31403b8fdeb7f506a79f5c12b`
  - source: `commonsenseqa`; gold: `B`; challenge score: `6.0`
  - details: `{'top_distractor_similarity': 0.5, 'plausible_choice_count': 3, 'duplicate_penalty': 0, 'cue_count': 0}`
- `3123cb4a7db53d79651750df6aa65cff`
  - source: `commonsenseqa`; gold: `D`; challenge score: `6.0`
  - details: `{'top_distractor_similarity': 0.3333, 'plausible_choice_count': 3, 'duplicate_penalty': 0, 'cue_count': 1}`
- `48cd96442f2e98d6c4e0fc42dde8a7af`
  - source: `commonsenseqa`; gold: `D`; challenge score: `6.0`
  - details: `{'top_distractor_similarity': 0.3333, 'plausible_choice_count': 3, 'duplicate_penalty': 0, 'cue_count': 1}`
- `53db057803482a444d7dee7c7e2ce2ee`
  - source: `commonsenseqa`; gold: `C`; challenge score: `6.0`
  - details: `{'top_distractor_similarity': 0.3333, 'plausible_choice_count': 3, 'duplicate_penalty': 0, 'cue_count': 1}`
- `59f14bdd1f2898765ec3b1c6656bc77a`
  - source: `commonsenseqa`; gold: `A`; challenge score: `6.0`
  - details: `{'top_distractor_similarity': 0.5, 'plausible_choice_count': 3, 'duplicate_penalty': 0, 'cue_count': 0}`
- `65b8228ef6fbd0b134d604db454c26d5`
  - source: `commonsenseqa`; gold: `A`; challenge score: `6.0`
  - details: `{'top_distractor_similarity': 0.3333, 'plausible_choice_count': 3, 'duplicate_penalty': 0, 'cue_count': 1}`
- `7a3f7b04cd96b88e92a8a49e4ea300d4`
  - source: `commonsenseqa`; gold: `E`; challenge score: `6.0`
  - details: `{'top_distractor_similarity': 0.3333, 'plausible_choice_count': 3, 'duplicate_penalty': 0, 'cue_count': 1}`
- `d4067753425d9abfeb24af22def49b58`
  - source: `commonsenseqa`; gold: `A`; challenge score: `6.0`
  - details: `{'top_distractor_similarity': 0.3333, 'plausible_choice_count': 3, 'duplicate_penalty': 0, 'cue_count': 1}`
- `3486b6b5c69878bb088d`
  - source: `strategyqa`; gold: `no`; challenge score: `10.0`
  - details: `{'cue_count': 2, 'named_entity_count': 4, 'comparator_count': 2, 'negation_count': 0}`
- `af024277ec12d7845001`
  - source: `strategyqa`; gold: `no`; challenge score: `10.0`
  - details: `{'cue_count': 2, 'named_entity_count': 4, 'comparator_count': 2, 'negation_count': 0}`
- `c8e267b583f722ff485e`
  - source: `strategyqa`; gold: `yes`; challenge score: `10.0`
  - details: `{'cue_count': 2, 'named_entity_count': 4, 'comparator_count': 2, 'negation_count': 0}`
- `1647d6f49307d7678d18`
  - source: `strategyqa`; gold: `yes`; challenge score: `9.5`
  - details: `{'cue_count': 3, 'named_entity_count': 2, 'comparator_count': 2, 'negation_count': 0}`
- `1337be5d470d68393db7`
  - source: `strategyqa`; gold: `yes`; challenge score: `9.0`
  - details: `{'cue_count': 2, 'named_entity_count': 3, 'comparator_count': 2, 'negation_count': 0}`
- `307c3ae5cff437eee937`
  - source: `strategyqa`; gold: `yes`; challenge score: `9.0`
  - details: `{'cue_count': 2, 'named_entity_count': 3, 'comparator_count': 2, 'negation_count': 0}`
- `30b5a9505949caa3c4cd`
  - source: `strategyqa`; gold: `no`; challenge score: `9.0`
  - details: `{'cue_count': 2, 'named_entity_count': 3, 'comparator_count': 2, 'negation_count': 0}`
- `3bd6c0cff096123cc207`
  - source: `strategyqa`; gold: `no`; challenge score: `9.0`
  - details: `{'cue_count': 2, 'named_entity_count': 3, 'comparator_count': 2, 'negation_count': 0}`
- `4f08d9a0fba58ad9ca73`
  - source: `strategyqa`; gold: `yes`; challenge score: `9.0`
  - details: `{'cue_count': 2, 'named_entity_count': 3, 'comparator_count': 2, 'negation_count': 0}`
- `6662006c915a62db5a89`
  - source: `strategyqa`; gold: `yes`; challenge score: `9.0`
  - details: `{'cue_count': 2, 'named_entity_count': 3, 'comparator_count': 2, 'negation_count': 0}`
- `71691ae5050e621c9b4c`
  - source: `strategyqa`; gold: `no`; challenge score: `9.0`
  - details: `{'cue_count': 2, 'named_entity_count': 3, 'comparator_count': 2, 'negation_count': 0}`
- `7f8417d42ef9ea1e5a8c`
  - source: `strategyqa`; gold: `yes`; challenge score: `9.0`
  - details: `{'cue_count': 2, 'named_entity_count': 3, 'comparator_count': 2, 'negation_count': 0}`
