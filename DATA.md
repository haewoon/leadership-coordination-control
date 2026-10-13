# Data

Source datasets are **not redistributed** in this repository. We ship only
(a) the construction/normalization scripts and (b) the curated-split selection
and exclusion rules (`data/splits/*_notes.md`). To reproduce the splits,
download each source dataset from its origin and run the corresponding script;
each dataset remains under **its own upstream license**, which you should
consult at the source. Nothing here is relicensed under this repository's MIT
license.

Expected layout (both directories are git-ignored; you create them locally):

- `data/raw/` original or near-original dataset files
- `data/processed/` normalized JSONL consumed by the runner

## Datasets used

| Regime | Dataset | Source (consult for license) | Build / prepare script |
|---|---|---|---|
| Closed-ended QA | CommonsenseQA | Talmor et al., 2019 (allenai) | `scripts/prepare_closed_ended.py`, `scripts/build_closed_ended_candidate_*.py`, `scripts/download_hf_closed_ended.py` |
| Closed-ended QA | StrategyQA | Geva et al., 2021 (allenai) | `scripts/prepare_closed_ended.py`, `scripts/build_closed_ended_candidate_*.py` |
| Abductive ambiguity | AlphaNLI / ART | Bhagavatula et al., 2020 (allenai) | `scripts/prepare_alpha_nli.py`, `scripts/build_alphanli_*.py` |
| Social-norm ambiguity | Social Chemistry 101 | Forbes et al., 2020 (allenai) | `scripts/prepare_social_chemistry*.py`, `scripts/build_social_chemistry_*.py` |
| Math extension | MATH-500 (Level 5 subset of MATH) | Hendrycks et al., 2021 | `scripts/build_math500_pilot.py` |
| Mixed workload | composed from the above | (per component dataset) | `scripts/build_mixed_benchmark_150.py` |
| Boundary probe | ANLI R3 | Nie et al., 2020 (facebookresearch) | `scripts/build_ambiguity_regimes.py` |
| Boundary probe | Winogrande XL | Sakaguchi et al., 2021 (allenai) | `scripts/build_ambiguity_regimes.py` |
| Boundary probe | Scruples (r/AmITheAsshole anecdotes) | Lourie et al., 2021 (allenai) | `scripts/build_scruples_regime.py` |

## Reproducing a curated split

1. Download the source dataset into `data/raw/` (see the source for access and
   license terms).
2. Run the listed `prepare_*` script to normalize it into `data/processed/`.
3. Run the listed `build_*` script to produce the curated split; the exact
   selection thresholds, areas, and exclusion rules are documented in the
   matching `data/splits/<split>_notes.md`.

The curated split note files are descriptive (selection rules, counts, and
rationale), not redistributions of the underlying items. Note files exist for
the main-matrix curated splits; the boundary-probe splits (ANLI R3,
Winogrande XL, Scruples, MATH-500 Level 5) are constructed entirely by their
build scripts, whose docstrings state the selection rules.
