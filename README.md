# Leadership as Coordination Control in Multi-Agent LLM Teams

Code and reproducibility artifact for the paper *Leadership as Coordination
Control: Behavioral Signatures and the Recovery-Advantage Boundary in
Multi-Agent LLM Teams* by Haewoon Kwak, published in *Transactions on Machine
Learning Research* (TMLR), 2026.

- Paper and reviews: <https://openreview.net/forum?id=gp77UrXyjT>
- arXiv preprint: <https://arxiv.org/abs/2606.19111>

The study operationalizes three classical leadership styles (transactional,
transformational, situational) as **controllers** over a shared coordination
action vocabulary (`explore`, `revise`, `accept`, `synthesize`), holding the
agent set and final aggregation fixed, and measures their **behavioral
signatures** (majority lock-in, exploration, recovery from an incorrect round-0
consensus) and accuracy across task regimes and open-weight model families. The
headline result is a contingency one: no controller dominates by accuracy;
process-level control adds value only in a restricted regime (an unreliable but
recoverable round-0 majority that plain interaction does not already repair),
presented in the paper as an explanatory hypothesis.

## Canonical reproduce path

**Every reported run is driven by a `src/config/sr0_*.json` config** (the
shared-round-0 family). These configs are the authority on execution settings:
main-matrix runs use `max_tokens=512`, the MATH-500 runs 1024 (3072 for
`gpt-oss-120b`), all at `temperature=0.35`, exactly as stated in the paper's
execution-settings appendix. This artifact is pruned to that canonical path;
superseded pilot-era configs and scripts (earlier model lineups, pre-sr0 shell
drivers) have been removed for clarity. Token-cost figures are computed from
per-call token counts logged in the run traces, not from the generation cap.

1. **Build the task splits.** Download the source datasets (see `DATA.md`) and
   run the `prepare_*`/`build_*` scripts; selection and exclusion rules for
   every curated split are in `data/splits/*_notes.md`.
2. **Configure the backend.** `cp .env.example .env` and fill `LLM_API_KEY` /
   `LLM_BASE_URL` for an OpenAI-compatible endpoint serving the three model
   checkpoints (`gpt-oss-120b`, `llama-4-scout`, `gemma-4-31B-it`).
3. **Generate configs (already included, regenerable).**
   `scripts/gen_sr0_configs.py`, `gen_sr0_ablation_configs.py`,
   `gen_sr0_ambiguity_configs.py`, `gen_sr0_mad_configs.py`,
   `gen_sr0_scruples_configs.py`, `gen_sr0_sens_configs.py`,
   `gen_sr0_coord_configs.py`.
4. **Run.** `scripts/run_sr0_core.py` (main matrix + ablations + probes),
   `run_sr0_ambiguity.py`, `run_sr0_mad.py`, `run_sr0_scruples.py`,
   `run_sr0_sens.py`, `run_sr0_coord.py` (unconstrained-coordinator baseline).
   All are idempotent and resumable. Round 0 is generated once per (task,
   seed) and reused across conditions; the coordinator baseline additionally
   reuses the *logged* round 0 of the original matrix via the
   `shared_round0_source` config key, so its comparisons are confound-free.
5. **Analyze.** Everything reported is produced by `scripts/analysis/sr0_*.py`
   plus the Bass and MATH-500 helpers (map below).

## Paper to code map (canonical entry points)

| Paper element | Where |
|---|---|
| Leadership controllers and baselines | `src/policies/`; pseudocode in paper App. C |
| Shared round-0 protocol and execution loop | `src/runner/` |
| Main accuracy matrix, per-controller gains, behavioral tables | `scripts/analysis/sr0_clean_analysis.py`, `sr0_make_tables.py`, `sr0_make_appendix.py` |
| Recovery / breakage decomposition and boundary probes | `scripts/analysis/sr0_boundary_tables.py`, `sr0_reliability_characterization.py` |
| Formal predictive test (paper Sec. 6.2) | `scripts/analysis/sr0_predictive_analysis.py`; table shipped at `outputs/analysis/sr0_predictive_table.{csv,json}` |
| Case studies (paper App. B.10) | `scripts/analysis/sr0_case_studies.py`; transcripts shipped at `outputs/analysis/sr0_case_studies.txt` |
| Arbitrary-control baseline (paper Sec. 6.1) | `src/policies/random_action_policy.py`; `scripts/gen_sr0_ablation_configs.py` |
| Unconstrained-coordinator baseline (paper App. B.11) | `src/policies/llm_coordinator_policy.py`; `scripts/gen_sr0_coord_configs.py`, `run_sr0_coord.py`, `analysis/sr0_coord_analysis.py` |
| MAD baseline and situational random-switch ablation | `scripts/run_sr0_mad.py`, `scripts/analysis/sr0_mad_table.py` |
| Bass two-component (accept / revise) ablation | `scripts/analysis/bass_*.py`, `sr0_make_ablation_tables.py` |
| MATH-500 Level 5 extension | `scripts/build_math500_pilot.py`, `scripts/analysis/math500_three_seed_aggregator.py` |
| Situational trigger sensitivity | `scripts/run_sr0_sens.py`, `scripts/analysis/sr0_sens_table.py` |
| Figures | `scripts/analysis/sr0_make_figures.py` |
| Statistics helpers (paired permutation, bootstrap CIs, BH) | `scripts/analysis/reviewer_followups.py` (shared helpers), `sr0_clean_analysis.py`, `stochastic_floor.py` |

## Serving note for the coordinator baseline

`llama-4-scout` left the primary backend between the original matrix and the
coordinator-baseline runs, so that model's coordinator rounds 1+ were served
through OpenRouter pinned to a single provider (DeepInfra,
`allow_fallbacks=false`), with round 0 injected verbatim from the original
logs; the paper's App. B.11 discloses this. The `sr0_llama_4_scout_*_coord.json`
configs record the routing. The other two models ran entirely on the original
backend.

## Repository layout

- `src/` experiment code (`agents/`, `policies/`, `runner/`, `models/`,
  `tasks/`, `prompts/`, `evaluation/`, `logging/`, `analysis/`, `config/`)
- `scripts/` dataset construction (`build_*`, `prepare_*`), sr0 config
  generation (`gen_sr0_*`), run drivers (`run_sr0_*`), and analysis
  (`scripts/analysis/`)
- `data/` data layout and curated-split construction notes (see `DATA.md`)
- `outputs/analysis/` shipped machine-readable analysis outputs
- `tests/` test stubs

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Source datasets are not redistributed here. `DATA.md` lists each dataset, its
source, and its license; the `scripts/build_*`/`scripts/prepare_*` utilities
plus `data/splits/*_notes.md` reconstruct the exact curated splits.

## Citation

If you use this code or build on the study, please cite the TMLR paper
(`CITATION.cff` carries the same information):

```bibtex
@article{kwak2026leadership,
  title   = {Leadership as Coordination Control: Behavioral Signatures and the
             Recovery-Advantage Boundary in Multi-Agent {LLM} Teams},
  author  = {Kwak, Haewoon},
  journal = {Transactions on Machine Learning Research},
  issn    = {2835-8856},
  year    = {2026},
  url     = {https://openreview.net/forum?id=gp77UrXyjT}
}
```

This repository matches the TMLR version of the paper. An earlier snapshot
matching the first arXiv version is archived on Zenodo at
<https://doi.org/10.5281/zenodo.20647820>; it predates the revision analyses
(the formal predictive test, the case studies, and the unconstrained-coordinator
baseline).

## License

Code is released under the MIT License (`LICENSE`). Dataset-derived material
follows the upstream dataset licenses (see `DATA.md`) and is not relicensed.
