# AlphaNLI Scale-Step 50

This file is the anti-cherrypicking scale-step evaluation set for the abductive regime.
Construction rule:
- start from the existing 50-item AlphaNLI reference pool;
- apply no additional item-level exclusions at this scale step;
- use this set as the broad reference evaluation while keeping the defended 30-item curated set for the main mechanistic comparison;
- if the directional result survives here, then a larger 100-scale expansion is justified.

Final scale-step size: 50

This artifact is intentionally broader and noisier than the defended curated set.
Its purpose is to test whether the AlphaNLI regime-level pattern survives beyond the stricter curated slice.
