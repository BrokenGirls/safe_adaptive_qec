# Safe Adaptive QEC

Reproducible research code for studying when an online surface-code decoder
should trust a noisy recalibration estimate.

The experiment uses:

- **Stim** to generate rotated surface-code memory circuits with drifting
  measurement noise;
- **PyMatching** for minimum-weight perfect-matching decoding;
- a syndrome-only calibration curve to estimate the current measurement error
  probability;
- paired decoding of the same detector samples to compare decoder policies.

## Research question

When measurement noise changes over time, can a confidence-aware update rule
approach an oracle decoder while avoiding harmful updates caused by finite
syndrome samples?

The initial benchmark compares five policies:

1. `static`: never recalibrate;
2. `periodic`: recalibrate every fixed number of windows;
3. `naive`: replace the assumed noise estimate after every window;
4. `confidence`: update only when the old model is outside a confidence
   interval, then shrink the update toward the old model;
5. `oracle`: decode with the true current noise model.

Data from window `t` is only used to update the decoder for window `t+1`; the
experiment therefore does not leak future information into the adaptive
decoders.

## Run

The parent workspace already contains a compatible virtual environment:

```bash
cd safe_adaptive_qec
../.venv/bin/python src/run_online_adaptation.py --quick
../.venv/bin/python src/run_online_adaptation.py
../.venv/bin/python src/run_budget_sweep.py
../.venv/bin/python src/run_distance_scaling.py
../.venv/bin/python src/run_drift_amplitude_sweep.py
../.venv/bin/python src/run_cost_aware_policy.py
```

The full run writes:

- `results/calibration_curve.csv`
- `results/epoch_results.csv`
- `results/adaptation_log.csv`
- `results/summary.csv`
- `figures/online_adaptation.png`
- `results/budget_sweep_summary.csv`
- `figures/budget_sweep.png`
- `results/distance_scaling_summary.csv`
- `figures/distance_scaling.png`
- `results/drift_amplitude_summary.csv`
- `figures/drift_amplitude_sweep.png`
- `results/cost_aware_utility_surface.csv`
- `results/cost_aware_summary.csv`
- `results/cost_aware_tradeoff_summary.csv`
- `figures/cost_aware.png`

## Current scope

The first study varies measurement error while keeping gate, reset, and data
noise fixed. This deliberately isolates one identifiable drift parameter. The
current stage adds a logical-utility trigger for cost-aware recalibration. The
next model extension will add spatial hot spots and simultaneous
data/measurement drift.

## Preliminary result

The first reproducible run used a distance-5 rotated surface code, three drift
patterns, 24 time windows, and 12,000 shots per window per logical basis. The
confidence-aware policy used only 5 updates in the step and burst experiments
(versus 24 for the naive policy) while recovering most of the logical-error
reduction available to the oracle decoder.

The finite-data sweep exposed an important boundary rather than universal
dominance: with only 50 shots per window, confidence gating reacted too slowly;
from roughly 100 shots per window onward, it stayed close to the oracle while
using about 4-6 updates over 18 windows. See `RESEARCH_REPORT.md` for exact
numbers, limitations, and the next experiments required before publication.

A six-replicate distance scan at `d = 3, 5, 7, 9` found that the cost of leaving
the decoder stale increased with distance, while syndrome-only measurement-noise
estimation became more precise. Confidence-aware adaptation recovered 89-97% of
the oracle improvement using an average of 3.8-6.2 updates over 18 windows.

A drift-amplitude scan then identified a failure mode in the current heuristic:
it can detect and react to very small statistically significant changes even
when recalibration has no measurable logical benefit. For measurement-noise
steps of one percentage point or larger, it recovered roughly 90% or more of
the oracle benefit. This motivates the next algorithm: trigger on estimated
logical utility, not significance alone.

The new cost-aware policy learns an independent logical-utility surface and
updates only when a conservative estimate of avoided failures over four future
windows exceeds the reconstruction cost. Across 3.24 million held-out memory
experiments per policy, the low-cost settings reduced updates by about 87-89%
relative to confidence gating while adding only 21-22 logical failures. Across
30 paired amplitude/seed blocks, this small logical difference was not
statistically distinguishable from zero. Larger cost settings trace the
expected frontier toward the static decoder. These results are promising but
still use the same one-parameter simulated noise family for utility training
and evaluation.

## Related work

- A. S. Darmawan, *Optimal adaptation of surface-code decoders to local noise*,
  arXiv:2403.08706.
- A. Remm et al., *Experimentally Informed Decoding of Stabilizer Codes Based
  on Syndrome Correlations*, arXiv:2502.17722.
- D. Bhardwaj et al., *Adaptive Estimation of Drifting Noise in Quantum Error
  Correction*, arXiv:2511.09491.
- C. Gidney, *Stim: a fast stabilizer circuit simulator*, Quantum 5, 497 (2021).
- O. Higgott and C. Gidney, *Sparse Blossom*, arXiv:2303.15933.


