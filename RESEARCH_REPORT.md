# Preliminary Research Report

## Confidence-aware online MWPM recalibration under drifting measurement noise

### Question

Can a surface-code decoder use a finite stream of syndrome measurements to
decide when recalibration is justified, while avoiding the cost and instability
of rebuilding decoder weights after every observation window?

### Method

The benchmark uses Stim rotated surface-code memory circuits and PyMatching
minimum-weight perfect matching. Gate, reset, and data noise are fixed at
`p = 0.005`; only the measurement error probability changes with time.

A high-statistics calibration table maps measurement error probability to the
mean detector firing rate. During evaluation, each time window estimates the
current measurement error rate using only its syndrome data. The estimate is
not allowed to affect decoding of that same window; it can only update the next
window. All policies decode the same sampled detector events.

Five policies are compared:

- **static**: retain the initial error model;
- **periodic**: update every four windows;
- **naive**: replace the model after every window;
- **confidence**: update only when the old model is inconsistent with a
  two-standard-error detector-rate interval, then shrink the update;
- **oracle**: use the true current measurement error probability.

The main experiment uses distance 5, five syndrome rounds, 24 time windows,
12,000 shots per window per logical basis, and three drift patterns: step, ramp,
and burst. Each policy therefore decodes 576,000 memory experiments per
scenario.

### Main results

| Drift | Policy | Logical failures | Logical error rate | Excess vs oracle | Updates |
|---|---:|---:|---:|---:|---:|
| Step | static | 13,942 | 0.024205 | 635 | 0 |
| Step | periodic | 13,427 | 0.023311 | 120 | 6 |
| Step | naive | 13,359 | 0.023193 | 52 | 24 |
| Step | confidence | 13,348 | 0.023174 | 41 | 5 |
| Step | oracle | 13,307 | 0.023102 | 0 | 0 |
| Ramp | static | 11,267 | 0.019561 | 273 | 0 |
| Ramp | periodic | 11,014 | 0.019122 | 20 | 6 |
| Ramp | naive | 11,001 | 0.019099 | 7 | 24 |
| Ramp | confidence | 10,993 | 0.019085 | -1 | 17 |
| Ramp | oracle | 10,994 | 0.019087 | 0 | 0 |
| Burst | static | 11,058 | 0.019198 | 313 | 0 |
| Burst | periodic | 10,895 | 0.018915 | 150 | 6 |
| Burst | naive | 10,813 | 0.018773 | 68 | 24 |
| Burst | confidence | 10,807 | 0.018762 | 62 | 5 |
| Burst | oracle | 10,745 | 0.018655 | 0 | 0 |

In the step experiment, confidence-aware recalibration recovered approximately
93.5% of the oracle improvement over the static decoder while making 79% fewer
updates than the naive policy. In the burst experiment it recovered about 80%
of the oracle improvement with the same reduction in updates. The small
negative excess in the ramp experiment is sampling variation, not evidence that
the method surpasses an oracle.

### Finite-data boundary

Eight independent replicates were run at each of six syndrome budgets for a
step change. The confidence-aware policy did not dominate naive adaptation:

- at 50 shots per window, confidence gating was too conservative and reached
  `1.026 x` the oracle logical error rate;
- at 100 shots, its aggregate logical error rate happened to equal the oracle
  result in the sampled data;
- from 300 to 12,000 shots, it remained within about 0.4-0.6% of the oracle;
- it used only 3.6-6.3 updates per 18 windows, versus 18 naive updates.

This is the first useful result of the project: there is a measurable region in
which fewer confidence-gated updates retain nearly all decoder performance, but
overly small windows can delay adaptation and cause avoidable logical errors.

### What the result does not establish

This is a preliminary computational result, not yet a publication claim.

1. Only one code distance and one base physical error rate were used.
2. The changing noise is synthetic Pauli measurement noise.
3. The estimator assumes gate and data noise remain fixed.
4. Detector outcomes within a shot are correlated; the current normal interval
   treats per-shot detector means as independent but does not model all temporal
   correlations.
5. Decoder reconstruction cost is represented by update count, not measured
   real-time latency.
6. The calibration curve is generated from the same simulator family used in
   evaluation, so simulation-to-hardware mismatch remains untested.

### Next experiments

1. Repeat distances 3, 5, 7, and 9 and measure scaling with code distance.
2. Replace the detector-rate lookup with syndrome-pair correlation estimates.
3. Add simultaneous data and measurement drift, local hot qubits, and bursts.
4. Replace the fixed significance threshold with a decision rule that minimizes
   logical regret plus recalibration cost.
5. Test on public hardware syndrome data if a compatible dataset is available.
6. Compare against sliding-window estimation and a change-point baseline.

### Reproducibility

Run:

```bash
../.venv/bin/python -m unittest discover -s tests -v
../.venv/bin/python src/run_online_adaptation.py
../.venv/bin/python src/run_budget_sweep.py
```

The tracked summary CSV files contain the reported aggregate values. Raw
per-window data are reproducible but ignored by Git to keep the repository
small.

