"""Find when a measurement-noise drift is large enough to justify adaptation."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_online_adaptation import (  # noqa: E402
    Config,
    build_calibration_curve,
    run_scenario,
)


RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


def summarize_amplitudes(results: pd.DataFrame, adaptation: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        results.groupby(["high_measurement_error", "method"], as_index=False)[
            [
                "shots",
                "logical_failures",
                "method_fails_oracle_succeeds",
                "method_succeeds_oracle_fails",
            ]
        ]
        .sum()
    )
    grouped["logical_error_rate"] = grouped["logical_failures"] / grouped["shots"]
    oracle_failures = (
        grouped[grouped["method"] == "oracle"]
        .set_index("high_measurement_error")["logical_failures"]
    )
    oracle_rates = (
        grouped[grouped["method"] == "oracle"]
        .set_index("high_measurement_error")["logical_error_rate"]
    )
    static_failures = (
        grouped[grouped["method"] == "static"]
        .set_index("high_measurement_error")["logical_failures"]
    )
    grouped["relative_to_oracle"] = grouped["logical_error_rate"] / grouped[
        "high_measurement_error"
    ].map(oracle_rates)
    grouped["paired_excess_failures_vs_oracle"] = (
        grouped["method_fails_oracle_succeeds"]
        - grouped["method_succeeds_oracle_fails"]
    )
    grouped["paired_excess_per_100k"] = (
        100_000 * grouped["paired_excess_failures_vs_oracle"] / grouped["shots"]
    )
    recovery = []
    for row in grouped.itertuples(index=False):
        available = static_failures[row.high_measurement_error] - oracle_failures[
            row.high_measurement_error
        ]
        gained = static_failures[row.high_measurement_error] - row.logical_failures
        recovery.append(gained / available if available > 0 else np.nan)
    grouped["fraction_oracle_improvement_recovered"] = recovery

    update_summary = (
        adaptation.groupby(["high_measurement_error", "replicate"], as_index=False)
        .agg(
            confidence_updates=("confidence_updated_after_epoch", "sum"),
            estimation_rmse=(
                "squared_estimation_error",
                lambda values: float(np.sqrt(np.mean(values))),
            ),
        )
        .groupby("high_measurement_error", as_index=False)
        .agg(
            confidence_updates_mean=("confidence_updates", "mean"),
            confidence_updates_sd=("confidence_updates", "std"),
            estimation_rmse_mean=("estimation_rmse", "mean"),
        )
    )
    grouped = grouped.merge(update_summary, on="high_measurement_error", how="left")
    return grouped.sort_values(["high_measurement_error", "method"]).reset_index(drop=True)


def plot_amplitudes(summary: pd.DataFrame, initial_q: float, output: Path) -> None:
    colors = {
        "static": "#6B7280",
        "periodic": "#7C3AED",
        "naive": "#D97706",
        "confidence": "#047857",
    }
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), constrained_layout=True)
    methods = ("static", "periodic", "naive", "confidence")
    for method in methods:
        group = summary[summary["method"] == method].sort_values(
            "high_measurement_error"
        )
        drift_pp = 100 * (group["high_measurement_error"] - initial_q)
        axes[0, 0].plot(
            drift_pp, group["paired_excess_per_100k"], marker="o",
            linewidth=1.8, color=colors[method], label=method,
        )
        axes[0, 1].plot(
            drift_pp, group["relative_to_oracle"], marker="o",
            linewidth=1.8, color=colors[method], label=method,
        )

    confidence = summary[summary["method"] == "confidence"].sort_values(
        "high_measurement_error"
    )
    drift_pp = 100 * (confidence["high_measurement_error"] - initial_q)
    axes[1, 0].errorbar(
        drift_pp, confidence["confidence_updates_mean"],
        yerr=confidence["confidence_updates_sd"].fillna(0), marker="o",
        capsize=3, linewidth=1.8, color=colors["confidence"],
    )
    axes[1, 1].plot(
        drift_pp, 100 * confidence["fraction_oracle_improvement_recovered"],
        marker="o", linewidth=1.8, color=colors["confidence"],
    )

    axes[0, 0].axhline(0, color="black", linewidth=1)
    axes[0, 0].set_ylabel("Paired excess failures per 100k shots")
    axes[0, 0].set_title("Logical cost relative to oracle")
    axes[0, 1].axhline(1, color="black", linewidth=1)
    axes[0, 1].set_ylabel("Logical error rate / oracle")
    axes[0, 1].set_title("Relative decoder performance")
    axes[1, 0].set_ylabel("Confidence updates per 18 epochs")
    axes[1, 0].set_title("Update response to drift amplitude")
    axes[1, 1].axhline(100, color="black", linewidth=1)
    axes[1, 1].set_ylabel("Oracle improvement recovered (%)")
    axes[1, 1].set_title("Benefit captured by confidence policy")
    for ax in axes.flat:
        ax.set_xlabel("Step size in measurement error (percentage points)")
        ax.grid(True, alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].legend(fontsize=8)
    fig.suptitle("When is decoder recalibration worth triggering?", fontsize=14)
    fig.savefig(output, dpi=220, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    high_q_values = (0.006, 0.0075, 0.010, 0.015, 0.020, 0.030)
    replicates = 5
    base = Config(
        distance=5,
        rounds=5,
        epochs=18,
        shots_per_epoch=3_000,
        calibration_shots=15_000,
        calibration_points=35,
        high_measurement_error=max(high_q_values),
        burst_measurement_error=0.035,
        periodic_interval=4,
    )
    calibration = build_calibration_curve(base)
    result_frames = []
    adaptation_frames = []
    for amplitude_index, high_q in enumerate(high_q_values):
        for replicate in range(replicates):
            config = replace(
                base,
                high_measurement_error=high_q,
                seed=base.seed + amplitude_index * 100_000 + replicate * 10_000,
            )
            print(
                f"high_q={high_q:.4f}, replicate={replicate + 1}/{replicates}"
            )
            results, adaptation = run_scenario(
                "step", amplitude_index, config, calibration
            )
            results["high_measurement_error"] = high_q
            results["replicate"] = replicate
            adaptation["high_measurement_error"] = high_q
            adaptation["replicate"] = replicate
            adaptation["squared_estimation_error"] = (
                adaptation["estimated_measurement_error_rate"]
                - adaptation["true_measurement_error_rate"]
            ) ** 2
            result_frames.append(results)
            adaptation_frames.append(adaptation)

    results = pd.concat(result_frames, ignore_index=True)
    adaptation = pd.concat(adaptation_frames, ignore_index=True)
    summary = summarize_amplitudes(results, adaptation)
    results.to_csv(RESULTS / "drift_amplitude_epoch_results.csv", index=False)
    adaptation.to_csv(RESULTS / "drift_amplitude_adaptation_log.csv", index=False)
    summary.to_csv(RESULTS / "drift_amplitude_summary.csv", index=False)
    plot_amplitudes(
        summary,
        base.initial_measurement_error,
        FIGURES / "drift_amplitude_sweep.png",
    )
    print("\nDrift-amplitude summary")
    print(
        summary[
            [
                "high_measurement_error", "method", "logical_failures",
                "logical_error_rate", "relative_to_oracle",
                "paired_excess_per_100k",
                "fraction_oracle_improvement_recovered",
                "confidence_updates_mean",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()

