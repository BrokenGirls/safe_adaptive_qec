"""Test adaptive-decoder performance as surface-code distance increases."""

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


def summarize_scaling(results: pd.DataFrame, adaptation: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        results.groupby(["distance", "method"], as_index=False)[
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
        .set_index("distance")["logical_failures"]
    )
    oracle_rates = (
        grouped[grouped["method"] == "oracle"]
        .set_index("distance")["logical_error_rate"]
    )
    static_failures = (
        grouped[grouped["method"] == "static"]
        .set_index("distance")["logical_failures"]
    )
    grouped["oracle_logical_error_rate"] = grouped["distance"].map(oracle_rates)
    grouped["relative_to_oracle"] = (
        grouped["logical_error_rate"] / grouped["oracle_logical_error_rate"]
    )
    grouped["paired_excess_failures_vs_oracle"] = (
        grouped["method_fails_oracle_succeeds"]
        - grouped["method_succeeds_oracle_fails"]
    )
    recovery = []
    for row in grouped.itertuples(index=False):
        available = static_failures[row.distance] - oracle_failures[row.distance]
        gained = static_failures[row.distance] - row.logical_failures
        recovery.append(gained / available if available else np.nan)
    grouped["fraction_oracle_improvement_recovered"] = recovery

    per_rep_updates = (
        adaptation.groupby(["distance", "replicate"], as_index=False)
        .agg(
            confidence_updates=("confidence_updated_after_epoch", "sum"),
            periodic_updates=("periodic_updated_after_epoch", "sum"),
            estimation_rmse=(
                "squared_estimation_error",
                lambda values: float(np.sqrt(np.mean(values))),
            ),
        )
    )
    update_summary = (
        per_rep_updates.groupby("distance", as_index=False)
        .agg(
            confidence_updates_mean=("confidence_updates", "mean"),
            confidence_updates_sd=("confidence_updates", "std"),
            estimation_rmse_mean=("estimation_rmse", "mean"),
            estimation_rmse_sd=("estimation_rmse", "std"),
        )
    )
    grouped = grouped.merge(update_summary, on="distance", how="left")
    grouped["updates_mean"] = 0.0
    grouped.loc[grouped["method"] == "naive", "updates_mean"] = adaptation[
        "epoch"
    ].nunique()
    grouped.loc[grouped["method"] == "periodic", "updates_mean"] = 4.0
    grouped.loc[grouped["method"] == "confidence", "updates_mean"] = grouped.loc[
        grouped["method"] == "confidence", "confidence_updates_mean"
    ]
    return grouped.sort_values(["distance", "method"]).reset_index(drop=True)


def plot_scaling(summary: pd.DataFrame, output: Path) -> None:
    colors = {
        "static": "#6B7280",
        "periodic": "#7C3AED",
        "naive": "#D97706",
        "confidence": "#047857",
        "oracle": "#1D4ED8",
    }
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), constrained_layout=True)
    for method in ("static", "periodic", "naive", "confidence", "oracle"):
        group = summary[summary["method"] == method].sort_values("distance")
        axes[0, 0].plot(
            group["distance"], group["logical_error_rate"], marker="o",
            linewidth=1.8, color=colors[method], label=method,
        )
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_ylabel("Logical error rate")
    axes[0, 0].set_title("Logical error scaling")

    for method in ("periodic", "naive", "confidence"):
        group = summary[summary["method"] == method].sort_values("distance")
        axes[0, 1].plot(
            group["distance"], 100 * group["fraction_oracle_improvement_recovered"],
            marker="o", linewidth=1.8, color=colors[method], label=method,
        )
    axes[0, 1].axhline(100, color="black", linewidth=1)
    axes[0, 1].set_ylabel("Oracle improvement recovered (%)")
    axes[0, 1].set_title("Benefit recovered from adaptation")

    confidence = summary[summary["method"] == "confidence"].sort_values("distance")
    axes[1, 0].errorbar(
        confidence["distance"], confidence["confidence_updates_mean"],
        yerr=confidence["confidence_updates_sd"].fillna(0), marker="o",
        capsize=3, linewidth=1.8, color=colors["confidence"], label="confidence",
    )
    axes[1, 0].axhline(18, color=colors["naive"], linestyle="--", label="naive")
    axes[1, 0].set_ylabel("Updates per 18 epochs")
    axes[1, 0].set_title("Recalibration frequency")

    axes[1, 1].errorbar(
        confidence["distance"], 100 * confidence["estimation_rmse_mean"],
        yerr=100 * confidence["estimation_rmse_sd"].fillna(0), marker="o",
        capsize=3, linewidth=1.8, color="#0F766E",
    )
    axes[1, 1].set_ylabel("Measurement-q RMSE (percentage points)")
    axes[1, 1].set_title("Syndrome-only estimation error")

    for ax in axes.flat:
        ax.set_xlabel("Code distance")
        ax.set_xticks(confidence["distance"])
        ax.grid(True, which="both", alpha=0.25)
        if ax is not axes[1, 1]:
            ax.legend(fontsize=8)
    fig.suptitle("Distance scaling under a step change in measurement noise", fontsize=14)
    fig.savefig(output, dpi=220, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    distances = (3, 5, 7, 9)
    replicates = 6
    base = Config(
        epochs=18,
        shots_per_epoch=4_000,
        calibration_shots=12_000,
        calibration_points=31,
        periodic_interval=4,
    )
    result_frames = []
    adaptation_frames = []
    calibration_frames = []
    for distance in distances:
        distance_config = replace(base, distance=distance, rounds=distance)
        print(f"Building calibration curve for distance {distance}...")
        calibration = build_calibration_curve(distance_config)
        calibration["distance"] = distance
        calibration_frames.append(calibration)
        for replicate in range(replicates):
            config = replace(
                distance_config,
                seed=base.seed + distance * 100_000 + replicate * 10_000,
            )
            print(f"distance={distance}, replicate={replicate + 1}/{replicates}")
            results, adaptation = run_scenario(
                "step", replicate, config, calibration
            )
            results["replicate"] = replicate
            adaptation["replicate"] = replicate
            adaptation["distance"] = distance
            adaptation["squared_estimation_error"] = (
                adaptation["estimated_measurement_error_rate"]
                - adaptation["true_measurement_error_rate"]
            ) ** 2
            result_frames.append(results)
            adaptation_frames.append(adaptation)

    results = pd.concat(result_frames, ignore_index=True)
    adaptation = pd.concat(adaptation_frames, ignore_index=True)
    calibration = pd.concat(calibration_frames, ignore_index=True)
    summary = summarize_scaling(results, adaptation)
    calibration.to_csv(RESULTS / "distance_calibration_curves.csv", index=False)
    results.to_csv(RESULTS / "distance_scaling_epoch_results.csv", index=False)
    adaptation.to_csv(RESULTS / "distance_scaling_adaptation_log.csv", index=False)
    summary.to_csv(RESULTS / "distance_scaling_summary.csv", index=False)
    plot_scaling(summary, FIGURES / "distance_scaling.png")
    print("\nDistance scaling summary")
    print(
        summary[
            [
                "distance", "method", "logical_failures", "logical_error_rate",
                "relative_to_oracle", "fraction_oracle_improvement_recovered",
                "updates_mean", "estimation_rmse_mean",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()

