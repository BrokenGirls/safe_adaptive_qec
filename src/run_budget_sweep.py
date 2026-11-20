"""Measure how syndrome-window size changes adaptive-decoder reliability."""

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
    METHODS,
    build_calibration_curve,
    run_scenario,
)


RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


def summarize_budget(results: pd.DataFrame, adaptation: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        results.groupby(["shots_per_epoch", "method"], as_index=False)[
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
    oracle = (
        grouped[grouped["method"] == "oracle"]
        .set_index("shots_per_epoch")["logical_error_rate"]
    )
    grouped["oracle_logical_error_rate"] = grouped["shots_per_epoch"].map(oracle)
    grouped["relative_to_oracle"] = (
        grouped["logical_error_rate"] / grouped["oracle_logical_error_rate"]
    )
    grouped["paired_excess_failures_vs_oracle"] = (
        grouped["method_fails_oracle_succeeds"]
        - grouped["method_succeeds_oracle_fails"]
    )
    grouped["paired_excess_per_100k"] = (
        100_000 * grouped["paired_excess_failures_vs_oracle"] / grouped["shots"]
    )
    updates = (
        adaptation.groupby(["shots_per_epoch", "replicate"], as_index=False)
        .agg(
            confidence_updates=("confidence_updated_after_epoch", "sum"),
            periodic_updates=("periodic_updated_after_epoch", "sum"),
        )
        .groupby("shots_per_epoch", as_index=False)
        .agg(
            confidence_updates_mean=("confidence_updates", "mean"),
            confidence_updates_sd=("confidence_updates", "std"),
            periodic_updates_mean=("periodic_updates", "mean"),
        )
    )
    grouped = grouped.merge(updates, on="shots_per_epoch", how="left")
    grouped["updates_mean"] = 0.0
    grouped.loc[grouped["method"] == "naive", "updates_mean"] = adaptation[
        "epoch"
    ].nunique()
    grouped.loc[grouped["method"] == "confidence", "updates_mean"] = grouped.loc[
        grouped["method"] == "confidence", "confidence_updates_mean"
    ]
    grouped.loc[grouped["method"] == "periodic", "updates_mean"] = grouped.loc[
        grouped["method"] == "periodic", "periodic_updates_mean"
    ]
    return grouped.sort_values(["shots_per_epoch", "method"]).reset_index(drop=True)


def plot_budget(summary: pd.DataFrame, output: Path) -> None:
    colors = {
        "static": "#6B7280",
        "periodic": "#7C3AED",
        "naive": "#D97706",
        "confidence": "#047857",
    }
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.4), constrained_layout=True)
    for method in ("static", "periodic", "naive", "confidence"):
        group = summary[summary["method"] == method].sort_values("shots_per_epoch")
        axes[0].plot(
            group["shots_per_epoch"], group["paired_excess_per_100k"],
            marker="o", linewidth=1.8, color=colors[method], label=method,
        )
        axes[1].plot(
            group["shots_per_epoch"], group["relative_to_oracle"],
            marker="o", linewidth=1.8, color=colors[method], label=method,
        )
    confidence = summary[summary["method"] == "confidence"].sort_values(
        "shots_per_epoch"
    )
    axes[2].errorbar(
        confidence["shots_per_epoch"],
        confidence["confidence_updates_mean"],
        yerr=confidence["confidence_updates_sd"].fillna(0),
        marker="o", capsize=3, color=colors["confidence"],
        label="confidence",
    )
    axes[2].axhline(18, color=colors["naive"], linestyle="--", label="naive")

    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set_ylabel("Paired excess failures per 100k shots")
    axes[0].set_title("Logical cost relative to oracle")
    axes[1].axhline(1, color="black", linewidth=1)
    axes[1].set_ylabel("Logical error rate / oracle")
    axes[1].set_title("Relative decoder performance")
    axes[2].set_ylabel("Updates per 18 epochs")
    axes[2].set_title("Recalibration frequency")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("Syndrome shots per epoch")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Finite-data study under a step change in measurement noise", fontsize=13)
    fig.savefig(output, dpi=220, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    base = Config(epochs=18, calibration_shots=20_000, calibration_points=31)
    calibration = build_calibration_curve(base)
    budgets = (50, 100, 300, 1_000, 3_000, 12_000)
    replicates = 8
    result_frames = []
    adaptation_frames = []
    for budget in budgets:
        for replicate in range(replicates):
            config = replace(
                base,
                shots_per_epoch=budget,
                seed=base.seed + 10_000 * replicate + budget,
            )
            print(f"budget={budget:5d}, replicate={replicate + 1}/{replicates}")
            results, adaptation = run_scenario(
                "step", replicate, config, calibration
            )
            results["shots_per_epoch"] = budget
            results["replicate"] = replicate
            adaptation["shots_per_epoch"] = budget
            adaptation["replicate"] = replicate
            result_frames.append(results)
            adaptation_frames.append(adaptation)

    results = pd.concat(result_frames, ignore_index=True)
    adaptation = pd.concat(adaptation_frames, ignore_index=True)
    summary = summarize_budget(results, adaptation)
    results.to_csv(RESULTS / "budget_sweep_epoch_results.csv", index=False)
    adaptation.to_csv(RESULTS / "budget_sweep_adaptation_log.csv", index=False)
    summary.to_csv(RESULTS / "budget_sweep_summary.csv", index=False)
    plot_budget(summary, FIGURES / "budget_sweep.png")
    print("\nBudget sweep summary")
    print(
        summary[
            [
                "shots_per_epoch", "method", "logical_error_rate",
                "relative_to_oracle", "paired_excess_per_100k", "updates_mean",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()

