"""Evaluate a cost-aware trigger for online MWPM decoder recalibration.

An independently sampled utility surface estimates the logical error rate for
each pair of true and decoder-assumed measurement error probabilities.  During
evaluation, completed syndrome windows estimate the current noise.  A decoder
is rebuilt only when the conservative predicted number of avoided logical
failures over a short planning horizon exceeds a user-specified update cost.

The utility-training samples and evaluation streams use disjoint deterministic
seed namespaces.  Information from epoch t can only change decoding at t + 1.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_online_adaptation import (  # noqa: E402
    BASES,
    Config,
    build_calibration_curve,
    confidence_recalibration_step,
    decode_failures,
    decoder_for,
    deterministic_seed,
    estimate_q,
    make_circuit,
    scenario_schedule,
)


RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
# Equivalent logical-failure cost assigned to one decoder reconstruction.  The
# wide range is intentional: low values prioritize logical performance, while
# high values expose the point at which waiting becomes preferable.
COSTS = (0.5, 5.0, 10.0, 25.0, 50.0, 100.0)


def cost_method(cost: float) -> str:
    return f"cost_{cost:g}"


@dataclass
class UtilitySurface:
    """Interpolated held-out logical loss for (true q, assumed q)."""

    true_grid: np.ndarray
    assumed_grid: np.ndarray
    rates: np.ndarray

    def __post_init__(self) -> None:
        self._interpolator = RegularGridInterpolator(
            (self.true_grid, self.assumed_grid),
            self.rates,
            bounds_error=False,
            fill_value=None,
        )

    def loss(self, true_q: float, assumed_q: float) -> float:
        true_q = float(np.clip(true_q, self.true_grid[0], self.true_grid[-1]))
        assumed_q = float(
            np.clip(assumed_q, self.assumed_grid[0], self.assumed_grid[-1])
        )
        return float(self._interpolator((true_q, assumed_q)))

    def conservative_benefit(
        self,
        current_q: float,
        candidate_q: float,
        q_low: float,
        q_high: float,
        points: int = 7,
    ) -> float:
        """Worst predicted per-shot benefit across an estimation interval."""
        interval = np.linspace(q_low, q_high, points)
        improvements = [
            self.loss(q, current_q) - self.loss(q, candidate_q) for q in interval
        ]
        return float(min(improvements))


def cost_aware_recalibration_step(
    *,
    current_q: float,
    candidate_q: float,
    q_low: float,
    q_high: float,
    utility: UtilitySurface,
    future_shots: int,
    update_cost: float,
    min_update_delta: float,
) -> tuple[float, bool, float]:
    """Return next q, decision, and conservative avoided-failure estimate."""
    conservative_rate = utility.conservative_benefit(
        current_q, candidate_q, q_low, q_high
    )
    predicted_avoided = conservative_rate * future_shots
    update = (
        predicted_avoided > update_cost
        and abs(candidate_q - current_q) >= min_update_delta
    )
    return (candidate_q if update else current_q), update, predicted_avoided


def build_utility_surface(
    config: Config,
    q_grid: np.ndarray,
    shots_per_basis: int,
) -> tuple[UtilitySurface, pd.DataFrame]:
    """Build a utility table from samples independent of online evaluation."""
    rates = np.zeros((len(q_grid), len(q_grid)), dtype=float)
    rows: list[dict] = []
    for true_index, true_q in enumerate(q_grid):
        samples: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for basis_index, basis in enumerate(BASES):
            circuit = make_circuit(
                basis=basis,
                distance=config.distance,
                rounds=config.rounds,
                base_error_rate=config.base_error_rate,
                measurement_error_rate=float(true_q),
            )
            # Namespace 700 is reserved for utility training. Evaluation uses
            # namespace 900 below, so the trigger cannot memorize test shots.
            seed = deterministic_seed(config.seed, 700, true_index, basis_index)
            samples[basis] = circuit.compile_detector_sampler(seed=seed).sample(
                shots=shots_per_basis,
                separate_observables=True,
            )

        for assumed_index, assumed_q in enumerate(q_grid):
            failures = 0
            for basis in BASES:
                detections, observables = samples[basis]
                failures += int(
                    np.count_nonzero(
                        decode_failures(
                            decoder_for(config, basis, float(assumed_q)),
                            detections,
                            observables,
                        )
                    )
                )
            rate = failures / (shots_per_basis * len(BASES))
            rates[true_index, assumed_index] = rate
            rows.append(
                {
                    "true_measurement_error_rate": float(true_q),
                    "decoder_measurement_error_rate": float(assumed_q),
                    "utility_training_shots": shots_per_basis * len(BASES),
                    "logical_failures": failures,
                    "logical_error_rate": rate,
                }
            )
        print(f"utility surface {true_index + 1}/{len(q_grid)}: q={true_q:.4f}")
    return UtilitySurface(q_grid, q_grid, rates), pd.DataFrame(rows)


def run_cost_scenario(
    *,
    config: Config,
    calibration: pd.DataFrame,
    utility: UtilitySurface,
    scenario_index: int,
    replicate: int,
    planning_horizon: int,
    costs: tuple[float, ...] = COSTS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run a held-out step-drift stream with paired samples across policies."""
    schedule = scenario_schedule("step", config)
    decoder_q = {
        "naive": config.initial_measurement_error,
        "confidence": config.initial_measurement_error,
        **{cost_method(cost): config.initial_measurement_error for cost in costs},
    }
    methods = (
        "static",
        "naive",
        "confidence",
        *(cost_method(cost) for cost in costs),
        "oracle",
    )
    result_rows: list[dict] = []
    adaptation_rows: list[dict] = []

    for epoch, true_q in enumerate(schedule):
        samples: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        shot_means: list[np.ndarray] = []
        for basis_index, basis in enumerate(BASES):
            circuit = make_circuit(
                basis=basis,
                distance=config.distance,
                rounds=config.rounds,
                base_error_rate=config.base_error_rate,
                measurement_error_rate=float(true_q),
            )
            seed = deterministic_seed(
                config.seed, 900, scenario_index, replicate, epoch, basis_index
            )
            detections, observables = circuit.compile_detector_sampler(seed=seed).sample(
                shots=config.shots_per_epoch,
                separate_observables=True,
            )
            samples[basis] = detections, observables
            shot_means.append(detections.mean(axis=1))

        q_hat, q_low, q_high, dfr, dfr_se = estimate_q(
            np.concatenate(shot_means), calibration, config.confidence_z
        )
        q_used = {
            "static": config.initial_measurement_error,
            **decoder_q,
            "oracle": float(true_q),
        }
        failures_by_method_basis: dict[tuple[str, str], np.ndarray] = {}
        for basis in BASES:
            detections, observables = samples[basis]
            for method in methods:
                failures_by_method_basis[(method, basis)] = decode_failures(
                    decoder_for(config, basis, q_used[method]),
                    detections,
                    observables,
                )

        for basis in BASES:
            oracle_fail = failures_by_method_basis[("oracle", basis)]
            for method in methods:
                method_fail = failures_by_method_basis[(method, basis)]
                result_rows.append(
                    {
                        "epoch": epoch,
                        "basis": basis,
                        "method": method,
                        "update_cost": (
                            float(method.split("_")[1])
                            if method.startswith("cost_") else np.nan
                        ),
                        "distance": config.distance,
                        "shots": config.shots_per_epoch,
                        "true_measurement_error_rate": float(true_q),
                        "decoder_measurement_error_rate": q_used[method],
                        "estimated_measurement_error_rate": q_hat,
                        "logical_failures": int(np.count_nonzero(method_fail)),
                        "method_fails_oracle_succeeds": int(
                            np.count_nonzero(method_fail & ~oracle_fail)
                        ),
                        "method_succeeds_oracle_fails": int(
                            np.count_nonzero(~method_fail & oracle_fail)
                        ),
                    }
                )

        old_q = decoder_q.copy()
        decoder_q["naive"] = q_hat

        decoder_q["confidence"], confidence_update, _ = (
            confidence_recalibration_step(
                current_q=decoder_q["confidence"],
                q_hat=q_hat,
                q_low=q_low,
                q_high=q_high,
                detector_firing_rate=dfr,
                detector_firing_rate_se=dfr_se,
                calibration=calibration,
                z=config.confidence_z,
                min_update_delta=config.min_update_delta,
            )
        )

        row = {
            "epoch": epoch,
            "true_measurement_error_rate": float(true_q),
            "estimated_measurement_error_rate": q_hat,
            "estimate_ci_low": q_low,
            "estimate_ci_high": q_high,
            "detector_firing_rate": dfr,
            "detector_firing_rate_se": dfr_se,
            "confidence_q_used": old_q["confidence"],
            "confidence_updated_after_epoch": confidence_update,
        }
        future_shots = planning_horizon * config.shots_per_epoch * len(BASES)
        for cost in costs:
            method = cost_method(cost)
            decoder_q[method], update, predicted_avoided = (
                cost_aware_recalibration_step(
                    current_q=decoder_q[method],
                    candidate_q=q_hat,
                    q_low=q_low,
                    q_high=q_high,
                    utility=utility,
                    future_shots=future_shots,
                    update_cost=cost,
                    min_update_delta=config.min_update_delta,
                )
            )
            row[f"{method}_q_used"] = old_q[method]
            row[f"{method}_predicted_avoided_failures"] = predicted_avoided
            row[f"{method}_updated_after_epoch"] = update
            row[f"{method}_q_next"] = decoder_q[method]
        adaptation_rows.append(row)

    return pd.DataFrame(result_rows), pd.DataFrame(adaptation_rows)


def summarize(
    results: pd.DataFrame,
    adaptation: pd.DataFrame,
    costs: tuple[float, ...] = COSTS,
) -> pd.DataFrame:
    grouped = (
        results.groupby(["high_measurement_error", "replicate", "method"], as_index=False)[
            [
                "shots",
                "logical_failures",
                "method_fails_oracle_succeeds",
                "method_succeeds_oracle_fails",
            ]
        ]
        .sum()
    )
    update_rows: list[dict] = []
    for (high_q, replicate), frame in adaptation.groupby(
        ["high_measurement_error", "replicate"]
    ):
        update_rows.extend(
            [
                {
                    "high_measurement_error": high_q,
                    "replicate": replicate,
                    "method": "static",
                    "updates": 0,
                },
                {
                    "high_measurement_error": high_q,
                    "replicate": replicate,
                    "method": "naive",
                    "updates": len(frame),
                },
                {
                    "high_measurement_error": high_q,
                    "replicate": replicate,
                    "method": "confidence",
                    "updates": int(frame["confidence_updated_after_epoch"].sum()),
                },
                {
                    "high_measurement_error": high_q,
                    "replicate": replicate,
                    "method": "oracle",
                    "updates": 0,
                },
                *[
                    {
                        "high_measurement_error": high_q,
                        "replicate": replicate,
                        "method": cost_method(cost),
                        "updates": int(
                            frame[f"{cost_method(cost)}_updated_after_epoch"].sum()
                        ),
                    }
                    for cost in costs
                ],
            ]
        )
    grouped = grouped.merge(pd.DataFrame(update_rows), on=[
        "high_measurement_error", "replicate", "method"
    ])
    per_amplitude = (
        grouped.groupby(["high_measurement_error", "method"], as_index=False)
        .agg(
            shots=("shots", "sum"),
            logical_failures=("logical_failures", "sum"),
            logical_failures_mean=("logical_failures", "mean"),
            logical_failures_sd=("logical_failures", "std"),
            updates_mean=("updates", "mean"),
            updates_sd=("updates", "std"),
            updates_total=("updates", "sum"),
            method_fails_oracle_succeeds=(
                "method_fails_oracle_succeeds", "sum"
            ),
            method_succeeds_oracle_fails=(
                "method_succeeds_oracle_fails", "sum"
            ),
        )
    )
    per_amplitude["logical_error_rate"] = (
        per_amplitude["logical_failures"] / per_amplitude["shots"]
    )
    oracle_rate = (
        per_amplitude[per_amplitude["method"] == "oracle"]
        .set_index("high_measurement_error")["logical_error_rate"]
    )
    per_amplitude["relative_to_oracle"] = per_amplitude[
        "logical_error_rate"
    ] / per_amplitude["high_measurement_error"].map(oracle_rate)
    per_amplitude["paired_excess_failures_vs_oracle"] = (
        per_amplitude["method_fails_oracle_succeeds"]
        - per_amplitude["method_succeeds_oracle_fails"]
    )
    per_amplitude["paired_excess_per_100k"] = (
        100_000
        * per_amplitude["paired_excess_failures_vs_oracle"]
        / per_amplitude["shots"]
    )
    return per_amplitude.sort_values(
        ["high_measurement_error", "method"]
    ).reset_index(drop=True)


def aggregate_tradeoff(summary: pd.DataFrame, replicates: int) -> pd.DataFrame:
    """Aggregate all drift amplitudes into one performance-cost frontier."""
    aggregate = (
        summary.groupby("method", as_index=False)
        .agg(
            shots=("shots", "sum"),
            logical_failures=("logical_failures", "sum"),
            paired_excess_failures_vs_oracle=(
                "paired_excess_failures_vs_oracle", "sum"
            ),
            updates_total=("updates_total", "sum"),
        )
    )
    aggregate["logical_error_rate"] = (
        aggregate["logical_failures"] / aggregate["shots"]
    )
    oracle_rate = float(
        aggregate.loc[aggregate["method"] == "oracle", "logical_error_rate"].iloc[0]
    )
    aggregate["relative_to_oracle"] = (
        aggregate["logical_error_rate"] / oracle_rate
    )
    aggregate["updates_per_six_scenarios"] = (
        aggregate["updates_total"] / replicates
    )
    aggregate["policy_update_cost"] = aggregate["method"].map(
        lambda method: float(method.split("_")[1])
        if method.startswith("cost_") else np.nan
    )
    aggregate["policy_total_objective"] = (
        aggregate["logical_failures"]
        + aggregate["policy_update_cost"] * aggregate["updates_total"]
    )
    return aggregate.sort_values("logical_failures").reset_index(drop=True)


def build_block_summary(
    results: pd.DataFrame,
    adaptation: pd.DataFrame,
) -> pd.DataFrame:
    """Create the compact 30-block table used by paired interval estimates."""
    failures = (
        results.groupby(
            ["high_measurement_error", "replicate", "method"], as_index=False
        )["logical_failures"]
        .sum()
        .pivot(
            index=["high_measurement_error", "replicate"],
            columns="method",
            values="logical_failures",
        )
        .reset_index()
    )
    updates = adaptation.groupby(
        ["high_measurement_error", "replicate"], as_index=False
    ).agg(
        confidence_updates=("confidence_updated_after_epoch", "sum"),
        cost_0_5_updates=("cost_0.5_updated_after_epoch", "sum"),
        cost_5_updates=("cost_5_updated_after_epoch", "sum"),
    )
    blocks = failures.merge(
        updates,
        on=["high_measurement_error", "replicate"],
        how="left",
        validate="one_to_one",
    )
    blocks["cost_0_5_minus_confidence"] = (
        blocks["cost_0.5"] - blocks["confidence"]
    )
    blocks["cost_5_minus_confidence"] = (
        blocks["cost_5"] - blocks["confidence"]
    )
    return blocks.rename(
        columns={
            "cost_0.5": "cost_0_5_failures",
            "cost_5": "cost_5_failures",
            "confidence": "confidence_failures",
            "oracle": "oracle_failures",
            "static": "static_failures",
        }
    ).sort_values(["high_measurement_error", "replicate"])


def plot_results(summary: pd.DataFrame, initial_q: float, output: Path) -> None:
    cost_methods = [cost_method(cost) for cost in COSTS]
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(cost_methods)))
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.5), constrained_layout=True)
    for method, color in zip(cost_methods, colors):
        group = summary[summary["method"] == method]
        drift = 100 * (group["high_measurement_error"] - initial_q)
        axes[0].plot(
            drift, group["paired_excess_per_100k"], marker="o",
            color=color, label=method.replace("cost_", "cost="),
        )
        axes[1].plot(
            drift, group["updates_mean"], marker="o", color=color,
            label=method.replace("cost_", "cost="),
        )

    for method, color, style in (
        ("static", "#6B7280", "--"),
        ("naive", "#D97706", ":"),
        ("confidence", "#047857", "-."),
    ):
        group = summary[summary["method"] == method]
        drift = 100 * (group["high_measurement_error"] - initial_q)
        axes[0].plot(
            drift, group["paired_excess_per_100k"], color=color,
            linestyle=style, marker="s", linewidth=1.5, label=method,
        )
        axes[1].plot(
            drift, group["updates_mean"], color=color,
            linestyle=style, marker="s", linewidth=1.5, label=method,
        )

    aggregate = (
        summary[summary["method"] != "oracle"]
        .groupby("method", as_index=False)
        .agg(
            mean_excess_per_100k=("paired_excess_per_100k", "mean"),
            mean_updates=("updates_mean", "mean"),
        )
    )
    coordinate_groups: dict[tuple[float, float], list[str]] = {}
    for row in aggregate.itertuples(index=False):
        color = "#047857" if row.method == "confidence" else (
            "#D97706" if row.method == "naive" else "#6B7280"
        )
        if row.method.startswith("cost_"):
            color = colors[cost_methods.index(row.method)]
        axes[2].scatter(
            row.mean_updates, row.mean_excess_per_100k, s=55, color=color
        )
        key = (round(row.mean_updates, 6), round(row.mean_excess_per_100k, 6))
        coordinate_groups.setdefault(key, []).append(
            row.method.replace("cost_", "c=")
        )
    for (x_value, y_value), labels in coordinate_groups.items():
        axes[2].annotate(
            "/".join(label.replace("c=", "") for label in labels)
            if len(labels) > 1 and all(label.startswith("c=") for label in labels)
            else "/".join(labels),
            (x_value, y_value), xytext=(4, 4), textcoords="offset points", fontsize=8,
        )

    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set_title("Held-out logical penalty")
    axes[0].set_ylabel("Paired excess failures per 100k shots")
    axes[1].set_title("Recalibration response")
    axes[1].set_ylabel("Mean updates per 18 epochs")
    axes[2].set_title("Aggregate performance-cost frontier")
    axes[2].set_xlabel("Mean updates")
    axes[2].set_ylabel("Mean excess failures per 100k")
    for ax in axes[:2]:
        ax.set_xlabel("Step size in measurement error (percentage points)")
        ax.grid(True, alpha=0.25)
    axes[2].grid(True, alpha=0.25)
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle("Cost-aware online decoder recalibration", fontsize=14)
    fig.savefig(output, dpi=220, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    high_q_values = (0.006, 0.0075, 0.010, 0.015, 0.020, 0.030)
    replicates = 5
    planning_horizon = 4
    base = Config(
        distance=5,
        rounds=5,
        epochs=18,
        shots_per_epoch=3_000,
        calibration_shots=15_000,
        calibration_points=35,
        high_measurement_error=max(high_q_values),
        burst_measurement_error=0.035,
        seed=20261119,
    )
    calibration = build_calibration_curve(base)
    utility_grid = np.array(
        [0.003, 0.005, 0.0075, 0.010, 0.014, 0.018, 0.023, 0.029, 0.036]
    )
    utility, utility_table = build_utility_surface(
        base, utility_grid, shots_per_basis=30_000
    )
    utility_table.to_csv(RESULTS / "cost_aware_utility_surface.csv", index=False)

    result_frames: list[pd.DataFrame] = []
    adaptation_frames: list[pd.DataFrame] = []
    for amplitude_index, high_q in enumerate(high_q_values):
        for replicate in range(replicates):
            config = replace(
                base,
                high_measurement_error=high_q,
                seed=base.seed + amplitude_index * 100_000 + replicate * 10_000,
            )
            print(
                f"evaluation high_q={high_q:.4f}, "
                f"replicate={replicate + 1}/{replicates}"
            )
            results, adaptation = run_cost_scenario(
                config=config,
                calibration=calibration,
                utility=utility,
                scenario_index=amplitude_index,
                replicate=replicate,
                planning_horizon=planning_horizon,
            )
            results["high_measurement_error"] = high_q
            results["replicate"] = replicate
            adaptation["high_measurement_error"] = high_q
            adaptation["replicate"] = replicate
            result_frames.append(results)
            adaptation_frames.append(adaptation)

    results = pd.concat(result_frames, ignore_index=True)
    adaptation = pd.concat(adaptation_frames, ignore_index=True)
    summary = summarize(results, adaptation)
    tradeoff = aggregate_tradeoff(summary, replicates)
    blocks = build_block_summary(results, adaptation)
    results.to_csv(RESULTS / "cost_aware_epoch_results.csv", index=False)
    adaptation.to_csv(RESULTS / "cost_aware_adaptation_log.csv", index=False)
    summary.to_csv(RESULTS / "cost_aware_summary.csv", index=False)
    tradeoff.to_csv(RESULTS / "cost_aware_tradeoff_summary.csv", index=False)
    blocks.to_csv(RESULTS / "cost_aware_block_summary.csv", index=False)
    plot_results(summary, base.initial_measurement_error, FIGURES / "cost_aware.png")
    print("\nCost-aware summary")
    print(
        summary[
            [
                "high_measurement_error", "method", "logical_failures",
                "relative_to_oracle", "paired_excess_per_100k", "updates_mean",
            ]
        ].to_string(index=False)
    )
    print("\nAggregate tradeoff")
    print(tradeoff.to_string(index=False))


if __name__ == "__main__":
    main()
