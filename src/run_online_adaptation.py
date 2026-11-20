"""Benchmark safe online recalibration of a surface-code MWPM decoder.

The simulator produces independent surface-code memory experiments in a time
ordered sequence of windows. Measurement noise changes between windows. Each
decoder sees exactly the same detector samples, and adaptive policies may only
use completed windows when choosing weights for the next window.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymatching
from scipy.stats import binomtest
import stim


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
BASES = ("x", "z")
METHODS = ("static", "periodic", "naive", "confidence", "oracle")


@dataclass(frozen=True)
class Config:
    distance: int = 5
    rounds: int = 5
    base_error_rate: float = 0.005
    initial_measurement_error: float = 0.005
    high_measurement_error: float = 0.020
    burst_measurement_error: float = 0.027
    epochs: int = 24
    shots_per_epoch: int = 12_000
    calibration_shots: int = 15_000
    calibration_points: int = 31
    periodic_interval: int = 4
    confidence_z: float = 2.0
    min_update_delta: float = 0.0005
    seed: int = 20261119


def make_circuit(
    *,
    basis: str,
    distance: int,
    rounds: int,
    base_error_rate: float,
    measurement_error_rate: float,
) -> stim.Circuit:
    """Build a circuit with independently controlled measurement noise."""
    if basis not in BASES:
        raise ValueError(f"unsupported basis: {basis}")
    if not 0 <= measurement_error_rate < 0.5:
        raise ValueError("measurement_error_rate must be in [0, 0.5)")
    return stim.Circuit.generated(
        f"surface_code:rotated_memory_{basis}",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=base_error_rate,
        after_reset_flip_probability=base_error_rate,
        before_measure_flip_probability=measurement_error_rate,
        before_round_data_depolarization=base_error_rate,
    )


@lru_cache(maxsize=None)
def cached_decoder(
    basis: str,
    distance: int,
    rounds: int,
    base_error_rate: float,
    measurement_error_rate_rounded: float,
) -> pymatching.Matching:
    circuit = make_circuit(
        basis=basis,
        distance=distance,
        rounds=rounds,
        base_error_rate=base_error_rate,
        measurement_error_rate=measurement_error_rate_rounded,
    )
    dem = circuit.detector_error_model(decompose_errors=True)
    return pymatching.Matching.from_detector_error_model(dem)


def decoder_for(config: Config, basis: str, q: float) -> pymatching.Matching:
    q = float(np.clip(q, 1e-6, 0.49))
    return cached_decoder(
        basis,
        config.distance,
        config.rounds,
        config.base_error_rate,
        round(q, 8),
    )


def deterministic_seed(*parts: int) -> int:
    state = np.random.SeedSequence(parts).generate_state(1, dtype=np.uint32)
    return int(state[0])


def build_calibration_curve(config: Config) -> pd.DataFrame:
    """Map measurement error probability to syndrome firing statistics."""
    upper = max(config.high_measurement_error, config.burst_measurement_error) * 1.15
    q_grid = np.linspace(0.001, upper, config.calibration_points)
    rows: list[dict] = []
    for q_index, q in enumerate(q_grid):
        shot_means: list[np.ndarray] = []
        detector_count = 0
        for basis_index, basis in enumerate(BASES):
            circuit = make_circuit(
                basis=basis,
                distance=config.distance,
                rounds=config.rounds,
                base_error_rate=config.base_error_rate,
                measurement_error_rate=float(q),
            )
            seed = deterministic_seed(config.seed, 100, q_index, basis_index)
            detections = circuit.compile_detector_sampler(seed=seed).sample(
                shots=config.calibration_shots
            )
            detector_count += detections.shape[1]
            shot_means.append(detections.mean(axis=1))
        values = np.concatenate(shot_means)
        standard_error = float(values.std(ddof=1) / math.sqrt(len(values)))
        rows.append(
            {
                "measurement_error_rate": float(q),
                "detector_firing_rate_raw": float(values.mean()),
                "detector_firing_rate_se": standard_error,
                "calibration_shots_per_basis": config.calibration_shots,
                "detectors_across_bases": detector_count,
            }
        )

    frame = pd.DataFrame(rows)
    # Monte Carlo noise can introduce tiny local reversals. Monotonicizing the
    # lookup table makes interpolation well-defined without fitting a flexible
    # model to the evaluation stream.
    frame["detector_firing_rate"] = np.maximum.accumulate(
        frame["detector_firing_rate_raw"].to_numpy()
    )
    return frame


def estimate_q(
    shot_detector_means: np.ndarray,
    calibration: pd.DataFrame,
    z: float,
) -> tuple[float, float, float, float, float]:
    """Estimate q and a normal-approximation interval from syndrome-only data."""
    dfr = float(shot_detector_means.mean())
    se = float(shot_detector_means.std(ddof=1) / math.sqrt(len(shot_detector_means)))
    dfr_low = dfr - z * se
    dfr_high = dfr + z * se
    curve_dfr = calibration["detector_firing_rate"].to_numpy()
    curve_q = calibration["measurement_error_rate"].to_numpy()
    q_hat = float(np.interp(dfr, curve_dfr, curve_q))
    q_low = float(np.interp(dfr_low, curve_dfr, curve_q))
    q_high = float(np.interp(dfr_high, curve_dfr, curve_q))
    return q_hat, q_low, q_high, dfr, se


def expected_dfr(q: float, calibration: pd.DataFrame) -> float:
    return float(
        np.interp(
            q,
            calibration["measurement_error_rate"].to_numpy(),
            calibration["detector_firing_rate"].to_numpy(),
        )
    )


def scenario_schedule(name: str, config: Config) -> np.ndarray:
    q0 = config.initial_measurement_error
    q1 = config.high_measurement_error
    n = config.epochs
    if name == "step":
        change = n // 3
        return np.array([q0] * change + [q1] * (n - change), dtype=float)
    if name == "ramp":
        hold = n // 4
        return np.concatenate(
            [np.full(hold, q0), np.linspace(q0, q1, n - hold)]
        )
    if name == "burst":
        values = np.full(n, q0, dtype=float)
        start = n // 2 - 2
        values[start : start + 4] = config.burst_measurement_error
        return values
    raise ValueError(f"unknown scenario: {name}")


def decode_failures(
    decoder: pymatching.Matching,
    detections: np.ndarray,
    actual_observables: np.ndarray,
) -> np.ndarray:
    predicted = np.asarray(decoder.decode_batch(detections), dtype=bool)
    return np.any(predicted != actual_observables, axis=1)


def run_scenario(
    name: str,
    scenario_index: int,
    config: Config,
    calibration: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    schedule = scenario_schedule(name, config)
    decoder_q = {
        "static": config.initial_measurement_error,
        "periodic": config.initial_measurement_error,
        "naive": config.initial_measurement_error,
        "confidence": config.initial_measurement_error,
    }
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
                config.seed, 200, scenario_index, epoch, basis_index
            )
            detections, observables = circuit.compile_detector_sampler(seed=seed).sample(
                shots=config.shots_per_epoch,
                separate_observables=True,
            )
            samples[basis] = (detections, observables)
            shot_means.append(detections.mean(axis=1))

        q_hat, q_low, q_high, dfr, dfr_se = estimate_q(
            np.concatenate(shot_means), calibration, config.confidence_z
        )
        q_used = {**decoder_q, "oracle": float(true_q)}

        failures_by_method_basis: dict[tuple[str, str], np.ndarray] = {}
        for basis in BASES:
            detections, observables = samples[basis]
            for method in METHODS:
                failures_by_method_basis[(method, basis)] = decode_failures(
                    decoder_for(config, basis, q_used[method]),
                    detections,
                    observables,
                )

        for basis in BASES:
            oracle_fail = failures_by_method_basis[("oracle", basis)]
            for method in METHODS:
                method_fail = failures_by_method_basis[(method, basis)]
                worse = int(np.count_nonzero(method_fail & ~oracle_fail))
                better = int(np.count_nonzero(~method_fail & oracle_fail))
                result_rows.append(
                    {
                        "scenario": name,
                        "epoch": epoch,
                        "basis": basis,
                        "method": method,
                        "distance": config.distance,
                        "rounds": config.rounds,
                        "shots": config.shots_per_epoch,
                        "true_measurement_error_rate": float(true_q),
                        "decoder_measurement_error_rate": q_used[method],
                        "estimated_measurement_error_rate": q_hat,
                        "estimate_ci_low": q_low,
                        "estimate_ci_high": q_high,
                        "detector_firing_rate": dfr,
                        "detector_firing_rate_se": dfr_se,
                        "logical_failures": int(np.count_nonzero(method_fail)),
                        "method_fails_oracle_succeeds": worse,
                        "method_succeeds_oracle_fails": better,
                    }
                )

        previous = decoder_q.copy()
        decoder_q["naive"] = q_hat
        periodic_update = (epoch + 1) % config.periodic_interval == 0
        if periodic_update:
            decoder_q["periodic"] = q_hat

        confidence_expected_dfr = expected_dfr(decoder_q["confidence"], calibration)
        significant = not (
            dfr - config.confidence_z * dfr_se
            <= confidence_expected_dfr
            <= dfr + config.confidence_z * dfr_se
        )
        large_enough = abs(q_hat - decoder_q["confidence"]) >= config.min_update_delta
        confidence_update = significant and large_enough
        alpha = 0.0
        if confidence_update:
            interval_width = max(q_high - q_low, 1e-9)
            displacement = abs(q_hat - decoder_q["confidence"])
            alpha = float(np.clip(1.0 - interval_width / max(displacement, 1e-9), 0.2, 0.9))
            decoder_q["confidence"] += alpha * (
                q_hat - decoder_q["confidence"]
            )

        adaptation_rows.append(
            {
                "scenario": name,
                "epoch": epoch,
                "true_measurement_error_rate": float(true_q),
                "estimated_measurement_error_rate": q_hat,
                "estimate_ci_low": q_low,
                "estimate_ci_high": q_high,
                "static_q_used": previous["static"],
                "periodic_q_used": previous["periodic"],
                "naive_q_used": previous["naive"],
                "confidence_q_used": previous["confidence"],
                "periodic_updated_after_epoch": periodic_update,
                "confidence_updated_after_epoch": confidence_update,
                "confidence_alpha": alpha,
                "periodic_q_next": decoder_q["periodic"],
                "naive_q_next": decoder_q["naive"],
                "confidence_q_next": decoder_q["confidence"],
            }
        )

    return pd.DataFrame(result_rows), pd.DataFrame(adaptation_rows)


def summarize(results: pd.DataFrame, adaptation: pd.DataFrame) -> pd.DataFrame:
    summary = (
        results.groupby(["scenario", "method"], as_index=False)[
            [
                "shots",
                "logical_failures",
                "method_fails_oracle_succeeds",
                "method_succeeds_oracle_fails",
            ]
        ]
        .sum()
    )
    summary["logical_error_rate"] = summary["logical_failures"] / summary["shots"]
    oracle_rates = (
        summary[summary["method"] == "oracle"]
        .set_index("scenario")["logical_error_rate"]
    )
    summary["oracle_logical_error_rate"] = summary["scenario"].map(oracle_rates)
    summary["relative_to_oracle"] = (
        summary["logical_error_rate"] / summary["oracle_logical_error_rate"]
    )
    summary["paired_excess_failures_vs_oracle"] = (
        summary["method_fails_oracle_succeeds"]
        - summary["method_succeeds_oracle_fails"]
    )
    summary["mcnemar_exact_p_value"] = [
        binomtest(int(worse), int(worse + better), p=0.5).pvalue
        if worse + better
        else 1.0
        for worse, better in zip(
            summary["method_fails_oracle_succeeds"],
            summary["method_succeeds_oracle_fails"],
        )
    ]
    update_counts = adaptation.groupby("scenario").agg(
        periodic_updates=("periodic_updated_after_epoch", "sum"),
        confidence_updates=("confidence_updated_after_epoch", "sum"),
    )
    summary = summary.merge(update_counts, on="scenario", how="left")
    summary["updates"] = 0
    summary.loc[summary["method"] == "naive", "updates"] = adaptation.groupby(
        "scenario"
    ).size().reindex(summary.loc[summary["method"] == "naive", "scenario"]).to_numpy()
    summary.loc[summary["method"] == "periodic", "updates"] = summary.loc[
        summary["method"] == "periodic", "periodic_updates"
    ]
    summary.loc[summary["method"] == "confidence", "updates"] = summary.loc[
        summary["method"] == "confidence", "confidence_updates"
    ]
    return summary.sort_values(["scenario", "method"]).reset_index(drop=True)


def plot_results(
    results: pd.DataFrame,
    adaptation: pd.DataFrame,
    summary: pd.DataFrame,
    output: Path,
) -> None:
    colors = {
        "static": "#6B7280",
        "periodic": "#7C3AED",
        "naive": "#D97706",
        "confidence": "#047857",
        "oracle": "#1D4ED8",
    }
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), constrained_layout=True)
    trace = adaptation[adaptation["scenario"] == "step"].copy()
    axes[0, 0].plot(
        trace["epoch"], trace["true_measurement_error_rate"] * 100,
        color="black", linewidth=2.4, label="true q",
    )
    for method, column in (
        ("periodic", "periodic_q_used"),
        ("naive", "naive_q_used"),
        ("confidence", "confidence_q_used"),
    ):
        axes[0, 0].plot(
            trace["epoch"], trace[column] * 100,
            marker="o", markersize=3, linewidth=1.4,
            color=colors[method], label=method,
        )
    axes[0, 0].fill_between(
        trace["epoch"], trace["estimate_ci_low"] * 100,
        trace["estimate_ci_high"] * 100,
        color="#F59E0B", alpha=0.14, label="estimate interval",
    )
    axes[0, 0].set_title("Step drift: assumed measurement noise")
    axes[0, 0].set_ylabel("Measurement error q (%)")
    axes[0, 0].legend(ncol=2, fontsize=8)

    step = (
        results[results["scenario"] == "step"]
        .groupby(["epoch", "method"], as_index=False)[["shots", "logical_failures"]]
        .sum()
        .sort_values(["method", "epoch"])
    )
    for method in METHODS:
        group = step[step["method"] == method].copy()
        cumulative_rate = group["logical_failures"].cumsum() / group["shots"].cumsum()
        axes[0, 1].plot(
            group["epoch"], cumulative_rate,
            color=colors[method], linewidth=1.8, label=method,
        )
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("Step drift: cumulative logical error rate")
    axes[0, 1].set_ylabel("Cumulative logical error rate")
    axes[0, 1].legend(fontsize=8)

    non_oracle = summary[summary["method"] != "oracle"].copy()
    scenarios = list(non_oracle["scenario"].drop_duplicates())
    methods = ["static", "periodic", "naive", "confidence"]
    x = np.arange(len(scenarios))
    width = 0.19
    for index, method in enumerate(methods):
        values = (
            non_oracle[non_oracle["method"] == method]
            .set_index("scenario")
            .reindex(scenarios)["paired_excess_failures_vs_oracle"]
        )
        axes[1, 0].bar(
            x + (index - 1.5) * width, values, width,
            color=colors[method], label=method,
        )
    axes[1, 0].axhline(0, color="black", linewidth=1)
    axes[1, 0].set_xticks(x, scenarios)
    axes[1, 0].set_title("Paired excess logical failures vs oracle")
    axes[1, 0].set_ylabel("Excess failures")
    axes[1, 0].legend(fontsize=8)

    confidence = summary[summary["method"] == "confidence"].set_index("scenario")
    naive = summary[summary["method"] == "naive"].set_index("scenario")
    periodic = summary[summary["method"] == "periodic"].set_index("scenario")
    table_data = []
    for scenario in scenarios:
        table_data.append(
            [
                scenario,
                f"{confidence.loc[scenario, 'relative_to_oracle']:.3f}",
                int(confidence.loc[scenario, "updates"]),
                f"{naive.loc[scenario, 'relative_to_oracle']:.3f}",
                f"{periodic.loc[scenario, 'relative_to_oracle']:.3f}",
            ]
        )
    axes[1, 1].axis("off")
    table = axes[1, 1].table(
        cellText=table_data,
        colLabels=["scenario", "conf./oracle", "updates", "naive/oracle", "periodic/oracle"],
        cellLoc="center", loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.55)
    axes[1, 1].set_title("Performance and update cost", pad=15)

    for ax in axes.flat[:3]:
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("Epoch" if ax is not axes[1, 0] else "Drift scenario")
    fig.suptitle(
        f"Confidence-aware online MWPM recalibration (d={results['distance'].iloc[0]})",
        fontsize=14,
    )
    fig.savefig(output, dpi=220, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Small smoke-test run")
    parser.add_argument(
        "--scenarios", nargs="+", choices=("step", "ramp", "burst"),
        default=("step", "ramp", "burst"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config()
    if args.quick:
        config = Config(
            distance=3,
            rounds=3,
            epochs=8,
            shots_per_epoch=1_500,
            calibration_shots=2_000,
            calibration_points=9,
            periodic_interval=2,
        )
        scenarios = ("step",)
    else:
        scenarios = tuple(args.scenarios)

    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    calibration = build_calibration_curve(config)
    calibration.to_csv(RESULTS / "calibration_curve.csv", index=False)

    result_frames = []
    adaptation_frames = []
    for scenario_index, scenario in enumerate(scenarios):
        print(f"Running {scenario} scenario...")
        frame, log = run_scenario(
            scenario, scenario_index, config, calibration
        )
        result_frames.append(frame)
        adaptation_frames.append(log)

    results = pd.concat(result_frames, ignore_index=True)
    adaptation = pd.concat(adaptation_frames, ignore_index=True)
    summary = summarize(results, adaptation)
    results.to_csv(RESULTS / "epoch_results.csv", index=False)
    adaptation.to_csv(RESULTS / "adaptation_log.csv", index=False)
    summary.to_csv(RESULTS / "summary.csv", index=False)
    plot_results(results, adaptation, summary, FIGURES / "online_adaptation.png")

    print("\nSummary")
    print(
        summary[
            [
                "scenario", "method", "shots", "logical_failures",
                "logical_error_rate", "relative_to_oracle",
                "paired_excess_failures_vs_oracle", "updates",
                "mcnemar_exact_p_value",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()

