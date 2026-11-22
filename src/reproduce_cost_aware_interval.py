"""Reproduce the paired t intervals reported for cost-aware policies."""

from pathlib import Path

import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
BLOCKS = ROOT / "results" / "cost_aware_block_summary.csv"


def paired_t_interval(values: pd.Series) -> tuple[float, float, float]:
    """Return sample mean and two-sided 95% Student-t interval."""
    mean = float(values.mean())
    low, high = stats.t.interval(
        0.95,
        len(values) - 1,
        loc=mean,
        scale=stats.sem(values),
    )
    return mean, float(low), float(high)


def main() -> None:
    blocks = pd.read_csv(BLOCKS)
    for policy in ("cost_0_5", "cost_5"):
        mean, low, high = paired_t_interval(
            blocks[f"{policy}_minus_confidence"]
        )
        print(
            f"{policy} minus confidence: mean={mean:.2f}, "
            f"95% t interval=({low:.2f}, {high:.2f}), n={len(blocks)}"
        )


if __name__ == "__main__":
    main()
