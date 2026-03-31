"""ROC curve generation and analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


@dataclass
class ROCData:
    """Container for ROC curve data.

    Attributes:
        far_values: Array of False Acceptance Rate values.
        frr_values: Array of False Rejection Rate values.
        thresholds: Array of threshold values (optional).
        eer: Equal Error Rate - where FAR ≈ FRR.
        eer_threshold: Threshold value at which EER occurs.
    """

    far_values: np.ndarray
    frr_values: np.ndarray
    thresholds: np.ndarray | None
    eer: float
    eer_threshold: float


def generate_roc(
    far_list: list[float] | np.ndarray,
    frr_list: list[float] | np.ndarray,
    thresholds: list[float] | np.ndarray | None = None,
) -> ROCData:
    """Generate ROC curve data from FAR and FRR values.

    Calculates the Equal Error Rate (EER) by finding the threshold
    where FAR and FRR are closest (minimizes |FAR - FRR|).

    Args:
        far_list: List or array of False Acceptance Rate values.
        frr_list: List or array of False Rejection Rate values.
        thresholds: Optional list or array of threshold values corresponding
            to each FAR/FRR pair.

    Returns:
        ROCData containing FAR/FRR arrays, EER value, and threshold at EER.

    Raises:
        ValueError: If far_list and frr_list have different lengths.

    Example:
        >>> far_values = [0.1, 0.05, 0.02, 0.01, 0.005]
        >>> frr_values = [0.0, 0.1, 0.3, 0.5, 0.7]
        >>> roc = generate_roc(far_values, frr_values)
        >>> print(f"EER: {roc.eer:.3f} at threshold {roc.eer_threshold:.3f}")
    """
    far_values = np.asarray(far_list, dtype=np.float64)
    frr_values = np.asarray(frr_list, dtype=np.float64)

    if len(far_values) != len(frr_values):
        raise ValueError(
            f"far_list and frr_list must have same length: "
            f"got {len(far_values)} and {len(frr_values)}"
        )

    if len(far_values) == 0:
        return ROCData(
            far_values=far_values,
            frr_values=frr_values,
            thresholds=None,
            eer=0.0,
            eer_threshold=0.0,
        )

    thresholds_arr: np.ndarray | None = None
    if thresholds is not None:
        thresholds_arr = np.asarray(thresholds, dtype=np.float64)

    # Find EER: point where |FAR - FRR| is minimized
    diff = np.abs(far_values - frr_values)
    eer_idx = int(np.argmin(diff))
    eer = float((far_values[eer_idx] + frr_values[eer_idx]) / 2.0)

    if thresholds_arr is not None:
        eer_threshold = float(thresholds_arr[eer_idx])
    else:
        eer_threshold = float(eer_idx)

    return ROCData(
        far_values=far_values,
        frr_values=frr_values,
        thresholds=thresholds_arr,
        eer=eer,
        eer_threshold=eer_threshold,
    )


def plot_roc(
    roc_data: ROCData,
    output_path: str | Path | None = None,
    show_plot: bool = False,
    figsize: tuple[float, float] = (8, 6),
) -> Optional[Path]:
    """Plot ROC curve (FAR vs FRR).

    Creates a DET curve-style plot with FAR on x-axis and FRR on y-axis.
    Optionally marks the EER point.

    Args:
        roc_data: ROCData containing curve data to plot.
        output_path: Optional path to save the figure. If provided, saves
            the plot as an image file.
        show_plot: Whether to display the plot interactively (default: False).
        figsize: Figure size as (width, height) in inches.

    Returns:
        Path to saved file if output_path was provided, otherwise None.

    Note:
        If matplotlib is not installed, this function returns None without error.
    """
    if not MATPLOTLIB_AVAILABLE:
        return None

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)

    # Sort by FAR for proper curve visualization
    sort_idx = np.argsort(roc_data.far_values)
    far_sorted = roc_data.far_values[sort_idx]
    frr_sorted = roc_data.frr_values[sort_idx]

    # Plot ROC curve
    ax.plot(far_sorted, frr_sorted, "b-", linewidth=2, label="ROC Curve")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random")

    # Mark EER point
    ax.scatter(
        [roc_data.far_values[np.argmin(np.abs(roc_data.far_values - roc_data.frr_values))]],
        [roc_data.frr_values[np.argmin(np.abs(roc_data.far_values - roc_data.frr_values))]],
        color="red",
        s=100,
        zorder=5,
        label=f"EER = {roc_data.eer:.3f}",
    )

    # Labels and styling
    ax.set_xlabel("False Acceptance Rate (FAR)", fontsize=12)
    ax.set_ylabel("False Rejection Rate (FRR)", fontsize=12)
    ax.set_title("ROC Curve (DET-style)", fontsize=14)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    ax.set_aspect("equal")

    fig.tight_layout()

    result_path: Path | None = None
    if output_path is not None:
        result_path = Path(output_path)
        fig.savefig(result_path, dpi=150, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return result_path
