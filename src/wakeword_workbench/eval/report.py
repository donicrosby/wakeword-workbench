"""Evaluation report generation."""

from __future__ import annotations

import csv
import io
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .far import calculate_far
from .frr import calculate_frr


@dataclass
class EvaluationReport:
    """Structured evaluation report with all key metrics."""

    far: float
    """False Acceptance Rate (false accepts per hour)."""

    frr: float
    """False Rejection Rate (fraction of wake words missed)."""

    eer: float
    """Equal Error Rate — the point where FAR equals FRR."""

    optimal_threshold: float
    """Threshold where FAR and FRR are most closely balanced."""

    audio_duration_hours: float
    """Total duration of audio evaluated, in hours."""

    total_samples: int
    """Total number of audio samples evaluated."""

    timestamp: str
    """ISO timestamp of when the report was generated."""

    metadata: dict[str, Any]
    """Additional context about the evaluation run."""

    def to_dict(self) -> dict[str, Any]:
        """Convert report to a plain dictionary."""
        return asdict(self)


def _calculate_eer(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    audio_duration_hours: float,
    thresholds: int = 200,
) -> tuple[float, float]:
    """Calculate Equal Error Rate and the optimal threshold.

    EER is the point on the ROC curve where FAR equals FRR.
    Uses threshold sweep and linear interpolation to find the crossing point.

    Args:
        predictions: Array of model confidence scores (0-1) for each frame.
        ground_truth: Binary array where 1 = wake word present, 0 = absent.
        audio_duration_hours: Total duration of audio in hours.
        thresholds: Number of threshold points to evaluate.

    Returns:
        Tuple of (eer, optimal_threshold).
    """
    threshold_values = np.linspace(0.0, 1.0, num=thresholds)

    far_values: list[float] = []
    frr_values: list[float] = []

    for t in threshold_values:
        threshold = float(t)
        far = calculate_far(predictions, None, audio_duration_hours, threshold=threshold)
        frr = calculate_frr(predictions, ground_truth, threshold=threshold)
        far_values.append(far)
        frr_values.append(frr)

    far_array = np.asarray(far_values, dtype=np.float64)
    frr_array = np.asarray(frr_values, dtype=np.float64)

    # Normalize FAR to same scale as FRR for comparison.
    # FRR is in [0,1]; FAR is false accepts/hour which could be any positive value.
    # Normalize FAR so both are on [0,1] for EER calculation.
    max_far = float(far_array.max())
    if max_far > 0:
        norm_far = far_array / max_far
    else:
        norm_far = far_array.copy()

    # Find the index where normalized FAR and FRR cross.
    diff = norm_far - frr_array

    # Find sign changes.
    sign_changes = np.where(np.diff(np.sign(diff)))[0]

    if len(sign_changes) == 0:
        # No crossing found — pick the closest point.
        closest_idx = int(np.argmin(np.abs(diff)))
        return float(frr_array[closest_idx]), float(threshold_values[closest_idx])

    idx = int(sign_changes[0])

    # Linear interpolation between idx and idx+1.
    t0 = threshold_values[idx]
    t1 = threshold_values[idx + 1]
    d0 = diff[idx]
    d1 = diff[idx + 1]

    # Interpolate the threshold where diff == 0.
    if d1 != d0:
        optimal_threshold = t0 - d0 * (t1 - t0) / (d1 - d0)
    else:
        optimal_threshold = (t0 + t1) / 2

    optimal_threshold = float(np.clip(optimal_threshold, 0.0, 1.0))

    # Compute EER at that threshold.
    far_at_eer = calculate_far(predictions, None, audio_duration_hours, threshold=optimal_threshold)
    frr_at_eer = calculate_frr(predictions, ground_truth, threshold=optimal_threshold)

    # EER is the average of normalized FAR and FRR at the crossing point.
    if max_far > 0:
        eer = (far_at_eer / max_far + frr_at_eer) / 2.0
    else:
        eer = frr_at_eer

    return float(eer), optimal_threshold


def _format_json(report: EvaluationReport) -> str:
    """Format report as pretty-printed JSON."""
    return json.dumps(report.to_dict(), indent=2, default=str)


def _format_csv(report: EvaluationReport) -> str:
    """Format report as CSV with a header row and a single data row."""
    meta_rows: list[list[str]] = []
    for key, value in sorted(report.metadata.items()):
        meta_rows.append([str(key), str(value)])

    rows: list[list[str]] = [
        ["far", str(report.far)],
        ["frr", str(report.frr)],
        ["eer", str(report.eer)],
        ["optimal_threshold", str(report.optimal_threshold)],
        ["audio_duration_hours", str(report.audio_duration_hours)],
        ["total_samples", str(report.total_samples)],
        ["timestamp", str(report.timestamp)],
    ]
    if meta_rows:
        rows.append([])
        rows.append(["metadata_key", "metadata_value"])
        rows.extend(meta_rows)

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerows(rows)
    return output.getvalue()


def _format_markdown(report: EvaluationReport) -> str:
    """Format report as a Markdown table with a summary section."""
    lines: list[str] = [
        "# Evaluation Report",
        "",
        f"**Generated:** {report.timestamp}",
        "",
        "## Metrics",
        "",
        "| Metric               | Value                     |",
        "|----------------------|---------------------------|",
        f"| FAR (false accepts/hr) | {report.far:.4f}              |",
        f"| FRR                   | {report.frr:.4f}              |",
        f"| EER                   | {report.eer:.4f}              |",
        f"| Optimal Threshold     | {report.optimal_threshold:.4f}              |",
        f"| Audio Duration (hrs)  | {report.audio_duration_hours:.4f}              |",
        f"| Total Samples          | {report.total_samples}                   |",
    ]

    if report.metadata:
        lines.append("")
        lines.append("## Metadata")
        lines.append("")
        for key, value in sorted(report.metadata.items()):
            lines.append(f"- **{key}:** {value}")

    return "\n".join(lines)


def generate_report(
    results: dict[str, Any],
    format: str = "json",
    output_path: Path | str | None = None,
) -> str | Path:
    """Generate an evaluation report from results.

    Accepts a results dictionary containing at minimum:
        - predictions: np.ndarray of model confidence scores
        - ground_truth: np.ndarray of binary labels (1 = wake word present)
        - audio_duration_hours: float

    Args:
        results: Dictionary with prediction/ground truth data and optional metadata.
        format: Output format — one of 'json', 'csv', or 'markdown'.
        output_path: Optional path to write the report to.
            If provided, returns the Path instead of the string.

    Returns:
        Formatted report as a string, or the output_path if written to disk.

    Raises:
        ValueError: If format is not one of 'json', 'csv', 'markdown',
            or if required keys are missing from results.
    """
    valid_formats = {"json", "csv", "markdown"}
    if format not in valid_formats:
        raise ValueError(
            f"Invalid format '{format}'. Must be one of: {', '.join(sorted(valid_formats))}"
        )

    required_keys = {"predictions", "ground_truth", "audio_duration_hours"}
    missing = required_keys - set(results.keys())
    if missing:
        raise ValueError(f"results missing required keys: {', '.join(sorted(missing))}")

    predictions: np.ndarray = np.asarray(results["predictions"])
    ground_truth: np.ndarray = np.asarray(results["ground_truth"])
    audio_duration_hours: float = float(results["audio_duration_hours"])
    metadata: dict[str, Any] = dict(results.get("metadata", {}))

    if "total_samples" in results:
        total_samples = int(results["total_samples"])
    else:
        total_samples = len(predictions)

    far = float(calculate_far(predictions, None, audio_duration_hours))
    frr = float(calculate_frr(predictions, ground_truth))

    eer, optimal_threshold = _calculate_eer(predictions, ground_truth, audio_duration_hours)

    timestamp = datetime.now(UTC).isoformat()

    report = EvaluationReport(
        far=far,
        frr=frr,
        eer=eer,
        optimal_threshold=optimal_threshold,
        audio_duration_hours=audio_duration_hours,
        total_samples=total_samples,
        timestamp=timestamp,
        metadata=metadata,
    )

    if format == "json":
        content = _format_json(report)
    elif format == "csv":
        content = _format_csv(report)
    else:  # markdown
        content = _format_markdown(report)

    if output_path is not None:
        path = Path(output_path)
        path.write_text(content, encoding="utf-8")
        return path

    return content
