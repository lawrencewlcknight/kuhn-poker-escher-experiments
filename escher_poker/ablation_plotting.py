"""Plotting helpers for multi-arm ESCHER ablation experiments."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np

from .constants import EXPLOITABILITY_THRESHOLD
from .experiment_utils import safe_stats


def _variant_label(variant_id: str, variants: List[Dict[str, Any]]) -> str:
    for variant in variants:
        if variant["variant_id"] == variant_id:
            return variant["label"]
    return variant_id


def _ordered_variant_ids(variants: List[Dict[str, Any]]) -> List[str]:
    return [variant["variant_id"] for variant in variants]


def _metric_stats_by_variant(
    summary_rows: List[Dict[str, Any]],
    variants: List[Dict[str, Any]],
    metric: str,
) -> List[Dict[str, Any]]:
    rows = []
    order = _ordered_variant_ids(variants)
    for variant_id in order:
        values = [
            row.get(metric, np.nan)
            for row in summary_rows
            if row["variant_id"] == variant_id
        ]
        if values:
            rows.append({
                "variant_id": variant_id,
                "label": _variant_label(variant_id, variants),
                **safe_stats(values),
            })
    return rows


def plot_mean_bar(
    summary_rows: List[Dict[str, Any]],
    variants: List[Dict[str, Any]],
    metric: str,
    ylabel: str,
    title: str,
    output_path: str | Path,
) -> None:
    """Plot variant means with standard-error bars."""
    rows = _metric_stats_by_variant(summary_rows, variants, metric)
    if not rows:
        return

    labels = [row["label"] for row in rows]
    means = [row["mean"] for row in rows]
    errors = [0.0 if not np.isfinite(row["se"]) else row["se"] for row in rows]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(labels, means, yerr=errors, capsize=4)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Variant")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_paired_delta(
    paired_rows: List[Dict[str, Any]],
    variants: List[Dict[str, Any]],
    metric: str,
    ylabel: str,
    title: str,
    output_path: str | Path,
) -> None:
    """Plot paired deltas versus the reference variant."""
    if not paired_rows:
        return
    rows = []
    order = _ordered_variant_ids(variants)
    for variant_id in order:
        values = [row.get(metric, np.nan) for row in paired_rows if row["variant_id"] == variant_id]
        if values:
            rows.append({
                "variant_id": variant_id,
                "label": _variant_label(variant_id, variants),
                **safe_stats(values),
            })
    if not rows:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(
        [row["label"] for row in rows],
        [row["mean"] for row in rows],
        yerr=[0.0 if not np.isfinite(row["se"]) else row["se"] for row in rows],
        capsize=4,
    )
    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Variant")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _mean_curve_rows(curve_rows: Iterable[Dict[str, Any]]) -> List[Dict[str, float]]:
    grouped = defaultdict(list)
    for row in curve_rows:
        grouped[int(row["iteration"])].append(row)

    mean_rows = []
    for iteration in sorted(grouped):
        rows = grouped[iteration]
        exploitability_values = [row["exploitability"] for row in rows]
        value_error_values = [row["policy_value_error"] for row in rows]
        mean_rows.append({
            "iteration": float(iteration),
            "nodes_touched": float(np.mean([row["nodes_touched"] for row in rows])),
            "exploitability_mean": safe_stats(exploitability_values)["mean"],
            "exploitability_se": safe_stats(exploitability_values)["se"],
            "policy_value_error_mean": safe_stats(value_error_values)["mean"],
            "policy_value_error_se": safe_stats(value_error_values)["se"],
        })
    return mean_rows


def plot_reference_intermediate_curves(
    curve_rows: List[Dict[str, Any]],
    reference_variant_id: str,
    run_dir: str | Path,
) -> None:
    """Plot intermediate exploitability/value-error curves for the reference arm."""
    run_dir = Path(run_dir)
    reference_rows = [
        row
        for row in curve_rows
        if (
            row["variant_id"] == reference_variant_id
            and not row.get("is_final_policy_evaluation", False)
        )
    ]
    grouped = _mean_curve_rows(reference_rows)
    if not grouped:
        return

    iterations = np.asarray([row["iteration"] for row in grouped], dtype=float)

    fig, ax = plt.subplots(figsize=(9, 5))
    means = np.asarray([row["exploitability_mean"] for row in grouped], dtype=float)
    errors = np.asarray([row["exploitability_se"] for row in grouped], dtype=float)
    ax.plot(iterations, means, marker="o", label="Mean exploitability")
    ax.fill_between(iterations, means - errors, means + errors, alpha=0.2, label="Mean +/- s.e.")
    ax.axhline(
        EXPLOITABILITY_THRESHOLD,
        linestyle="--",
        linewidth=1,
        label="Exploitability threshold",
    )
    ax.set_xlabel("ESCHER iteration")
    ax.set_ylabel("Exploitability")
    ax.set_title("ESCHER baseline: intermediate exploitability curve")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        run_dir / "baseline_intermediate_exploitability_curve.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    means = np.asarray([row["policy_value_error_mean"] for row in grouped], dtype=float)
    errors = np.asarray([row["policy_value_error_se"] for row in grouped], dtype=float)
    ax.plot(iterations, means, marker="o", label="Mean value error")
    ax.fill_between(iterations, means - errors, means + errors, alpha=0.2, label="Mean +/- s.e.")
    ax.set_xlabel("ESCHER iteration")
    ax.set_ylabel(r"Absolute error from $-1/18$")
    ax.set_title("ESCHER baseline: intermediate policy-value error")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        run_dir / "baseline_intermediate_policy_value_error_curve.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_policy_training_ablation(
    summary_rows: List[Dict[str, Any]],
    curve_rows: List[Dict[str, Any]],
    paired_rows: List[Dict[str, Any]],
    variants: List[Dict[str, Any]],
    reference_variant_id: str,
    run_dir: str | Path,
) -> None:
    """Create the standard thesis plots for policy-training ablations."""
    run_dir = Path(run_dir)
    plot_mean_bar(
        summary_rows,
        variants,
        "final_exploitability",
        "Mean final exploitability",
        "ESCHER: final exploitability by policy-training regime",
        run_dir / "final_exploitability_by_variant.png",
    )
    plot_mean_bar(
        summary_rows,
        variants,
        "final_policy_value_error",
        "Mean final policy-value error",
        "ESCHER: final value error by policy-training regime",
        run_dir / "final_policy_value_error_by_variant.png",
    )
    plot_mean_bar(
        summary_rows,
        variants,
        "elapsed_seconds",
        "Mean wall-clock seconds",
        "ESCHER: runtime by policy-training regime",
        run_dir / "runtime_by_variant.png",
    )
    plot_mean_bar(
        summary_rows,
        variants,
        "policy_gradient_steps_expected",
        "Expected policy-gradient steps",
        "ESCHER: average-policy training budget by variant",
        run_dir / "policy_gradient_budget_by_variant.png",
    )
    plot_paired_delta(
        paired_rows,
        variants,
        "delta_final_exploitability_vs_baseline",
        "Delta final exploitability vs ESCHER baseline",
        "ESCHER: paired final-exploitability difference versus baseline",
        run_dir / "paired_delta_final_exploitability_vs_baseline.png",
    )
    plot_reference_intermediate_curves(curve_rows, reference_variant_id, run_dir)
