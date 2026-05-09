"""Plotting helpers for ESCHER checkpoint-stability experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np


def _matrix_to_arrays(matrix: Dict[int, Dict[int, float]]):
    rows = sorted(matrix)
    cols = sorted({col for row in matrix.values() for col in row})
    values = np.asarray([[matrix[row].get(col, np.nan) for col in cols] for row in rows], dtype=float)
    return rows, cols, values


def plot_heatmap(
    matrix: Dict[int, Dict[int, float]],
    title: str,
    output_path: str | Path,
    *,
    cmap: str = "coolwarm",
    center_zero: bool = True,
    annotate: bool = True,
    colorbar_label: str = "Seat-averaged EV of row checkpoint vs column checkpoint",
) -> None:
    """Plot a checkpoint-by-checkpoint matrix."""
    row_labels, col_labels, values = _matrix_to_arrays(matrix)
    finite = values[np.isfinite(values)]
    vmax = float(np.max(np.abs(finite))) if center_zero and finite.size else 1.0
    if vmax == 0:
        vmax = 1.0

    fig, ax = plt.subplots(figsize=(9, 7))
    if center_zero:
        image = ax.imshow(values, cmap=cmap, vmin=-vmax, vmax=vmax)
    else:
        image = ax.imshow(values, cmap=cmap)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("Column checkpoint")
    ax.set_ylabel("Row checkpoint")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label=colorbar_label)

    if annotate:
        for row_idx in range(values.shape[0]):
            for col_idx in range(values.shape[1]):
                value = values[row_idx, col_idx]
                if np.isfinite(value):
                    color = "white" if center_zero and abs(value) > 0.5 * vmax else "black"
                    ax.text(
                        col_idx,
                        row_idx,
                        f"{value:.3f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color=color,
                    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_checkpoint_training_summary(
    checkpoint_rows: List[Dict[str, Any]],
    baseline_rows: List[Dict[str, Any]],
    final_iteration: int,
    run_dir: str | Path,
) -> None:
    """Plot checkpoint exploitability and final checkpoint-vs-baseline deltas."""
    run_dir = Path(run_dir)
    if checkpoint_rows:
        checkpoints = sorted({row["checkpoint_iteration"] for row in checkpoint_rows})
        means = []
        sems = []
        for checkpoint in checkpoints:
            values = np.asarray([
                row["exploitability_recomputed"]
                for row in checkpoint_rows
                if row["checkpoint_iteration"] == checkpoint
            ], dtype=float)
            means.append(float(np.mean(values)))
            sems.append(float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.errorbar(checkpoints, means, yerr=sems, marker="o", capsize=3, label="Checkpointed arm")
        if baseline_rows:
            base_values = np.asarray([row["exploitability_recomputed"] for row in baseline_rows], dtype=float)
            base_mean = float(np.mean(base_values))
            base_sem = (
                float(np.std(base_values, ddof=1) / np.sqrt(len(base_values)))
                if len(base_values) > 1
                else 0.0
            )
            ax.axhline(base_mean, linestyle="--", linewidth=1.5, label="Continuous baseline mean")
            ax.fill_between(
                [min(checkpoints), max(checkpoints)],
                base_mean - base_sem,
                base_mean + base_sem,
                alpha=0.15,
                label="Continuous baseline +/- s.e.",
            )
        ax.set_xlabel("Checkpoint iteration")
        ax.set_ylabel("Exploitability (NashConv/2)")
        ax.set_title("Kuhn Poker ESCHER: Checkpoint Exploitability")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            run_dir / "checkpoint_exploitability_with_continuous_baseline.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)

    if checkpoint_rows and baseline_rows:
        final_rows = {
            row["seed"]: row
            for row in checkpoint_rows
            if row["checkpoint_iteration"] == final_iteration
        }
        baseline_by_seed = {row["seed"]: row for row in baseline_rows}
        paired = []
        for seed in sorted(set(final_rows) & set(baseline_by_seed)):
            paired.append({
                "seed": seed,
                "delta": (
                    final_rows[seed]["exploitability_recomputed"]
                    - baseline_by_seed[seed]["exploitability_recomputed"]
                ),
            })
        if paired:
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.axhline(0.0, linestyle="--", linewidth=1)
            ax.bar([str(row["seed"]) for row in paired], [row["delta"] for row in paired])
            ax.set_xlabel("Seed")
            ax.set_ylabel("Delta exploitability\n(checkpointed final - continuous baseline)")
            ax.set_title("Final Checkpointed ESCHER versus Continuous Baseline")
            ax.tick_params(axis="x", rotation=45)
            ax.grid(True, axis="y", alpha=0.3)
            fig.tight_layout()
            fig.savefig(
                run_dir / "checkpointed_final_minus_continuous_baseline_exploitability.png",
                dpi=200,
                bbox_inches="tight",
            )
            plt.close(fig)


def plot_checkpoint_head_to_head_outputs(
    analysis: Dict[str, Any],
    final_iteration: int,
    equivalence_epsilon: float,
    run_dir: str | Path,
    *,
    annotate_heatmap: bool = True,
) -> None:
    """Create thesis plots for exact checkpoint head-to-head analysis."""
    run_dir = Path(run_dir)
    mean_matrix = analysis["matrix_values"]["mean_matrix"]
    win_fraction_matrix = analysis["matrix_values"]["win_fraction_matrix"]
    plot_heatmap(
        mean_matrix,
        "ESCHER exact head-to-head matrix across checkpoints\nMean seat-averaged EV across seeds",
        run_dir / "head_to_head_exact_mean_matrix.png",
        annotate=annotate_heatmap,
    )

    later_vs_earlier = {
        row: {
            col: (value if row > col else np.nan)
            for col, value in cols.items()
        }
        for row, cols in mean_matrix.items()
    }
    plot_heatmap(
        later_vs_earlier,
        "Later-vs-earlier ESCHER checkpoint EV\nPositive cells support monotonic improvement",
        run_dir / "head_to_head_later_vs_earlier_matrix.png",
        annotate=annotate_heatmap,
    )
    plot_heatmap(
        win_fraction_matrix,
        "Fraction of seeds where row ESCHER checkpoint clearly beats column checkpoint",
        run_dir / "head_to_head_seed_win_fraction_matrix.png",
        cmap="viridis",
        center_zero=False,
        annotate=annotate_heatmap,
        colorbar_label=f"Fraction with EV > {equivalence_epsilon}",
    )

    aggregate = analysis["head_to_head_aggregate_strength_summary"]
    if aggregate:
        x = np.asarray([row["checkpoint"] for row in aggregate], dtype=float)

        fig, ax = plt.subplots(figsize=(9, 5))
        y = np.asarray([row["mean_EV_vs_earlier_mean"] for row in aggregate], dtype=float)
        yerr = np.asarray([row["mean_EV_vs_earlier_sem"] for row in aggregate], dtype=float)
        ax.errorbar(x, y, yerr=yerr, marker="o", capsize=3)
        ax.axhline(0.0, linestyle="--", linewidth=1)
        ax.set_xlabel("Checkpoint iteration")
        ax.set_ylabel("Mean EV vs earlier checkpoints")
        ax.set_title("Does later ESCHER training improve head-to-head performance?")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(run_dir / "head_to_head_strength_vs_earlier_aggregate.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 5))
        y = np.asarray([row["EV_vs_previous_mean"] for row in aggregate], dtype=float)
        yerr = np.asarray([row["EV_vs_previous_sem"] for row in aggregate], dtype=float)
        ax.errorbar(x, y, yerr=yerr, marker="o", capsize=3)
        ax.axhline(0.0, linestyle="--", linewidth=1)
        ax.set_xlabel("Checkpoint iteration")
        ax.set_ylabel("EV vs immediately previous checkpoint")
        ax.set_title("Adjacent ESCHER checkpoint improvement")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(
            run_dir / "head_to_head_vs_previous_checkpoint_aggregate.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 5))
        y = np.asarray([row["exploitability_mean"] for row in aggregate], dtype=float)
        yerr = np.asarray([row["exploitability_sem"] for row in aggregate], dtype=float)
        ax.errorbar(x, y, yerr=yerr, marker="o", capsize=3, label="Checkpointed arm")
        baseline_rows = analysis["final_checkpoint_vs_continuous_baseline"]
        if baseline_rows:
            base_values = np.asarray(
                [row["continuous_baseline_exploitability"] for row in baseline_rows],
                dtype=float,
            )
            base_mean = float(np.mean(base_values))
            base_sem = (
                float(np.std(base_values, ddof=1) / np.sqrt(len(base_values)))
                if len(base_values) > 1
                else 0.0
            )
            ax.axhline(base_mean, linestyle="--", linewidth=1.5, label="Continuous baseline mean")
            ax.fill_between([x.min(), x.max()], base_mean - base_sem, base_mean + base_sem, alpha=0.15)
        ax.set_xlabel("Checkpoint iteration")
        ax.set_ylabel("Exploitability")
        ax.set_title("ESCHER checkpoint exploitability over training")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(run_dir / "checkpoint_exploitability_aggregate.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    strength_rows = analysis["head_to_head_strength_with_metrics"]
    if strength_rows:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(
            [row["exploitability"] for row in strength_rows],
            [row["mean_EV_vs_all_other_checkpoints"] for row in strength_rows],
        )
        ax.axhline(0.0, linestyle="--", linewidth=1)
        ax.set_xlabel("Exploitability")
        ax.set_ylabel("Mean EV vs all other checkpoints")
        ax.set_title("ESCHER equilibrium quality versus head-to-head strength")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(run_dir / "exploitability_vs_head_to_head_strength.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    baseline_rows = analysis["final_checkpoint_vs_continuous_baseline"]
    if baseline_rows:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.axhline(0.0, linestyle="--", linewidth=1)
        ax.bar(
            [str(row["seed"]) for row in baseline_rows],
            [row["delta_exploitability_checkpointed_minus_baseline"] for row in baseline_rows],
        )
        ax.set_xlabel("Seed")
        ax.set_ylabel("Delta exploitability\n(final checkpointed - continuous baseline)")
        ax.set_title("Final ESCHER checkpoint versus continuous baseline")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(
            run_dir / "final_checkpoint_minus_continuous_baseline_exploitability.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)
