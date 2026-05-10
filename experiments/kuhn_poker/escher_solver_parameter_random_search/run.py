"""CLI entry point for the ESCHER solver-parameter random search."""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Optional

# Keep execution CPU-only by default for reproducibility unless explicitly overridden.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")
os.environ.setdefault("XDG_CACHE_HOME", str((Path("outputs") / ".cache").resolve()))
os.environ.setdefault("MPLCONFIGDIR", str((Path("outputs") / ".matplotlib_cache").resolve()))
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pyspiel  # noqa: E402
import tensorflow as tf  # noqa: E402
from tqdm import tqdm  # noqa: E402

from escher_poker.constants import EXPLOITABILITY_THRESHOLD, KUHN_GAME_VALUE_PLAYER_0  # noqa: E402
from escher_poker.experiment_utils import create_run_dir, json_safe  # noqa: E402
from escher_poker.hyperparameter_search import (  # noqa: E402
    aggregate_summaries,
    paired_differences_vs_baseline,
    run_single_hyperparameter_seed,
    select_confirmation_variants,
    with_stage_overrides,
)

from .config import (  # noqa: E402
    BASELINE_VARIANT_ID,
    CONFIRMATION_EVALUATION_INTERVAL,
    CONFIRMATION_ITERATIONS,
    CONFIRMATION_SEEDS,
    CONFIRMATION_TOP_K,
    DEFAULT_CONFIG,
    N_RANDOM_CANDIDATES,
    RANDOM_SEARCH_SEED,
    SCREENING_EVALUATION_INTERVAL,
    SCREENING_ITERATIONS,
    SCREENING_SEEDS,
    build_screening_configs,
    parse_seeds,
)

_LOGGER = logging.getLogger("escher_poker.experiment")


def parse_int_tuple(value: Optional[str]):
    if value is None:
        return None
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _str2bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"true", "t", "yes", "y", "1"}:
        return True
    if lowered in {"false", "f", "no", "n", "0"}:
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {value!r}")


def build_base_config(args) -> dict:
    config = deepcopy(DEFAULT_CONFIG)
    overrides = {
        "num_traversals": args.traversals,
        "num_val_fn_traversals": args.value_traversals,
        "learning_rate": args.learning_rate,
        "experiment_name": args.experiment_name,
        "memory_capacity": args.memory_capacity,
        "batch_size_regret": args.batch_size_regret,
        "batch_size_value": args.batch_size_value,
        "batch_size_average_policy": args.batch_size_average_policy,
        "policy_network_train_steps": args.policy_network_train_steps,
        "regret_network_train_steps": args.regret_network_train_steps,
        "value_network_train_steps": args.value_network_train_steps,
        "reinitialize_regret_networks": args.reinitialize_regret_networks,
        "reinitialize_value_network": args.reinitialize_value_network,
        "policy_network_layers": parse_int_tuple(args.policy_network_layers),
        "regret_network_layers": parse_int_tuple(args.regret_network_layers),
        "value_network_layers": parse_int_tuple(args.value_network_layers),
        "expl": args.expl,
        "val_expl": args.val_expl,
        "importance_sampling": args.importance_sampling,
        "importance_sampling_threshold": args.importance_sampling_threshold,
        "clear_value_buffer": args.clear_value_buffer,
        "val_bootstrap": args.val_bootstrap,
        "use_balanced_probs": args.use_balanced_probs,
        "val_op_prob": args.val_op_prob,
        "all_actions": args.all_actions,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    return config


def explicit_config_overrides(args) -> dict:
    """Return CLI overrides that should also constrain sampled candidates."""
    overrides = {
        "num_traversals": args.traversals,
        "num_val_fn_traversals": args.value_traversals,
        "learning_rate": args.learning_rate,
        "memory_capacity": args.memory_capacity,
        "batch_size_regret": args.batch_size_regret,
        "batch_size_value": args.batch_size_value,
        "batch_size_average_policy": args.batch_size_average_policy,
        "policy_network_train_steps": args.policy_network_train_steps,
        "regret_network_train_steps": args.regret_network_train_steps,
        "value_network_train_steps": args.value_network_train_steps,
        "reinitialize_regret_networks": args.reinitialize_regret_networks,
        "reinitialize_value_network": args.reinitialize_value_network,
        "policy_network_layers": parse_int_tuple(args.policy_network_layers),
        "regret_network_layers": parse_int_tuple(args.regret_network_layers),
        "value_network_layers": parse_int_tuple(args.value_network_layers),
        "expl": args.expl,
        "val_expl": args.val_expl,
        "importance_sampling": args.importance_sampling,
        "importance_sampling_threshold": args.importance_sampling_threshold,
        "clear_value_buffer": args.clear_value_buffer,
        "val_bootstrap": args.val_bootstrap,
        "use_balanced_probs": args.use_balanced_probs,
        "val_op_prob": args.val_op_prob,
        "all_actions": args.all_actions,
    }
    return {key: value for key, value in overrides.items() if value is not None}


def _configure_logging(run_dir: Path, verbose: bool) -> None:
    log_level = logging.DEBUG if verbose else logging.INFO
    log_format = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    logging.basicConfig(level=log_level, format=log_format, stream=sys.stdout)
    file_handler = logging.FileHandler(run_dir / "experiment.log", encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(file_handler)


def _ordered_fieldnames(rows: List[Dict[str, Any]], preferred: Iterable[str]) -> List[str]:
    fields = []
    for field in preferred:
        if any(field in row for row in rows):
            fields.append(field)
    extras = sorted({key for row in rows for key in row.keys()} - set(fields))
    return fields + extras


def _write_csv(path: Path, rows: List[Dict[str, Any]], preferred_fields: Iterable[str]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = _ordered_fieldnames(rows, preferred_fields)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(json_safe(rows))


def _run_stage(
    configs: List[Dict[str, Any]],
    seeds: List[int],
    stage_name: str,
    num_iterations: int,
    evaluation_interval: int,
) -> List[Dict[str, Any]]:
    stage_results = []
    for config in tqdm(configs, desc=f"{stage_name} variants"):
        stage_config = with_stage_overrides(
            config,
            stage_name,
            num_iterations,
            evaluation_interval,
        )
        for seed in tqdm(seeds, desc=stage_config["variant_id"], leave=False):
            _LOGGER.info("Running %s seed=%s stage=%s", stage_config["variant_id"], seed, stage_name)
            stage_results.append(run_single_hyperparameter_seed(seed, stage_config, stage_name))
    return stage_results


def _curve_rows(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for result in results:
        diagnostics = result["diagnostics"]
        for idx, iteration in enumerate(result["iterations"]):
            row = {
                "stage": result["stage"],
                "variant_id": result["variant_id"],
                "seed": result["seed"],
                "iteration": int(iteration),
                "nodes_touched": (
                    float(result["nodes_touched"][idx])
                    if idx < len(result["nodes_touched"])
                    else np.nan
                ),
                "wall_clock_seconds": (
                    float(result["wall_clock_seconds"][idx])
                    if idx < len(result["wall_clock_seconds"])
                    else np.nan
                ),
                "exploitability": float(result["exploitability"][idx]),
                "average_policy_value": (
                    float(result["average_policy_value"][idx])
                    if idx < len(result["average_policy_value"])
                    else np.nan
                ),
                "policy_value_error": (
                    float(result["policy_value_error"][idx])
                    if idx < len(result["policy_value_error"])
                    else np.nan
                ),
            }
            for key in [
                "policy_loss",
                "value_loss",
                "value_test_loss",
                "regret_loss_player_0",
                "regret_loss_player_1",
                "average_policy_buffer_size",
                "regret_buffer_size_player_0",
                "regret_buffer_size_player_1",
                "value_buffer_size",
                "value_test_buffer_size",
            ]:
                arr = diagnostics.get(key)
                if arr is not None and len(arr) > idx:
                    row[key] = float(arr[idx])
            rows.append(row)
    return rows


def _aggregate_delta_summaries(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    metrics = [
        "delta_final_exploitability",
        "delta_best_exploitability",
        "delta_final_window_mean_exploitability",
        "delta_exploitability_auc",
        "delta_policy_value_error",
        "delta_wall_clock_seconds",
        "delta_nodes_touched",
    ]
    aggregate_rows = []
    for variant_id in sorted({row["variant_id"] for row in rows}):
        variant_rows = [row for row in rows if row["variant_id"] == variant_id]
        aggregate = {"variant_id": variant_id, "n_runs": len(variant_rows)}
        for metric in metrics:
            values = np.asarray([row.get(metric, np.nan) for row in variant_rows], dtype=float)
            finite = values[np.isfinite(values)]
            aggregate[f"{metric}_mean"] = float(np.mean(finite)) if finite.size else np.nan
            aggregate[f"{metric}_std"] = (
                float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
            )
            aggregate[f"{metric}_se"] = (
                float(np.std(finite, ddof=1) / np.sqrt(finite.size))
                if finite.size > 1
                else 0.0
            )
            aggregate[f"{metric}_n_finite"] = int(finite.size)
        aggregate_rows.append(aggregate)
    return aggregate_rows


def _group_stage_results(results: List[Dict[str, Any]], stage_name: str):
    grouped = {}
    for result in results:
        if result["stage"] != stage_name:
            continue
        grouped.setdefault(result["variant_id"], []).append(result)
    return grouped


def _plot_stage_curves(results: List[Dict[str, Any]], stage_name: str, run_dir: Path) -> None:
    grouped = _group_stage_results(results, stage_name)
    if not grouped:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for variant_id, variant_results in grouped.items():
        xs = variant_results[0]["iterations"]
        ymat = np.vstack([result["exploitability"] for result in variant_results])
        if np.all(~np.isfinite(ymat)):
            continue
        mean = np.nanmean(ymat, axis=0)
        ax.plot(xs, mean, linewidth=2, label=variant_id)
    ax.axhline(EXPLOITABILITY_THRESHOLD, linestyle="--", label="Exploitability threshold")
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Exploitability (NashConv/2)")
    ax.set_title(f"ESCHER Solver-Parameter Random Search: {stage_name.title()} Exploitability")
    ax.grid(True)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(run_dir / f"{stage_name}_exploitability_by_iteration.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for variant_id, variant_results in grouped.items():
        xmat = np.vstack([result["nodes_touched"] for result in variant_results])
        ymat = np.vstack([result["exploitability"] for result in variant_results])
        if np.all(~np.isfinite(xmat)) or np.all(~np.isfinite(ymat)):
            continue
        ax.plot(np.nanmean(xmat, axis=0), np.nanmean(ymat, axis=0), linewidth=2, label=variant_id)
    ax.axhline(EXPLOITABILITY_THRESHOLD, linestyle="--", label="Exploitability threshold")
    ax.set_xlabel("Nodes touched")
    ax.set_ylabel("Exploitability (NashConv/2)")
    ax.set_title(f"ESCHER Solver-Parameter Random Search: {stage_name.title()} Exploitability by Nodes")
    ax.grid(True)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(run_dir / f"{stage_name}_exploitability_by_nodes.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_final_metric_bars(
    summary_rows: List[Dict[str, Any]],
    stage_name: str,
    metric: str,
    ylabel: str,
    filename: str,
    run_dir: Path,
) -> None:
    rows = [row for row in summary_rows if row["stage"] == stage_name]
    if not rows:
        return
    aggregate = aggregate_summaries(rows)
    labels = [row["variant_id"] for row in aggregate]
    means = [row[f"{metric}_mean"] for row in aggregate]
    ses = [row[f"{metric}_se"] for row in aggregate]
    fig, ax = plt.subplots(figsize=(max(8, 0.8 * len(labels)), 5))
    ax.bar(labels, means, yerr=ses, capsize=3)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Variant")
    ax.set_title(f"ESCHER Solver-Parameter Random Search: {stage_name.title()} {ylabel}")
    ax.grid(True, axis="y")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(run_dir / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_paired_delta(paired_rows: List[Dict[str, Any]], run_dir: Path) -> None:
    if not paired_rows:
        return
    variant_ids = sorted({row["variant_id"] for row in paired_rows})
    positions = np.arange(len(variant_ids))
    means = []
    ses = []
    for variant_id in variant_ids:
        values = np.asarray([
            row.get("delta_final_exploitability", np.nan)
            for row in paired_rows
            if row["variant_id"] == variant_id
        ], dtype=float)
        finite = values[np.isfinite(values)]
        means.append(float(np.mean(finite)) if finite.size else np.nan)
        ses.append(
            float(np.std(finite, ddof=1) / np.sqrt(finite.size))
            if finite.size > 1
            else 0.0
        )
    fig, ax = plt.subplots(figsize=(max(8, 0.8 * len(variant_ids)), 5))
    ax.bar(positions, means, yerr=ses, capsize=3)
    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_xticks(positions)
    ax.set_xticklabels(variant_ids, rotation=45, ha="right")
    ax.set_ylabel("Delta final exploitability vs baseline")
    ax.set_title("ESCHER Solver-Parameter Random Search: Confirmation Paired Deltas")
    ax.grid(True, axis="y")
    fig.tight_layout()
    fig.savefig(run_dir / "confirmation_paired_delta_final_exploitability.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_confirmation_diagnostics(
    confirmation_results: List[Dict[str, Any]],
    run_dir: Path,
) -> None:
    grouped = _group_stage_results(confirmation_results, "confirmation")
    if not grouped:
        return
    for diag_key, ylabel, filename in [
        ("policy_loss", "Policy MSE loss", "confirmation_policy_loss_diagnostic.png"),
        ("value_loss", "Value-network train MSE loss", "confirmation_value_loss_diagnostic.png"),
        ("value_test_loss", "Value-network test MSE loss", "confirmation_value_test_loss_diagnostic.png"),
        ("regret_loss_player_0", "Regret-network P0 MSE loss", "confirmation_regret_loss_p0_diagnostic.png"),
        ("regret_loss_player_1", "Regret-network P1 MSE loss", "confirmation_regret_loss_p1_diagnostic.png"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        plotted = False
        for variant_id, variant_results in grouped.items():
            curves = []
            for result in variant_results:
                arr = result["diagnostics"].get(diag_key)
                if arr is not None and len(arr):
                    curves.append(np.asarray(arr, dtype=float))
            if not curves:
                continue
            min_len = min(len(curve) for curve in curves)
            ymat = np.vstack([curve[:min_len] for curve in curves])
            xs = np.asarray(variant_results[0]["iterations"][:min_len], dtype=float)
            mean = np.nanmean(ymat, axis=0)
            se = (
                np.nanstd(ymat, axis=0, ddof=1) / np.sqrt(ymat.shape[0])
                if ymat.shape[0] > 1
                else np.zeros_like(mean)
            )
            ax.plot(xs, mean, linewidth=2, label=variant_id)
            ax.fill_between(xs, mean - se, mean + se, alpha=0.15)
            plotted = True
        if not plotted:
            plt.close(fig)
            continue
        ax.set_xlabel("Training iteration")
        ax.set_ylabel(ylabel)
        ax.set_title(f"ESCHER Solver-Parameter Random Search: {ylabel}")
        ax.grid(True)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(run_dir / filename, dpi=200, bbox_inches="tight")
        plt.close(fig)


def _plot_outputs(
    screening_results: List[Dict[str, Any]],
    confirmation_results: List[Dict[str, Any]],
    screening_summary_rows: List[Dict[str, Any]],
    confirmation_summary_rows: List[Dict[str, Any]],
    paired_rows: List[Dict[str, Any]],
    run_dir: Path,
) -> None:
    _plot_stage_curves(screening_results, "screening", run_dir)
    _plot_stage_curves(confirmation_results, "confirmation", run_dir)
    _plot_final_metric_bars(
        screening_summary_rows,
        "screening",
        "final_exploitability",
        "Final exploitability",
        "screening_final_exploitability_by_variant.png",
        run_dir,
    )
    _plot_final_metric_bars(
        confirmation_summary_rows,
        "confirmation",
        "final_exploitability",
        "Final exploitability",
        "confirmation_final_exploitability_by_variant.png",
        run_dir,
    )
    _plot_final_metric_bars(
        confirmation_summary_rows,
        "confirmation",
        "final_window_mean_exploitability",
        "Final-window exploitability",
        "confirmation_final_window_exploitability_by_variant.png",
        run_dir,
    )
    _plot_final_metric_bars(
        confirmation_summary_rows,
        "confirmation",
        "final_policy_value_error",
        "Final policy-value error from -1/18",
        "confirmation_policy_value_error_by_variant.png",
        run_dir,
    )
    _plot_paired_delta(paired_rows, run_dir)
    _plot_confirmation_diagnostics(confirmation_results, run_dir)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Kuhn poker ESCHER solver-parameter random search."
    )
    parser.add_argument("--output-root", default="outputs", help="Root directory for outputs.")
    parser.add_argument("--screening-seeds", default=None)
    parser.add_argument("--confirmation-seeds", default=None)
    parser.add_argument("--screening-iterations", type=int, default=SCREENING_ITERATIONS)
    parser.add_argument("--confirmation-iterations", type=int, default=CONFIRMATION_ITERATIONS)
    parser.add_argument("--screening-evaluation-interval", type=int, default=SCREENING_EVALUATION_INTERVAL)
    parser.add_argument(
        "--confirmation-evaluation-interval",
        type=int,
        default=CONFIRMATION_EVALUATION_INTERVAL,
    )
    parser.add_argument("--n-random-candidates", type=int, default=N_RANDOM_CANDIDATES)
    parser.add_argument("--random-search-seed", type=int, default=RANDOM_SEARCH_SEED)
    parser.add_argument("--confirmation-top-k", type=int, default=CONFIRMATION_TOP_K)
    parser.add_argument("--skip-confirmation", action="store_true")
    parser.add_argument("--traversals", type=int, default=None)
    parser.add_argument("--value-traversals", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--memory-capacity", type=int, default=None)
    parser.add_argument("--batch-size-regret", type=int, default=None)
    parser.add_argument("--batch-size-value", type=int, default=None)
    parser.add_argument("--batch-size-average-policy", type=int, default=None)
    parser.add_argument("--policy-network-train-steps", type=int, default=None)
    parser.add_argument("--regret-network-train-steps", type=int, default=None)
    parser.add_argument("--value-network-train-steps", type=int, default=None)
    parser.add_argument("--policy-network-layers", default=None)
    parser.add_argument("--regret-network-layers", default=None)
    parser.add_argument("--value-network-layers", default=None)
    parser.add_argument("--reinitialize-regret-networks", type=_str2bool, default=None)
    parser.add_argument("--reinitialize-value-network", type=_str2bool, default=None)
    parser.add_argument("--expl", type=float, default=None)
    parser.add_argument("--val-expl", type=float, default=None)
    parser.add_argument("--importance-sampling", type=_str2bool, default=None)
    parser.add_argument("--importance-sampling-threshold", type=float, default=None)
    parser.add_argument("--clear-value-buffer", type=_str2bool, default=None)
    parser.add_argument("--val-bootstrap", type=_str2bool, default=None)
    parser.add_argument("--use-balanced-probs", type=_str2bool, default=None)
    parser.add_argument("--val-op-prob", type=float, default=None)
    parser.add_argument("--all-actions", type=_str2bool, default=None)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    base_config = build_base_config(args)
    screening_seeds = parse_seeds(args.screening_seeds, SCREENING_SEEDS)
    confirmation_seeds = parse_seeds(args.confirmation_seeds, CONFIRMATION_SEEDS)
    screening_configs = build_screening_configs(
        base_config,
        n_random_candidates=args.n_random_candidates,
        random_search_seed=args.random_search_seed,
    )
    forced_overrides = explicit_config_overrides(args)
    if forced_overrides:
        for config in screening_configs:
            config.update(forced_overrides)

    run_dir = create_run_dir(args.output_root, base_config["experiment_name"])
    _configure_logging(run_dir, args.verbose)
    _LOGGER.info("Export directory: %s", run_dir.resolve())
    _LOGGER.info("Screening variants: %s", [config["variant_id"] for config in screening_configs])
    _LOGGER.info("Screening seeds: %s", screening_seeds)
    _LOGGER.info("Confirmation seeds: %s", confirmation_seeds)

    screening_results = _run_stage(
        screening_configs,
        screening_seeds,
        "screening",
        args.screening_iterations,
        args.screening_evaluation_interval,
    )
    screening_summary_rows = [result["summary"] for result in screening_results]
    screening_aggregate_rows = aggregate_summaries(screening_summary_rows)
    selected_variant_ids = select_confirmation_variants(
        screening_aggregate_rows,
        BASELINE_VARIANT_ID,
        args.confirmation_top_k,
    )
    candidate_lookup = {config["variant_id"]: config for config in screening_configs}
    confirmation_configs = [dict(base_config)]
    confirmation_configs[0]["variant_id"] = BASELINE_VARIANT_ID
    confirmation_configs.extend(candidate_lookup[variant_id] for variant_id in selected_variant_ids)

    confirmation_results = []
    confirmation_summary_rows = []
    confirmation_aggregate_rows = []
    paired_rows = []
    paired_aggregate_rows = []
    if not args.skip_confirmation:
        confirmation_results = _run_stage(
            confirmation_configs,
            confirmation_seeds,
            "confirmation",
            args.confirmation_iterations,
            args.confirmation_evaluation_interval,
        )
        confirmation_summary_rows = [result["summary"] for result in confirmation_results]
        confirmation_aggregate_rows = aggregate_summaries(confirmation_summary_rows)
        paired_rows = paired_differences_vs_baseline(
            confirmation_summary_rows,
            BASELINE_VARIANT_ID,
        )
        paired_aggregate_rows = _aggregate_delta_summaries(paired_rows) if paired_rows else []

    _write_csv(run_dir / "screening_seed_summary.csv", screening_summary_rows, [
        "stage", "variant_id", "seed", "status", "final_exploitability",
        "best_exploitability", "final_window_mean_exploitability", "exploitability_auc",
    ])
    _write_csv(run_dir / "screening_aggregate_by_variant.csv", screening_aggregate_rows, [
        "variant_id", "final_exploitability_mean", "final_window_mean_exploitability_mean",
        "exploitability_auc_mean", "n_runs", "n_completed",
    ])
    _write_csv(run_dir / "confirmation_seed_summary.csv", confirmation_summary_rows, [
        "stage", "variant_id", "seed", "status", "final_exploitability",
        "best_exploitability", "final_window_mean_exploitability", "exploitability_auc",
    ])
    _write_csv(run_dir / "confirmation_aggregate_by_variant.csv", confirmation_aggregate_rows, [
        "variant_id", "final_exploitability_mean", "final_window_mean_exploitability_mean",
        "exploitability_auc_mean", "n_runs", "n_completed",
    ])
    _write_csv(run_dir / "confirmation_paired_differences_vs_baseline.csv", paired_rows, [
        "variant_id", "seed", "delta_final_exploitability",
        "delta_final_window_mean_exploitability", "delta_exploitability_auc",
    ])
    _write_csv(run_dir / "confirmation_paired_difference_summary.csv", paired_aggregate_rows, [
        "variant_id", "delta_final_exploitability_mean",
        "delta_final_window_mean_exploitability_mean", "delta_exploitability_auc_mean",
    ])
    screening_curve_rows = _curve_rows(screening_results)
    confirmation_curve_rows = _curve_rows(confirmation_results)
    all_curve_rows = screening_curve_rows + confirmation_curve_rows
    curve_fields = [
        "stage", "variant_id", "seed", "iteration", "nodes_touched",
        "wall_clock_seconds", "exploitability", "average_policy_value", "policy_value_error",
        "policy_loss", "value_loss", "value_test_loss", "regret_loss_player_0",
        "regret_loss_player_1", "average_policy_buffer_size",
        "regret_buffer_size_player_0", "regret_buffer_size_player_1",
        "value_buffer_size", "value_test_buffer_size",
    ]
    _write_csv(run_dir / "checkpoint_curves.csv", all_curve_rows, curve_fields)
    _write_csv(run_dir / "screening_checkpoint_curves.csv", screening_curve_rows, curve_fields)
    _write_csv(run_dir / "confirmation_checkpoint_curves.csv", confirmation_curve_rows, curve_fields)

    with open(run_dir / "aggregate_summary.json", "w", encoding="utf-8") as f:
        json.dump(json_safe({
            "screening": screening_aggregate_rows,
            "confirmation": confirmation_aggregate_rows,
            "paired_confirmation_vs_baseline": paired_aggregate_rows,
        }), f, indent=2)

    metadata = {
        "experiment_name": base_config["experiment_name"],
        "baseline_variant_id": BASELINE_VARIANT_ID,
        "baseline_config": base_config,
        "screening_iterations": args.screening_iterations,
        "screening_evaluation_interval": args.screening_evaluation_interval,
        "screening_seeds": screening_seeds,
        "confirmation_iterations": args.confirmation_iterations,
        "confirmation_evaluation_interval": args.confirmation_evaluation_interval,
        "confirmation_seeds": confirmation_seeds,
        "confirmation_top_k": args.confirmation_top_k,
        "selected_variant_ids": selected_variant_ids,
        "screening_configs": screening_configs,
        "confirmation_configs": confirmation_configs,
        "random_search_seed": args.random_search_seed,
        "forced_cli_overrides": forced_overrides,
        "kuhn_game_value_player_0": KUHN_GAME_VALUE_PLAYER_0,
        "tensorflow_version": tf.__version__,
        "pyspiel_version": getattr(pyspiel, "__version__", "unknown"),
    }
    with open(run_dir / "experiment_metadata.json", "w", encoding="utf-8") as f:
        json.dump(json_safe(metadata), f, indent=2)

    np.savez_compressed(
        run_dir / "solver_parameter_random_search_curves.npz",
        screening_seed_summary=np.asarray(json_safe(screening_summary_rows), dtype=object),
        confirmation_seed_summary=np.asarray(json_safe(confirmation_summary_rows), dtype=object),
        checkpoint_curves=np.asarray(json_safe(all_curve_rows), dtype=object),
    )

    _plot_outputs(
        screening_results,
        confirmation_results,
        screening_summary_rows,
        confirmation_summary_rows,
        paired_rows,
        run_dir,
    )
    _LOGGER.info("Selected for confirmation: %s", selected_variant_ids)
    _LOGGER.info("Saved outputs to: %s", run_dir.resolve())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
