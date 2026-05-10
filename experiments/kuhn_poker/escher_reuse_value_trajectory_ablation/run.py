"""CLI entry point for the ESCHER value-trajectory reuse ablation."""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import gc
import json
import logging
import os
from pathlib import Path
import sys
import time
import traceback
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
from open_spiel.python import policy  # noqa: E402
from open_spiel.python.algorithms import exploitability, expected_game_score  # noqa: E402
from tqdm import tqdm  # noqa: E402

from escher_poker.constants import EXPLOITABILITY_THRESHOLD, KUHN_GAME_VALUE_PLAYER_0  # noqa: E402
from escher_poker.experiment_utils import (  # noqa: E402
    create_run_dir,
    final_window_mean,
    first_nodes_to_threshold,
    first_time_to_threshold,
    json_safe,
    make_escher_solver,
    normalised_auc,
    safe_stats,
    to_float,
)
from escher_poker.seeding import set_seed_tf  # noqa: E402

from .config import (  # noqa: E402
    DEFAULT_CONFIG,
    REFERENCE_VARIANT_ID,
    TREATMENT_VARIANT_ID,
    VARIANTS,
    make_variant_config,
    parse_seeds,
)

_LOGGER = logging.getLogger("escher_poker.experiment")

METRICS_TO_SUMMARISE = [
    "final_exploitability",
    "best_exploitability",
    "final_window_mean_exploitability",
    "final_policy_value_error",
    "best_policy_value_error",
    "exploitability_auc_by_iteration",
    "exploitability_auc_by_nodes",
    "final_nodes_touched",
    "final_wall_clock_seconds",
    "total_outer_wall_clock_seconds",
    "final_policy_loss",
    "final_value_loss",
    "final_value_test_loss",
    "final_regret_loss_player_0",
    "final_regret_loss_player_1",
    "dedicated_value_train_traversals_per_iteration",
    "dedicated_value_train_traversals_total",
]

PAIRED_METRICS = [
    "delta_final_exploitability_reuse_minus_baseline",
    "delta_best_exploitability_reuse_minus_baseline",
    "delta_final_window_mean_exploitability_reuse_minus_baseline",
    "delta_final_policy_value_error_reuse_minus_baseline",
    "delta_exploitability_auc_by_iteration_reuse_minus_baseline",
    "delta_exploitability_auc_by_nodes_reuse_minus_baseline",
    "delta_final_nodes_touched_reuse_minus_baseline",
    "delta_final_wall_clock_seconds_reuse_minus_baseline",
    "delta_dedicated_value_train_traversals_total_reuse_minus_baseline",
]


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


def build_config(args) -> dict:
    config = deepcopy(DEFAULT_CONFIG)
    overrides = {
        "num_iterations": args.iterations,
        "num_traversals": args.traversals,
        "num_val_fn_traversals": args.value_traversals,
        "value_test_traversals": args.value_test_traversals,
        "check_exploitability_every": args.evaluation_interval,
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
        "bootstrap_value_with_separate_traversal": (
            args.bootstrap_value_with_separate_traversal
        ),
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    return config


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


def cleanup_memory() -> None:
    gc.collect()
    try:
        tf.keras.backend.clear_session()
    except Exception:
        pass


def _safe_last(values: Iterable[float]) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(values[-1]) if values.size else np.nan


def _safe_min(values: Iterable[float]) -> float:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return float(np.min(finite)) if finite.size else np.nan


def run_single_seed_variant(
    seed: int,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Run one seed for one value-trajectory reuse variant."""
    set_seed_tf(seed)
    game = pyspiel.load_game(config["game_name"])
    solver = make_escher_solver(game, config)
    start = time.time()
    result_tuple = solver.solve()
    elapsed = time.time() - start
    _regret_losses, policy_loss, convs, nodes_touched, avg_policy_values, diagnostics = result_tuple

    exploitability_curve = np.asarray(convs, dtype=np.float64) / 2.0
    nodes_touched = np.asarray(nodes_touched, dtype=np.float64)
    avg_policy_values = np.asarray(avg_policy_values, dtype=np.float64)
    value_error = np.abs(avg_policy_values - KUHN_GAME_VALUE_PLAYER_0)
    diagnostics = {key: np.asarray(value) for key, value in diagnostics.items()}
    iterations = diagnostics.get("iteration", np.asarray([], dtype=int)).astype(int)
    wall_clock = diagnostics.get("wall_clock_seconds", np.asarray([], dtype=float)).astype(float)

    final_policy = policy.tabular_policy_from_callable(game, solver.action_probabilities)
    final_nash_conv = exploitability.nash_conv(game, final_policy)
    final_policy_value = expected_game_score.policy_value(
        game.new_initial_state(),
        [final_policy] * game.num_players(),
    )[0]

    dedicated_value_traversals_per_iteration = (
        0
        if config.get("reuse_regret_traversals_for_value", False)
        else int(config["num_val_fn_traversals"])
    )
    dedicated_value_traversals_total = (
        dedicated_value_traversals_per_iteration * int(config["num_iterations"])
    )
    summary = {
        "variant_id": config["variant_id"],
        "variant_label": config["variant_label"],
        "seed": int(seed),
        "status": "completed",
        "reuse_regret_traversals_for_value": bool(
            config["reuse_regret_traversals_for_value"]
        ),
        "dedicated_value_train_traversals_per_iteration": (
            dedicated_value_traversals_per_iteration
        ),
        "dedicated_value_train_traversals_total": dedicated_value_traversals_total,
        "final_exploitability": _safe_last(exploitability_curve),
        "best_exploitability": _safe_min(exploitability_curve),
        "final_window_mean_exploitability": final_window_mean(exploitability_curve),
        "final_policy_value": float(final_policy_value),
        "final_policy_value_error": float(abs(final_policy_value - KUHN_GAME_VALUE_PLAYER_0)),
        "best_policy_value_error": _safe_min(value_error),
        "exploitability_auc_by_iteration": normalised_auc(iterations, exploitability_curve),
        "exploitability_auc_by_nodes": normalised_auc(nodes_touched, exploitability_curve),
        "final_nodes_touched": _safe_last(nodes_touched),
        "final_wall_clock_seconds": _safe_last(wall_clock),
        "total_outer_wall_clock_seconds": float(elapsed),
        "nodes_to_exploitability_threshold": first_nodes_to_threshold(
            nodes_touched,
            exploitability_curve,
            config["exploitability_threshold"],
        ),
        "seconds_to_exploitability_threshold": first_time_to_threshold(
            wall_clock,
            exploitability_curve,
            config["exploitability_threshold"],
        ),
        "final_policy_loss": (
            _safe_last(diagnostics["policy_loss"])
            if "policy_loss" in diagnostics
            else to_float(policy_loss)
        ),
        "final_value_loss": _safe_last(diagnostics.get("value_loss", [])),
        "final_value_test_loss": _safe_last(diagnostics.get("value_test_loss", [])),
        "final_regret_loss_player_0": _safe_last(
            diagnostics.get("regret_loss_player_0", [])
        ),
        "final_regret_loss_player_1": _safe_last(
            diagnostics.get("regret_loss_player_1", [])
        ),
        "final_average_policy_buffer_size": _safe_last(
            diagnostics.get("average_policy_buffer_size", [])
        ),
        "final_regret_buffer_size_player_0": _safe_last(
            diagnostics.get("regret_buffer_size_player_0", [])
        ),
        "final_regret_buffer_size_player_1": _safe_last(
            diagnostics.get("regret_buffer_size_player_1", [])
        ),
        "final_nash_conv_recomputed": float(final_nash_conv),
    }

    curve_rows = []
    for idx, iteration in enumerate(iterations):
        row = {
            "variant_id": config["variant_id"],
            "variant_label": config["variant_label"],
            "seed": int(seed),
            "checkpoint_index": int(idx),
            "iteration": int(iteration),
            "nodes_touched": (
                float(nodes_touched[idx]) if idx < len(nodes_touched) else np.nan
            ),
            "wall_clock_seconds": (
                float(wall_clock[idx]) if idx < len(wall_clock) else np.nan
            ),
            "exploitability": (
                float(exploitability_curve[idx])
                if idx < len(exploitability_curve)
                else np.nan
            ),
            "average_policy_value": (
                float(avg_policy_values[idx]) if idx < len(avg_policy_values) else np.nan
            ),
            "policy_value_error": (
                float(value_error[idx]) if idx < len(value_error) else np.nan
            ),
        }
        for key, arr in diagnostics.items():
            if len(arr) > idx:
                row[key] = to_float(arr[idx])
        curve_rows.append(row)

    result = {
        "variant_id": config["variant_id"],
        "variant_label": config["variant_label"],
        "seed": int(seed),
        "iterations": iterations,
        "nodes_touched": nodes_touched,
        "wall_clock_seconds": wall_clock,
        "exploitability": exploitability_curve,
        "average_policy_value": avg_policy_values,
        "policy_value_error": value_error,
        "diagnostics": diagnostics,
        "summary": summary,
        "curves": curve_rows,
    }

    del solver
    cleanup_memory()
    return result


def _summarise_by_variant(summary_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for variant_id in sorted({row["variant_id"] for row in summary_rows}):
        variant_rows = [row for row in summary_rows if row["variant_id"] == variant_id]
        label = variant_rows[0].get("variant_label", variant_id)
        for metric in METRICS_TO_SUMMARISE:
            rows.append({
                "variant_id": variant_id,
                "variant_label": label,
                "metric": metric,
                **safe_stats([row.get(metric, np.nan) for row in variant_rows]),
            })
    return rows


def _paired_differences(summary_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_variant_seed = {
        (row["variant_id"], int(row["seed"])): row
        for row in summary_rows
    }
    baseline_seeds = {
        int(row["seed"])
        for row in summary_rows
        if row["variant_id"] == REFERENCE_VARIANT_ID
    }
    treatment_seeds = {
        int(row["seed"])
        for row in summary_rows
        if row["variant_id"] == TREATMENT_VARIANT_ID
    }
    rows = []
    for seed in sorted(baseline_seeds & treatment_seeds):
        baseline = by_variant_seed[(REFERENCE_VARIANT_ID, seed)]
        treatment = by_variant_seed[(TREATMENT_VARIANT_ID, seed)]
        rows.append({
            "variant_id": TREATMENT_VARIANT_ID,
            "seed": int(seed),
            "delta_final_exploitability_reuse_minus_baseline": float(
                treatment["final_exploitability"] - baseline["final_exploitability"]
            ),
            "delta_best_exploitability_reuse_minus_baseline": float(
                treatment["best_exploitability"] - baseline["best_exploitability"]
            ),
            "delta_final_window_mean_exploitability_reuse_minus_baseline": float(
                treatment["final_window_mean_exploitability"]
                - baseline["final_window_mean_exploitability"]
            ),
            "delta_final_policy_value_error_reuse_minus_baseline": float(
                treatment["final_policy_value_error"]
                - baseline["final_policy_value_error"]
            ),
            "delta_exploitability_auc_by_iteration_reuse_minus_baseline": float(
                treatment["exploitability_auc_by_iteration"]
                - baseline["exploitability_auc_by_iteration"]
            ),
            "delta_exploitability_auc_by_nodes_reuse_minus_baseline": float(
                treatment["exploitability_auc_by_nodes"]
                - baseline["exploitability_auc_by_nodes"]
            ),
            "delta_final_nodes_touched_reuse_minus_baseline": float(
                treatment["final_nodes_touched"] - baseline["final_nodes_touched"]
            ),
            "delta_final_wall_clock_seconds_reuse_minus_baseline": float(
                treatment["final_wall_clock_seconds"]
                - baseline["final_wall_clock_seconds"]
            ),
            "delta_dedicated_value_train_traversals_total_reuse_minus_baseline": float(
                treatment["dedicated_value_train_traversals_total"]
                - baseline["dedicated_value_train_traversals_total"]
            ),
        })
    return rows


def _paired_summary(paired_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for metric in PAIRED_METRICS:
        rows.append({
            "variant_id": TREATMENT_VARIANT_ID,
            "metric": metric,
            **safe_stats([row.get(metric, np.nan) for row in paired_rows]),
        })
    return rows


def _curve_rows(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for result in results:
        rows.extend(result["curves"])
    return rows


def _variant_label(variant_id: str) -> str:
    for variant in VARIANTS:
        if variant["variant_id"] == variant_id:
            return variant["variant_label"]
    return variant_id


def _mean_curve(curve_rows: List[Dict[str, Any]], variant_id: str, metric: str, x_col: str):
    grouped = {}
    for row in curve_rows:
        if row["variant_id"] == variant_id:
            grouped.setdefault(int(row["iteration"]), []).append(row)
    rows = []
    for iteration in sorted(grouped):
        stats = safe_stats([row[metric] for row in grouped[iteration]])
        x_stats = safe_stats([row[x_col] for row in grouped[iteration]])
        rows.append({
            "iteration": iteration,
            "x_mean": x_stats["mean"],
            "mean": stats["mean"],
            "se": 0.0 if not np.isfinite(stats["se"]) else stats["se"],
        })
    return rows


def _plot_curve(
    curve_rows: List[Dict[str, Any]],
    metric: str,
    ylabel: str,
    title: str,
    filename: str,
    run_dir: Path,
    *,
    x_col: str = "iteration",
) -> None:
    if not curve_rows:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for variant in VARIANTS:
        variant_id = variant["variant_id"]
        means = _mean_curve(curve_rows, variant_id, metric, x_col)
        if not means:
            continue
        x = np.asarray([row["x_mean"] for row in means], dtype=float)
        mean = np.asarray([row["mean"] for row in means], dtype=float)
        se = np.asarray([row["se"] for row in means], dtype=float)
        ax.plot(x, mean, linewidth=2, label=_variant_label(variant_id))
        ax.fill_between(x, mean - se, mean + se, alpha=0.15)
    if metric == "exploitability":
        ax.axhline(EXPLOITABILITY_THRESHOLD, linestyle="--", linewidth=1)
    ax.set_xlabel(x_col.replace("_", " ").title())
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_final_bar(summary_rows: List[Dict[str, Any]], run_dir: Path) -> None:
    labels = []
    means = []
    errors = []
    for variant in VARIANTS:
        values = [
            row["final_exploitability"]
            for row in summary_rows
            if row["variant_id"] == variant["variant_id"]
        ]
        if not values:
            continue
        stats = safe_stats(values)
        labels.append(variant["variant_label"])
        means.append(stats["mean"])
        errors.append(0.0 if not np.isfinite(stats["se"]) else stats["se"])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, means, yerr=errors, capsize=4)
    ax.set_ylabel("Final exploitability (NashConv/2)")
    ax.set_title("ESCHER value-trajectory reuse ablation: final exploitability")
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(run_dir / "final_exploitability_reuse_ablation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_metric_bar(
    summary_rows: List[Dict[str, Any]],
    metric: str,
    ylabel: str,
    title: str,
    filename: str,
    run_dir: Path,
) -> None:
    labels = []
    means = []
    errors = []
    for variant in VARIANTS:
        values = [
            row[metric]
            for row in summary_rows
            if row["variant_id"] == variant["variant_id"] and metric in row
        ]
        if not values:
            continue
        stats = safe_stats(values)
        labels.append(variant["variant_label"])
        means.append(stats["mean"])
        errors.append(0.0 if not np.isfinite(stats["se"]) else stats["se"])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, means, yerr=errors, capsize=4)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(run_dir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_paired_delta(paired_rows: List[Dict[str, Any]], run_dir: Path) -> None:
    if not paired_rows:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.scatter(
        [row["seed"] for row in paired_rows],
        [row["delta_final_exploitability_reuse_minus_baseline"] for row in paired_rows],
        s=50,
    )
    ax.set_xlabel("Seed")
    ax.set_ylabel("Reuse - baseline final exploitability")
    ax.set_title("Paired effect of reusing value trajectories")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(
        run_dir / "paired_final_exploitability_delta_reuse_minus_baseline.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def _plot_outputs(
    curve_rows: List[Dict[str, Any]],
    summary_rows: List[Dict[str, Any]],
    paired_rows: List[Dict[str, Any]],
    run_dir: Path,
) -> None:
    _plot_curve(
        curve_rows,
        "exploitability",
        "Exploitability (NashConv/2)",
        "ESCHER value-trajectory reuse ablation: exploitability",
        "exploitability_by_iteration_reuse_ablation.png",
        run_dir,
    )
    _plot_curve(
        curve_rows,
        "exploitability",
        "Exploitability (NashConv/2)",
        "ESCHER value-trajectory reuse ablation: exploitability by nodes",
        "exploitability_by_nodes_reuse_ablation.png",
        run_dir,
        x_col="nodes_touched",
    )
    _plot_curve(
        curve_rows,
        "policy_value_error",
        "Absolute error from -1/18",
        "ESCHER value-trajectory reuse ablation: policy-value error",
        "policy_value_error_reuse_ablation.png",
        run_dir,
    )
    for key, title, filename in [
        ("policy_loss", "Average-policy network loss", "policy_loss_reuse_ablation.png"),
        ("value_loss", "History-value train loss", "value_loss_reuse_ablation.png"),
        ("value_test_loss", "History-value test loss", "value_test_loss_reuse_ablation.png"),
    ]:
        _plot_curve(curve_rows, key, "MSE loss", title, filename, run_dir)
    _plot_final_bar(summary_rows, run_dir)
    _plot_metric_bar(
        summary_rows,
        "final_wall_clock_seconds",
        "Final wall-clock seconds",
        "ESCHER value-trajectory reuse ablation: runtime",
        "final_wall_clock_seconds_reuse_ablation.png",
        run_dir,
    )
    _plot_metric_bar(
        summary_rows,
        "final_nodes_touched",
        "Final nodes touched",
        "ESCHER value-trajectory reuse ablation: nodes touched",
        "final_nodes_touched_reuse_ablation.png",
        run_dir,
    )
    _plot_metric_bar(
        summary_rows,
        "dedicated_value_train_traversals_total",
        "Dedicated value-training traversals",
        "ESCHER value-trajectory reuse ablation: dedicated value traversals",
        "dedicated_value_traversals_reuse_ablation.png",
        run_dir,
    )
    _plot_paired_delta(paired_rows, run_dir)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Kuhn poker ESCHER value-trajectory reuse ablation."
    )
    parser.add_argument("--output-root", default="outputs", help="Root directory for outputs.")
    parser.add_argument("--seeds", default=None, help="Comma-separated seed list.")
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--traversals", type=int, default=None)
    parser.add_argument("--value-traversals", type=int, default=None)
    parser.add_argument("--value-test-traversals", type=int, default=None)
    parser.add_argument("--evaluation-interval", type=int, default=None)
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
    parser.add_argument("--bootstrap-value-with-separate-traversal", type=_str2bool, default=None)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    base_config = build_config(args)
    seeds = parse_seeds(args.seeds)
    variant_configs = [
        make_variant_config(base_config, variant)
        for variant in VARIANTS
    ]

    run_dir = create_run_dir(args.output_root, base_config["experiment_name"])
    _configure_logging(run_dir, args.verbose)
    _LOGGER.info("Export directory: %s", run_dir.resolve())
    _LOGGER.info("Running seeds: %s", seeds)
    _LOGGER.info("Running variants: %s", [config["variant_id"] for config in variant_configs])

    results = []
    failures = []
    for config in variant_configs:
        for seed in tqdm(seeds, desc=config["variant_id"]):
            try:
                results.append(run_single_seed_variant(seed, config))
            except Exception as exc:  # pragma: no cover - operational robustness
                _LOGGER.error("variant=%s seed=%s failed: %s", config["variant_id"], seed, exc)
                failures.append({
                    "variant_id": config["variant_id"],
                    "seed": int(seed),
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                })

    summary_rows = [result["summary"] for result in results]
    curve_rows = _curve_rows(results)
    aggregate_rows = _summarise_by_variant(summary_rows)
    paired_rows = _paired_differences(summary_rows)
    paired_summary_rows = _paired_summary(paired_rows)

    _write_csv(run_dir / "seed_summary.csv", summary_rows, [
        "variant_id", "seed", "status", "reuse_regret_traversals_for_value",
        "final_exploitability", "best_exploitability",
        "final_window_mean_exploitability", "final_policy_value_error",
        "exploitability_auc_by_iteration",
        "dedicated_value_train_traversals_total",
    ])
    _write_csv(run_dir / "variant_aggregate_summary.csv", aggregate_rows, [
        "variant_id", "metric", "mean", "std", "se", "n_finite",
    ])
    _write_csv(run_dir / "paired_differences_vs_baseline.csv", paired_rows, [
        "variant_id", "seed", "delta_final_exploitability_reuse_minus_baseline",
        "delta_best_exploitability_reuse_minus_baseline",
        "delta_final_window_mean_exploitability_reuse_minus_baseline",
        "delta_final_policy_value_error_reuse_minus_baseline",
        "delta_exploitability_auc_by_iteration_reuse_minus_baseline",
        "delta_final_nodes_touched_reuse_minus_baseline",
        "delta_final_wall_clock_seconds_reuse_minus_baseline",
        "delta_dedicated_value_train_traversals_total_reuse_minus_baseline",
    ])
    _write_csv(run_dir / "paired_difference_summary.csv", paired_summary_rows, [
        "variant_id", "metric", "mean", "std", "se", "n_finite",
    ])
    _write_csv(run_dir / "checkpoint_curves.csv", curve_rows, [
        "variant_id", "seed", "iteration", "nodes_touched", "wall_clock_seconds",
        "exploitability", "average_policy_value", "policy_value_error",
        "policy_loss", "value_loss", "value_test_loss",
    ])

    with open(run_dir / "aggregate_summary.json", "w", encoding="utf-8") as f:
        json.dump(json_safe({
            "by_variant": aggregate_rows,
            "paired_reuse_minus_baseline": paired_summary_rows,
        }), f, indent=2)
    with open(run_dir / "paired_difference_summary.json", "w", encoding="utf-8") as f:
        json.dump(json_safe(paired_summary_rows), f, indent=2)
    with open(run_dir / "experiment_metadata.json", "w", encoding="utf-8") as f:
        json.dump(json_safe({
            "experiment_name": base_config["experiment_name"],
            "base_config": base_config,
            "variants": VARIANTS,
            "reference_variant_id": REFERENCE_VARIANT_ID,
            "treatment_variant_id": TREATMENT_VARIANT_ID,
            "seeds": seeds,
            "kuhn_game_value_player_0": KUHN_GAME_VALUE_PLAYER_0,
            "implementation_note": (
                "The reuse treatment records history-value targets during player-0 "
                "regret traversals and skips the dedicated value-training traversal pass."
            ),
            "tensorflow_version": tf.__version__,
            "pyspiel_version": getattr(pyspiel, "__version__", "unknown"),
        }), f, indent=2)

    if failures:
        with open(run_dir / "failed_runs.json", "w", encoding="utf-8") as f:
            json.dump(json_safe(failures), f, indent=2)

    if results:
        np.savez_compressed(
            run_dir / "reuse_value_trajectory_ablation_curves.npz",
            seeds=np.asarray(seeds),
            seed_summary=np.asarray(json_safe(summary_rows), dtype=object),
            checkpoint_curves=np.asarray(json_safe(curve_rows), dtype=object),
        )
    _plot_outputs(curve_rows, summary_rows, paired_rows, run_dir)

    _LOGGER.info("Saved outputs to: %s", run_dir.resolve())
    return 0 if not failures else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
