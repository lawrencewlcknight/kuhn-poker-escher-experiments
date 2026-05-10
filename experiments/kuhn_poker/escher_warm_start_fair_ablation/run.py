"""CLI entry point for the ESCHER warm-start fair ablation."""

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
from tqdm import tqdm  # noqa: E402

from escher_poker.constants import EXPLOITABILITY_THRESHOLD, KUHN_GAME_VALUE_PLAYER_0  # noqa: E402
from escher_poker.experiment_utils import (  # noqa: E402
    create_run_dir,
    evaluate_final_policy,
    final_window_mean,
    first_nodes_to_threshold,
    first_time_to_threshold,
    json_safe,
    make_escher_solver,
    normalised_auc,
    safe_stats,
    to_float,
)
from escher_poker.policy_snapshots import (  # noqa: E402
    load_full_solver_checkpoint,
    save_full_solver_checkpoint,
)
from escher_poker.seeding import set_seed_tf  # noqa: E402

from .config import (  # noqa: E402
    ARM_CONTINUOUS,
    ARM_LABELS,
    ARM_WARM_START,
    DEFAULT_CONFIG,
    RESTORE_RNG_STATE,
    SAVE_FULL_CHECKPOINTS,
    parse_seeds,
)

_LOGGER = logging.getLogger("escher_poker.experiment")

SUMMARY_METRICS = [
    "final_exploitability",
    "final_exploitability_recomputed",
    "best_exploitability",
    "final_window_mean_exploitability",
    "normalised_exploitability_auc_by_nodes",
    "normalised_exploitability_auc_by_iteration",
    "final_policy_value_error",
    "final_policy_value_error_recomputed",
    "best_policy_value_error",
    "final_nodes_touched",
    "final_wall_clock_seconds",
    "total_outer_wall_clock_seconds",
    "policy_training_events",
]

PAIRED_METRICS = [
    "delta_final_exploitability_warm_minus_baseline",
    "delta_final_exploitability_recomputed_warm_minus_baseline",
    "delta_best_exploitability_warm_minus_baseline",
    "delta_final_window_mean_exploitability_warm_minus_baseline",
    "delta_auc_nodes_warm_minus_baseline",
    "delta_auc_iteration_warm_minus_baseline",
    "delta_final_policy_value_error_warm_minus_baseline",
    "delta_final_policy_value_error_recomputed_warm_minus_baseline",
    "delta_wall_clock_seconds_warm_minus_baseline",
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
        "warm_start_boundary": args.warm_start_boundary,
        "num_traversals": args.traversals,
        "num_val_fn_traversals": args.value_traversals,
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


def _extract_result(
    seed: int,
    arm: str,
    result_tuple,
    *,
    iteration_offset: int = 0,
    wall_clock_offset: float = 0.0,
) -> Dict[str, Any]:
    _regret_losses, policy_loss, convs, nodes_touched, avg_policy_values, diagnostics = result_tuple
    exploitability_curve = np.asarray(convs, dtype=np.float64) / 2.0
    nodes = np.asarray(nodes_touched, dtype=np.float64)
    values = np.asarray(avg_policy_values, dtype=np.float64)
    diagnostics = {key: np.asarray(value) for key, value in diagnostics.items()}

    raw_iterations = diagnostics.get("iteration", np.arange(len(exploitability_curve)))
    iterations = raw_iterations.astype(int) - 1 + int(iteration_offset)
    raw_wall_clock = diagnostics.get("wall_clock_seconds", np.zeros(len(exploitability_curve)))
    wall_clock = raw_wall_clock.astype(float) + float(wall_clock_offset)
    value_error = np.abs(values - KUHN_GAME_VALUE_PLAYER_0)

    curve_rows = []
    for idx, iteration in enumerate(iterations):
        row = {
            "seed": int(seed),
            "arm": arm,
            "arm_label": ARM_LABELS[arm],
            "checkpoint_index": int(idx),
            "iteration": int(iteration),
            "nodes_touched": float(nodes[idx]) if idx < len(nodes) else np.nan,
            "wall_clock_seconds": (
                float(wall_clock[idx]) if idx < len(wall_clock) else np.nan
            ),
            "exploitability": (
                float(exploitability_curve[idx])
                if idx < len(exploitability_curve)
                else np.nan
            ),
            "average_policy_value": float(values[idx]) if idx < len(values) else np.nan,
            "policy_value_error": (
                float(value_error[idx]) if idx < len(value_error) else np.nan
            ),
        }
        for key, arr in diagnostics.items():
            if len(arr) > idx:
                row[key] = to_float(arr[idx])
        curve_rows.append(row)

    return {
        "seed": int(seed),
        "arm": arm,
        "iterations": iterations,
        "nodes_touched": nodes,
        "wall_clock_seconds": wall_clock,
        "exploitability": exploitability_curve,
        "average_policy_value": values,
        "policy_value_error": value_error,
        "policy_loss": to_float(policy_loss),
        "curves": curve_rows,
    }


def _summarise_result(
    seed: int,
    arm: str,
    result: Dict[str, Any],
    config: Dict[str, Any],
    final_eval: Dict[str, float],
    *,
    outer_wall_clock_seconds: float,
    checkpoint_path: str = "",
) -> Dict[str, Any]:
    exploitability_curve = np.asarray(result["exploitability"], dtype=np.float64)
    nodes = np.asarray(result["nodes_touched"], dtype=np.float64)
    iterations = np.asarray(result["iterations"], dtype=np.float64)
    wall_clock = np.asarray(result["wall_clock_seconds"], dtype=np.float64)
    value_error = np.asarray(result["policy_value_error"], dtype=np.float64)
    total_iterations = int(config["num_iterations"])
    policy_events = int(len(exploitability_curve))

    return {
        "seed": int(seed),
        "arm": arm,
        "arm_label": ARM_LABELS[arm],
        "status": "completed",
        "final_exploitability": _safe_last(exploitability_curve),
        "final_exploitability_recomputed": final_eval["final_exploitability"],
        "final_nash_conv_recomputed": final_eval["final_nash_conv_recomputed"],
        "best_exploitability": _safe_min(exploitability_curve),
        "final_window_mean_exploitability": final_window_mean(exploitability_curve),
        "normalised_exploitability_auc_by_nodes": normalised_auc(nodes, exploitability_curve),
        "normalised_exploitability_auc_by_iteration": normalised_auc(
            iterations,
            exploitability_curve,
        ),
        "final_policy_value": _safe_last(result["average_policy_value"]),
        "final_policy_value_recomputed": final_eval["final_policy_value"],
        "final_policy_value_error": _safe_last(value_error),
        "final_policy_value_error_recomputed": final_eval["final_policy_value_error"],
        "best_policy_value_error": _safe_min(value_error),
        "final_nodes_touched": _safe_last(nodes),
        "final_wall_clock_seconds": _safe_last(wall_clock),
        "total_outer_wall_clock_seconds": float(outer_wall_clock_seconds),
        "nodes_to_exploitability_threshold": first_nodes_to_threshold(
            nodes,
            exploitability_curve,
            config["exploitability_threshold"],
        ),
        "seconds_to_exploitability_threshold": first_time_to_threshold(
            wall_clock,
            exploitability_curve,
            config["exploitability_threshold"],
        ),
        "policy_training_events": policy_events,
        "policy_network_gradient_steps": int(
            policy_events * config["policy_network_train_steps"]
        ),
        "regret_network_gradient_steps_per_player": int(
            total_iterations * config["regret_network_train_steps"]
        ),
        "value_network_gradient_steps": int(
            total_iterations * config["value_network_train_steps"]
        ),
        "total_iterations": total_iterations,
        "warm_start_boundary": int(config["warm_start_boundary"]),
        "evaluation_interval": int(config["check_exploitability_every"]),
        "final_policy_loss": result["policy_loss"],
        "checkpoint_path": checkpoint_path,
    }


def _deduplicate_curves(curve_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sorted_rows = sorted(curve_rows, key=lambda row: (row["iteration"], row["nodes_touched"]))
    deduped = {}
    for row in sorted_rows:
        deduped.setdefault(int(row["iteration"]), row)
    return [deduped[key] for key in sorted(deduped)]


def _result_from_curve_rows(seed: int, arm: str, curve_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "seed": int(seed),
        "arm": arm,
        "iterations": np.asarray([row["iteration"] for row in curve_rows], dtype=int),
        "nodes_touched": np.asarray([row["nodes_touched"] for row in curve_rows], dtype=float),
        "wall_clock_seconds": np.asarray(
            [row["wall_clock_seconds"] for row in curve_rows],
            dtype=float,
        ),
        "exploitability": np.asarray([row["exploitability"] for row in curve_rows], dtype=float),
        "average_policy_value": np.asarray(
            [row["average_policy_value"] for row in curve_rows],
            dtype=float,
        ),
        "policy_value_error": np.asarray(
            [row["policy_value_error"] for row in curve_rows],
            dtype=float,
        ),
        "policy_loss": np.nan,
        "curves": curve_rows,
    }


def run_continuous_baseline(seed: int, config: Dict[str, Any]) -> Dict[str, Any]:
    """Run the uninterrupted ESCHER baseline arm for one seed."""
    set_seed_tf(seed)
    game = pyspiel.load_game(config["game_name"])
    solver = make_escher_solver(game, config, num_iterations=int(config["num_iterations"]))
    start = time.perf_counter()
    result = _extract_result(seed, ARM_CONTINUOUS, solver.solve())
    final_eval = evaluate_final_policy(game, solver)
    result["summary"] = _summarise_result(
        seed,
        ARM_CONTINUOUS,
        result,
        config,
        final_eval,
        outer_wall_clock_seconds=time.perf_counter() - start,
    )
    del solver
    cleanup_memory()
    return result


def run_warm_start_arm(
    seed: int,
    config: Dict[str, Any],
    checkpoint_dir: Path,
    *,
    save_checkpoint: bool,
    restore_rng_state: bool,
) -> Dict[str, Any]:
    """Run checkpoint/resume with the same headline budget as the baseline arm."""
    boundary = int(config["warm_start_boundary"])
    total = int(config["num_iterations"])
    continuation = total - boundary
    if boundary <= 0:
        raise ValueError("warm_start_boundary must be positive.")
    if continuation <= 0:
        raise ValueError("warm_start_boundary must be smaller than num_iterations.")

    set_seed_tf(seed)
    game = pyspiel.load_game(config["game_name"])
    checkpoint_path = (
        checkpoint_dir
        / f"kuhn_poker_escher_seed_{int(seed)}_warm_start_boundary_{boundary}.pkl"
    )
    start = time.perf_counter()

    solver_initial = make_escher_solver(game, config, num_iterations=boundary)
    phase1 = _extract_result(seed, ARM_WARM_START, solver_initial.solve())
    save_full_solver_checkpoint(
        solver_initial,
        checkpoint_path,
        include_rng=restore_rng_state,
    )
    phase1_time = _safe_last(phase1["wall_clock_seconds"])
    del solver_initial
    cleanup_memory()

    solver_continued = make_escher_solver(game, config, num_iterations=continuation)
    load_full_solver_checkpoint(
        solver_continued,
        checkpoint_path,
        restore_rng=restore_rng_state,
    )
    phase2 = _extract_result(
        seed,
        ARM_WARM_START,
        solver_continued.solve(),
        iteration_offset=boundary,
        wall_clock_offset=0.0 if not np.isfinite(phase1_time) else phase1_time,
    )
    curve_rows = _deduplicate_curves(phase1["curves"] + phase2["curves"])
    result = _result_from_curve_rows(seed, ARM_WARM_START, curve_rows)
    result["policy_loss"] = phase2["policy_loss"]
    final_eval = evaluate_final_policy(game, solver_continued)
    result["summary"] = _summarise_result(
        seed,
        ARM_WARM_START,
        result,
        config,
        final_eval,
        outer_wall_clock_seconds=time.perf_counter() - start,
        checkpoint_path=str(checkpoint_path) if save_checkpoint else "",
    )
    if not save_checkpoint:
        checkpoint_path.unlink(missing_ok=True)

    del solver_continued
    cleanup_memory()
    return result


def _aggregate_by_arm(summary_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    aggregate_rows = []
    for arm in [ARM_CONTINUOUS, ARM_WARM_START]:
        arm_rows = [row for row in summary_rows if row["arm"] == arm]
        if not arm_rows:
            continue
        for metric in SUMMARY_METRICS:
            aggregate_rows.append({
                "arm": arm,
                "arm_label": ARM_LABELS[arm],
                "metric": metric,
                **safe_stats([row.get(metric, np.nan) for row in arm_rows]),
            })
    return aggregate_rows


def _paired_differences(summary_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_arm_seed = {(row["arm"], int(row["seed"])): row for row in summary_rows}
    continuous_seeds = {
        int(row["seed"]) for row in summary_rows if row["arm"] == ARM_CONTINUOUS
    }
    warm_seeds = {
        int(row["seed"]) for row in summary_rows if row["arm"] == ARM_WARM_START
    }
    rows = []
    for seed in sorted(continuous_seeds & warm_seeds):
        baseline = by_arm_seed[(ARM_CONTINUOUS, seed)]
        warm = by_arm_seed[(ARM_WARM_START, seed)]
        rows.append({
            "seed": int(seed),
            "delta_final_exploitability_warm_minus_baseline": float(
                warm["final_exploitability"] - baseline["final_exploitability"]
            ),
            "delta_final_exploitability_recomputed_warm_minus_baseline": float(
                warm["final_exploitability_recomputed"]
                - baseline["final_exploitability_recomputed"]
            ),
            "delta_best_exploitability_warm_minus_baseline": float(
                warm["best_exploitability"] - baseline["best_exploitability"]
            ),
            "delta_final_window_mean_exploitability_warm_minus_baseline": float(
                warm["final_window_mean_exploitability"]
                - baseline["final_window_mean_exploitability"]
            ),
            "delta_auc_nodes_warm_minus_baseline": float(
                warm["normalised_exploitability_auc_by_nodes"]
                - baseline["normalised_exploitability_auc_by_nodes"]
            ),
            "delta_auc_iteration_warm_minus_baseline": float(
                warm["normalised_exploitability_auc_by_iteration"]
                - baseline["normalised_exploitability_auc_by_iteration"]
            ),
            "delta_final_policy_value_error_warm_minus_baseline": float(
                warm["final_policy_value_error"] - baseline["final_policy_value_error"]
            ),
            "delta_final_policy_value_error_recomputed_warm_minus_baseline": float(
                warm["final_policy_value_error_recomputed"]
                - baseline["final_policy_value_error_recomputed"]
            ),
            "delta_wall_clock_seconds_warm_minus_baseline": float(
                warm["total_outer_wall_clock_seconds"]
                - baseline["total_outer_wall_clock_seconds"]
            ),
            "baseline_final_exploitability": float(baseline["final_exploitability"]),
            "warm_start_final_exploitability": float(warm["final_exploitability"]),
            "baseline_recomputed_final_exploitability": float(
                baseline["final_exploitability_recomputed"]
            ),
            "warm_start_recomputed_final_exploitability": float(
                warm["final_exploitability_recomputed"]
            ),
            "baseline_final_nodes_touched": float(baseline["final_nodes_touched"]),
            "warm_start_final_nodes_touched": float(warm["final_nodes_touched"]),
            "baseline_policy_training_events": int(baseline["policy_training_events"]),
            "warm_start_policy_training_events": int(warm["policy_training_events"]),
        })
    return rows


def _aggregate_paired(paired_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for metric in PAIRED_METRICS:
        rows.append({
            "metric": metric,
            **safe_stats([row.get(metric, np.nan) for row in paired_rows]),
        })
    return rows


def _curve_rows(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for result in results:
        rows.extend(result["curves"])
    return rows


def _paired_curve_rows(curve_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_seed_iteration_arm = {
        (int(row["seed"]), int(row["iteration"]), row["arm"]): row
        for row in curve_rows
    }
    seeds = sorted({int(row["seed"]) for row in curve_rows})
    iterations = sorted({int(row["iteration"]) for row in curve_rows})
    rows = []
    for seed in seeds:
        for iteration in iterations:
            baseline = by_seed_iteration_arm.get((seed, iteration, ARM_CONTINUOUS))
            warm = by_seed_iteration_arm.get((seed, iteration, ARM_WARM_START))
            if baseline is None or warm is None:
                continue
            rows.append({
                "seed": int(seed),
                "iteration": int(iteration),
                "delta_exploitability_warm_minus_baseline": float(
                    warm["exploitability"] - baseline["exploitability"]
                ),
                "delta_policy_value_error_warm_minus_baseline": float(
                    warm["policy_value_error"] - baseline["policy_value_error"]
                ),
            })
    return rows


def _mean_curve(curve_rows: List[Dict[str, Any]], arm: str, metric: str, x_col: str):
    grouped = {}
    for row in curve_rows:
        if row["arm"] != arm:
            continue
        grouped.setdefault(int(row["iteration"]), []).append(row)
    rows = []
    for iteration in sorted(grouped):
        values = [row[metric] for row in grouped[iteration]]
        x_values = [row[x_col] for row in grouped[iteration]]
        stats = safe_stats(values)
        rows.append({
            "iteration": iteration,
            "x_mean": safe_stats(x_values)["mean"],
            "mean": stats["mean"],
            "se": 0.0 if not np.isfinite(stats["se"]) else stats["se"],
        })
    return rows


def _plot_arm_curves(
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
    fig, ax = plt.subplots(figsize=(9, 5))
    for arm in [ARM_CONTINUOUS, ARM_WARM_START]:
        arm_seed_rows = {}
        for row in curve_rows:
            if row["arm"] == arm:
                arm_seed_rows.setdefault(row["seed"], []).append(row)
        for seed_rows in arm_seed_rows.values():
            ordered = sorted(seed_rows, key=lambda row: row["iteration"])
            ax.plot(
                [row[x_col] for row in ordered],
                [row[metric] for row in ordered],
                alpha=0.18,
                linewidth=1,
            )
        means = _mean_curve(curve_rows, arm, metric, x_col)
        if not means:
            continue
        x = np.asarray([row["x_mean"] for row in means], dtype=float)
        mean = np.asarray([row["mean"] for row in means], dtype=float)
        se = np.asarray([row["se"] for row in means], dtype=float)
        ax.plot(x, mean, linewidth=2, label=f"{ARM_LABELS[arm]} mean")
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


def _plot_paired_curves(
    paired_curve_rows: List[Dict[str, Any]],
    config: Dict[str, Any],
    run_dir: Path,
) -> None:
    if not paired_curve_rows:
        return
    grouped = {}
    for row in paired_curve_rows:
        grouped.setdefault(int(row["iteration"]), []).append(row)
    x = np.asarray(sorted(grouped), dtype=float)
    means = []
    errors = []
    for iteration in x.astype(int):
        stats = safe_stats([
            row["delta_exploitability_warm_minus_baseline"]
            for row in grouped[iteration]
        ])
        means.append(stats["mean"])
        errors.append(0.0 if not np.isfinite(stats["se"]) else stats["se"])
    mean = np.asarray(means, dtype=float)
    se = np.asarray(errors, dtype=float)

    fig, ax = plt.subplots(figsize=(9, 5))
    by_seed = {}
    for row in paired_curve_rows:
        by_seed.setdefault(row["seed"], []).append(row)
    for seed_rows in by_seed.values():
        ordered = sorted(seed_rows, key=lambda row: row["iteration"])
        ax.plot(
            [row["iteration"] for row in ordered],
            [row["delta_exploitability_warm_minus_baseline"] for row in ordered],
            alpha=0.18,
            linewidth=1,
        )
    ax.plot(x, mean, linewidth=2, label="Mean paired difference")
    ax.fill_between(x, mean - se, mean + se, alpha=0.15)
    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.axvline(
        int(config["warm_start_boundary"]),
        linestyle=":",
        linewidth=1,
        label="Warm-start boundary",
    )
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Warm-start exploitability - baseline exploitability")
    ax.set_title("ESCHER warm-start ablation: paired exploitability difference")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        run_dir / "warm_start_paired_delta_exploitability_warm_minus_baseline.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def _plot_paired_summary(paired_rows: List[Dict[str, Any]], run_dir: Path) -> None:
    if not paired_rows:
        return
    metrics = [
        "delta_final_exploitability_warm_minus_baseline",
        "delta_best_exploitability_warm_minus_baseline",
        "delta_auc_nodes_warm_minus_baseline",
    ]
    labels = ["Final exploitability", "Best exploitability", "AUC by nodes"]
    stats_rows = [safe_stats([row[metric] for row in paired_rows]) for metric in metrics]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        labels,
        [row["mean"] for row in stats_rows],
        yerr=[0.0 if not np.isfinite(row["se"]) else row["se"] for row in stats_rows],
        capsize=4,
    )
    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_ylabel("Warm-start - baseline")
    ax.set_title("ESCHER warm-start ablation: mean paired differences")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(
        run_dir / "warm_start_paired_difference_summary_bar_chart.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def _plot_outputs(
    curve_rows: List[Dict[str, Any]],
    paired_curve_rows: List[Dict[str, Any]],
    paired_rows: List[Dict[str, Any]],
    config: Dict[str, Any],
    run_dir: Path,
) -> None:
    _plot_arm_curves(
        curve_rows,
        "exploitability",
        "Exploitability (NashConv/2)",
        "ESCHER warm-start ablation: exploitability by iteration",
        "warm_start_exploitability_by_iteration.png",
        run_dir,
        x_col="iteration",
    )
    _plot_arm_curves(
        curve_rows,
        "exploitability",
        "Exploitability (NashConv/2)",
        "ESCHER warm-start ablation: exploitability by nodes",
        "warm_start_exploitability_by_nodes.png",
        run_dir,
        x_col="nodes_touched",
    )
    _plot_arm_curves(
        curve_rows,
        "policy_value_error",
        "Absolute error from -1/18",
        "ESCHER warm-start ablation: policy-value error",
        "warm_start_policy_value_error_by_iteration.png",
        run_dir,
        x_col="iteration",
    )
    _plot_paired_curves(paired_curve_rows, config, run_dir)
    _plot_paired_summary(paired_rows, run_dir)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Kuhn poker ESCHER warm-start fair ablation."
    )
    parser.add_argument("--output-root", default="outputs", help="Root directory for outputs.")
    parser.add_argument("--seeds", default=None, help="Comma-separated seed list.")
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--warm-start-boundary", type=int, default=None)
    parser.add_argument("--traversals", type=int, default=None)
    parser.add_argument("--value-traversals", type=int, default=None)
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
    parser.add_argument("--save-full-checkpoints", type=_str2bool, default=SAVE_FULL_CHECKPOINTS)
    parser.add_argument("--restore-rng-state", type=_str2bool, default=RESTORE_RNG_STATE)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    config = build_config(args)
    seeds = parse_seeds(args.seeds)

    run_dir = create_run_dir(args.output_root, config["experiment_name"])
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _configure_logging(run_dir, args.verbose)
    _LOGGER.info("Export directory: %s", run_dir.resolve())
    _LOGGER.info("Running seeds: %s", seeds)
    _LOGGER.info("Warm-start boundary: %s", config["warm_start_boundary"])

    results = []
    failures = []
    for seed in tqdm(seeds, desc="ESCHER warm-start seeds"):
        for arm in [ARM_CONTINUOUS, ARM_WARM_START]:
            try:
                _LOGGER.info("Running arm=%s seed=%s", arm, seed)
                if arm == ARM_CONTINUOUS:
                    results.append(run_continuous_baseline(seed, config))
                else:
                    results.append(
                        run_warm_start_arm(
                            seed,
                            config,
                            checkpoint_dir,
                            save_checkpoint=args.save_full_checkpoints,
                            restore_rng_state=args.restore_rng_state,
                        )
                    )
            except Exception as exc:  # pragma: no cover - operational robustness
                _LOGGER.error("arm=%s seed=%s failed: %s", arm, seed, exc)
                failures.append({
                    "seed": int(seed),
                    "arm": arm,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                })

    summary_rows = [result["summary"] for result in results]
    aggregate_rows = _aggregate_by_arm(summary_rows)
    paired_rows = _paired_differences(summary_rows)
    paired_aggregate_rows = _aggregate_paired(paired_rows)
    curve_rows = _curve_rows(results)
    paired_curve_rows = _paired_curve_rows(curve_rows)

    _write_csv(run_dir / "seed_summary.csv", summary_rows, [
        "seed", "arm", "status", "final_exploitability",
        "final_exploitability_recomputed", "best_exploitability",
        "final_window_mean_exploitability", "final_policy_value_error",
        "final_policy_value_error_recomputed", "final_nodes_touched",
        "total_outer_wall_clock_seconds", "policy_training_events",
    ])
    _write_csv(run_dir / "aggregate_summary.csv", aggregate_rows, [
        "arm", "metric", "mean", "std", "se", "n_finite",
    ])
    _write_csv(run_dir / "paired_summary.csv", paired_rows, [
        "seed", "delta_final_exploitability_warm_minus_baseline",
        "delta_final_exploitability_recomputed_warm_minus_baseline",
        "delta_best_exploitability_warm_minus_baseline",
        "delta_final_window_mean_exploitability_warm_minus_baseline",
        "delta_auc_nodes_warm_minus_baseline",
        "delta_final_policy_value_error_warm_minus_baseline",
    ])
    _write_csv(run_dir / "paired_aggregate_summary.csv", paired_aggregate_rows, [
        "metric", "mean", "std", "se", "n_finite",
    ])
    _write_csv(run_dir / "checkpoint_curves.csv", curve_rows, [
        "seed", "arm", "checkpoint_index", "iteration", "nodes_touched",
        "wall_clock_seconds", "exploitability", "average_policy_value",
        "policy_value_error", "policy_loss", "value_loss", "value_test_loss",
        "regret_loss_player_0", "regret_loss_player_1",
    ])
    _write_csv(run_dir / "paired_checkpoint_differences.csv", paired_curve_rows, [
        "seed", "iteration", "delta_exploitability_warm_minus_baseline",
        "delta_policy_value_error_warm_minus_baseline",
    ])

    with open(run_dir / "aggregate_summary.json", "w", encoding="utf-8") as f:
        json.dump(json_safe({
            "by_arm": aggregate_rows,
            "paired_warm_start_minus_baseline": paired_aggregate_rows,
        }), f, indent=2)
    with open(run_dir / "experiment_metadata.json", "w", encoding="utf-8") as f:
        json.dump(json_safe({
            "experiment_name": config["experiment_name"],
            "config": config,
            "seeds": seeds,
            "arms": ARM_LABELS,
            "save_full_checkpoints": bool(args.save_full_checkpoints),
            "restore_rng_state": bool(args.restore_rng_state),
            "kuhn_game_value_player_0": KUHN_GAME_VALUE_PLAYER_0,
            "interpretation": (
                "paired fair ESCHER warm-start/checkpoint-resume ablation; "
                "positive warm-start minus baseline deltas mean warm-start is worse "
                "for error metrics"
            ),
            "tensorflow_version": tf.__version__,
            "pyspiel_version": getattr(pyspiel, "__version__", "unknown"),
        }), f, indent=2)

    if failures:
        with open(run_dir / "failed_runs.json", "w", encoding="utf-8") as f:
            json.dump(json_safe(failures), f, indent=2)

    if summary_rows:
        np.savez_compressed(
            run_dir / "warm_start_fair_ablation_curves.npz",
            seeds=np.asarray(seeds),
            seed_summary=np.asarray(json_safe(summary_rows), dtype=object),
            paired_summary=np.asarray(json_safe(paired_rows), dtype=object),
        )
    _plot_outputs(curve_rows, paired_curve_rows, paired_rows, config, run_dir)

    _LOGGER.info("Saved outputs to: %s", run_dir.resolve())
    return 0 if not failures else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

