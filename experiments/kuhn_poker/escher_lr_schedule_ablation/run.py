"""CLI entry point for the ESCHER learning-rate schedule ablation."""

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
    BASELINE_SCHEDULE_ID,
    DEFAULT_CONFIG,
    OPTIONAL_EXTRA_SCHEDULES,
    SCHEDULE_CONFIGS,
    parse_schedule_ids,
    parse_seeds,
    schedule_lookup,
)

_LOGGER = logging.getLogger("escher_poker.experiment")

SUMMARY_METRICS = [
    "final_exploitability",
    "best_exploitability",
    "final_window_mean_exploitability",
    "final_window_std_exploitability",
    "exploitability_auc",
    "final_policy_value_error",
    "best_policy_value_error",
    "final_nodes_touched",
    "final_wall_clock_seconds",
    "total_outer_wall_clock_seconds",
    "final_learning_rate",
    "final_policy_loss",
    "final_value_loss",
    "final_value_test_loss",
    "final_regret_loss_player_0",
    "final_regret_loss_player_1",
]

PAIRED_METRICS = [
    "delta_final_exploitability",
    "delta_best_exploitability",
    "delta_final_window_mean_exploitability",
    "delta_exploitability_auc",
    "delta_final_policy_value_error",
    "delta_wall_clock_seconds",
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


def build_base_config(args) -> dict:
    config = deepcopy(DEFAULT_CONFIG)
    overrides = {
        "num_iterations": args.iterations,
        "num_traversals": args.traversals,
        "num_val_fn_traversals": args.value_traversals,
        "check_exploitability_every": args.evaluation_interval,
        "learning_rate": args.learning_rate,
        "learning_rate_end": args.learning_rate_end,
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


def build_schedule_configs(args, base_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    lookup = schedule_lookup(include_optional=args.include_optional_schedules)
    schedule_ids = parse_schedule_ids(
        args.schedules,
        include_optional=args.include_optional_schedules,
    )
    configs = []
    for schedule_id in schedule_ids:
        schedule_config = dict(lookup[schedule_id])
        if schedule_config["learning_rate_schedule"] != "constant":
            schedule_config["learning_rate_end"] = float(base_config["learning_rate_end"])
        else:
            schedule_config["learning_rate_end"] = float(base_config["learning_rate"])
        configs.append(schedule_config)
    return configs


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


def _final_window_std(values: Iterable[float], window: int = 5) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return np.nan
    window_values = values[-min(window, values.size):]
    return float(np.std(window_values, ddof=1)) if len(window_values) > 1 else 0.0


def _make_variant_config(
    base_config: Dict[str, Any],
    schedule_config: Dict[str, Any],
) -> Dict[str, Any]:
    config = dict(base_config)
    config.update(schedule_config)
    return config


def run_single_schedule_seed(
    schedule_config: Dict[str, Any],
    seed: int,
    base_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Run one schedule/seed pair and return curves plus summary rows."""
    config = _make_variant_config(base_config, schedule_config)
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
    learning_rate = diagnostics.get(
        "learning_rate",
        np.asarray([np.nan] * len(iterations), dtype=float),
    ).astype(float)

    final_policy = policy.tabular_policy_from_callable(game, solver.action_probabilities)
    final_nash_conv = exploitability.nash_conv(game, final_policy)
    final_policy_value = expected_game_score.policy_value(
        game.new_initial_state(),
        [final_policy] * game.num_players(),
    )[0]

    summary = {
        "schedule": schedule_config["schedule"],
        "schedule_label": schedule_config["schedule_label"],
        "learning_rate_schedule": schedule_config["learning_rate_schedule"],
        "seed": int(seed),
        "status": "completed",
        "final_exploitability": _safe_last(exploitability_curve),
        "best_exploitability": _safe_min(exploitability_curve),
        "final_window_mean_exploitability": final_window_mean(exploitability_curve),
        "final_window_std_exploitability": _final_window_std(exploitability_curve),
        "exploitability_auc": normalised_auc(iterations, exploitability_curve),
        "final_policy_value": float(final_policy_value),
        "final_policy_value_error": float(abs(final_policy_value - KUHN_GAME_VALUE_PLAYER_0)),
        "best_policy_value_error": _safe_min(value_error),
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
        "final_learning_rate": _safe_last(learning_rate),
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
        "final_value_buffer_size": _safe_last(diagnostics.get("value_buffer_size", [])),
        "final_nash_conv_recomputed": float(final_nash_conv),
        "learning_rate": float(config["learning_rate"]),
        "learning_rate_end": float(config["learning_rate_end"]),
        "learning_rate_decay_rate": float(config["learning_rate_decay_rate"]),
        "learning_rate_warmup_iterations": int(config["learning_rate_warmup_iterations"]),
    }

    curve_rows = []
    for idx, iteration in enumerate(iterations):
        row = {
            "schedule": schedule_config["schedule"],
            "schedule_label": schedule_config["schedule_label"],
            "learning_rate_schedule": schedule_config["learning_rate_schedule"],
            "seed": int(seed),
            "checkpoint_index": int(idx),
            "iteration": int(iteration),
            "nodes_touched": (
                float(nodes_touched[idx]) if idx < len(nodes_touched) else np.nan
            ),
            "wall_clock_seconds": (
                float(wall_clock[idx]) if idx < len(wall_clock) else np.nan
            ),
            "learning_rate": (
                float(learning_rate[idx]) if idx < len(learning_rate) else np.nan
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
        "schedule": schedule_config["schedule"],
        "schedule_label": schedule_config["schedule_label"],
        "learning_rate_schedule": schedule_config["learning_rate_schedule"],
        "seed": int(seed),
        "iterations": iterations,
        "learning_rate": learning_rate,
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


def _aggregate_by_schedule(summary_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for schedule in sorted({row["schedule"] for row in summary_rows}):
        schedule_rows = [row for row in summary_rows if row["schedule"] == schedule]
        label = schedule_rows[0].get("schedule_label", schedule)
        for metric in SUMMARY_METRICS:
            rows.append({
                "schedule": schedule,
                "schedule_label": label,
                "metric": metric,
                **safe_stats([row.get(metric, np.nan) for row in schedule_rows]),
            })
    return rows


def _paired_differences(summary_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_schedule_seed = {
        (row["schedule"], int(row["seed"])): row
        for row in summary_rows
    }
    baseline_seeds = {
        int(row["seed"])
        for row in summary_rows
        if row["schedule"] == BASELINE_SCHEDULE_ID
    }
    rows = []
    schedules = sorted({
        row["schedule"]
        for row in summary_rows
        if row["schedule"] != BASELINE_SCHEDULE_ID
    })
    for schedule in schedules:
        schedule_seeds = {
            int(row["seed"])
            for row in summary_rows
            if row["schedule"] == schedule
        }
        label = next(
            row.get("schedule_label", schedule)
            for row in summary_rows
            if row["schedule"] == schedule
        )
        for seed in sorted(baseline_seeds & schedule_seeds):
            baseline = by_schedule_seed[(BASELINE_SCHEDULE_ID, seed)]
            variant = by_schedule_seed[(schedule, seed)]
            rows.append({
                "schedule": schedule,
                "schedule_label": label,
                "seed": int(seed),
                "delta_final_exploitability": float(
                    variant["final_exploitability"] - baseline["final_exploitability"]
                ),
                "delta_best_exploitability": float(
                    variant["best_exploitability"] - baseline["best_exploitability"]
                ),
                "delta_final_window_mean_exploitability": float(
                    variant["final_window_mean_exploitability"]
                    - baseline["final_window_mean_exploitability"]
                ),
                "delta_exploitability_auc": float(
                    variant["exploitability_auc"] - baseline["exploitability_auc"]
                ),
                "delta_final_policy_value_error": float(
                    variant["final_policy_value_error"]
                    - baseline["final_policy_value_error"]
                ),
                "delta_wall_clock_seconds": float(
                    variant["final_wall_clock_seconds"]
                    - baseline["final_wall_clock_seconds"]
                ),
            })
    return rows


def _aggregate_paired(paired_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for schedule in sorted({row["schedule"] for row in paired_rows}):
        schedule_rows = [row for row in paired_rows if row["schedule"] == schedule]
        label = schedule_rows[0].get("schedule_label", schedule)
        for metric in PAIRED_METRICS:
            rows.append({
                "schedule": schedule,
                "schedule_label": label,
                "metric": metric,
                **safe_stats([row.get(metric, np.nan) for row in schedule_rows]),
            })
    return rows


def _curve_rows(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for result in results:
        rows.extend(result["curves"])
    return rows


def _schedule_order(schedule_configs: List[Dict[str, Any]]) -> List[str]:
    return [config["schedule"] for config in schedule_configs]


def _mean_curve(curve_rows: List[Dict[str, Any]], schedule: str, metric: str, x_col: str):
    grouped = {}
    for row in curve_rows:
        if row["schedule"] == schedule:
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


def _plot_curve_by_schedule(
    curve_rows: List[Dict[str, Any]],
    schedule_configs: List[Dict[str, Any]],
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
    labels = {config["schedule"]: config["schedule_label"] for config in schedule_configs}
    fig, ax = plt.subplots(figsize=(8, 5))
    for schedule in _schedule_order(schedule_configs):
        mean_rows = _mean_curve(curve_rows, schedule, metric, x_col)
        if not mean_rows:
            continue
        x = np.asarray([row["x_mean"] for row in mean_rows], dtype=float)
        mean = np.asarray([row["mean"] for row in mean_rows], dtype=float)
        se = np.asarray([row["se"] for row in mean_rows], dtype=float)
        ax.plot(x, mean, linewidth=2, label=labels[schedule])
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


def _plot_summary_bars(
    summary_rows: List[Dict[str, Any]],
    schedule_configs: List[Dict[str, Any]],
    metric: str,
    ylabel: str,
    title: str,
    filename: str,
    run_dir: Path,
) -> None:
    if not summary_rows:
        return
    labels = []
    means = []
    errors = []
    for config in schedule_configs:
        values = [
            row.get(metric, np.nan)
            for row in summary_rows
            if row["schedule"] == config["schedule"]
        ]
        if not values:
            continue
        stats = safe_stats(values)
        labels.append(config["schedule_label"])
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


def _plot_paired_bars(
    paired_rows: List[Dict[str, Any]],
    metric: str,
    ylabel: str,
    title: str,
    filename: str,
    run_dir: Path,
) -> None:
    if not paired_rows:
        return
    schedules = sorted({row["schedule"] for row in paired_rows})
    labels = []
    means = []
    errors = []
    for schedule in schedules:
        rows = [row for row in paired_rows if row["schedule"] == schedule]
        stats = safe_stats([row.get(metric, np.nan) for row in rows])
        labels.append(rows[0].get("schedule_label", schedule))
        means.append(stats["mean"])
        errors.append(0.0 if not np.isfinite(stats["se"]) else stats["se"])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, means, yerr=errors, capsize=4)
    ax.axhline(0.0, linewidth=1, linestyle="--")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(run_dir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_outputs(
    results: List[Dict[str, Any]],
    curve_rows: List[Dict[str, Any]],
    summary_rows: List[Dict[str, Any]],
    paired_rows: List[Dict[str, Any]],
    schedule_configs: List[Dict[str, Any]],
    run_dir: Path,
) -> None:
    if not results:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for config in schedule_configs:
        schedule_results = [
            result for result in results if result["schedule"] == config["schedule"]
        ]
        if not schedule_results:
            continue
        first_result = schedule_results[0]
        ax.plot(
            first_result["iterations"],
            first_result["learning_rate"],
            linewidth=2,
            label=config["schedule_label"],
        )
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Learning rate")
    ax.set_title("ESCHER learning-rate schedules")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "lr_schedule_learning_rates.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    _plot_curve_by_schedule(
        curve_rows,
        schedule_configs,
        "exploitability",
        "Exploitability (NashConv/2)",
        "ESCHER learning-rate ablation: exploitability",
        "lr_schedule_exploitability_by_iteration.png",
        run_dir,
    )
    _plot_curve_by_schedule(
        curve_rows,
        schedule_configs,
        "exploitability",
        "Exploitability (NashConv/2)",
        "ESCHER learning-rate ablation: exploitability by nodes",
        "lr_schedule_exploitability_by_nodes.png",
        run_dir,
        x_col="nodes_touched",
    )
    _plot_curve_by_schedule(
        curve_rows,
        schedule_configs,
        "policy_value_error",
        "Absolute error from -1/18",
        "ESCHER learning-rate ablation: policy-value error",
        "lr_schedule_policy_value_error_by_iteration.png",
        run_dir,
    )
    diagnostics = [
        (
            "policy_loss",
            "Policy MSE loss",
            "lr_schedule_policy_loss_diagnostic.png",
            "Policy-network loss",
        ),
        (
            "value_loss",
            "Value MSE loss",
            "lr_schedule_value_loss_diagnostic.png",
            "History-value loss",
        ),
        (
            "value_test_loss",
            "Value test MSE loss",
            "lr_schedule_value_test_loss_diagnostic.png",
            "History-value test loss",
        ),
        (
            "regret_loss_player_0",
            "Regret MSE loss",
            "lr_schedule_regret_loss_p0_diagnostic.png",
            "Regret loss: player 0",
        ),
        (
            "regret_loss_player_1",
            "Regret MSE loss",
            "lr_schedule_regret_loss_p1_diagnostic.png",
            "Regret loss: player 1",
        ),
    ]
    for key, ylabel, filename, title in diagnostics:
        _plot_curve_by_schedule(
            curve_rows,
            schedule_configs,
            key,
            ylabel,
            title,
            filename,
            run_dir,
        )

    _plot_summary_bars(
        summary_rows,
        schedule_configs,
        "final_exploitability",
        "Final exploitability",
        "ESCHER learning-rate ablation: final exploitability",
        "lr_schedule_final_exploitability_by_schedule.png",
        run_dir,
    )
    _plot_summary_bars(
        summary_rows,
        schedule_configs,
        "exploitability_auc",
        "Normalised exploitability AUC",
        "ESCHER learning-rate ablation: exploitability AUC",
        "lr_schedule_exploitability_auc_by_schedule.png",
        run_dir,
    )
    _plot_paired_bars(
        paired_rows,
        "delta_final_exploitability",
        "Delta final exploitability",
        "Paired final exploitability difference vs baseline",
        "lr_schedule_paired_delta_final_exploitability.png",
        run_dir,
    )
    _plot_paired_bars(
        paired_rows,
        "delta_exploitability_auc",
        "Delta exploitability AUC",
        "Paired exploitability AUC difference vs baseline",
        "lr_schedule_paired_delta_auc.png",
        run_dir,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Kuhn poker ESCHER learning-rate schedule ablation."
    )
    parser.add_argument("--output-root", default="outputs", help="Root directory for outputs.")
    parser.add_argument("--seeds", default=None, help="Comma-separated seed list.")
    parser.add_argument(
        "--schedules",
        default=None,
        help="Comma-separated schedule ids. Defaults to constant and cosine.",
    )
    parser.add_argument("--include-optional-schedules", type=_str2bool, default=False)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--traversals", type=int, default=None)
    parser.add_argument("--value-traversals", type=int, default=None)
    parser.add_argument("--evaluation-interval", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--learning-rate-end", type=float, default=None)
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
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    base_config = build_base_config(args)
    schedule_configs = build_schedule_configs(args, base_config)
    seeds = parse_seeds(args.seeds)

    run_dir = create_run_dir(args.output_root, base_config["experiment_name"])
    _configure_logging(run_dir, args.verbose)
    _LOGGER.info("Export directory: %s", run_dir.resolve())
    _LOGGER.info("Running seeds: %s", seeds)
    _LOGGER.info("Running schedules: %s", [cfg["schedule"] for cfg in schedule_configs])

    results = []
    failures = []
    for schedule_config in schedule_configs:
        for seed in tqdm(seeds, desc=schedule_config["schedule"]):
            try:
                results.append(run_single_schedule_seed(schedule_config, seed, base_config))
            except Exception as exc:  # pragma: no cover - operational robustness
                _LOGGER.error(
                    "schedule=%s seed=%s failed: %s",
                    schedule_config["schedule"],
                    seed,
                    exc,
                )
                failures.append({
                    "schedule": schedule_config["schedule"],
                    "seed": int(seed),
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                })

    summary_rows = [result["summary"] for result in results]
    curve_rows = _curve_rows(results)
    aggregate_rows = _aggregate_by_schedule(summary_rows)
    paired_rows = _paired_differences(summary_rows)
    paired_aggregate_rows = _aggregate_paired(paired_rows)

    _write_csv(run_dir / "seed_summary.csv", summary_rows, [
        "schedule", "seed", "status", "final_exploitability",
        "best_exploitability", "final_window_mean_exploitability",
        "exploitability_auc", "final_policy_value_error", "final_learning_rate",
    ])
    _write_csv(run_dir / "schedule_aggregate_summary.csv", aggregate_rows, [
        "schedule", "metric", "mean", "std", "se", "n_finite",
    ])
    _write_csv(run_dir / "paired_differences_vs_baseline.csv", paired_rows, [
        "schedule", "seed", "delta_final_exploitability",
        "delta_best_exploitability", "delta_final_window_mean_exploitability",
        "delta_exploitability_auc", "delta_final_policy_value_error",
    ])
    _write_csv(run_dir / "paired_difference_summary.csv", paired_aggregate_rows, [
        "schedule", "metric", "mean", "std", "se", "n_finite",
    ])
    _write_csv(run_dir / "checkpoint_curves.csv", curve_rows, [
        "schedule", "seed", "iteration", "nodes_touched", "wall_clock_seconds",
        "learning_rate", "exploitability", "average_policy_value",
        "policy_value_error", "policy_loss", "value_loss", "value_test_loss",
        "regret_loss_player_0", "regret_loss_player_1",
    ])

    with open(run_dir / "aggregate_summary.json", "w", encoding="utf-8") as f:
        json.dump(json_safe({
            "by_schedule": aggregate_rows,
            "paired_vs_baseline": paired_aggregate_rows,
        }), f, indent=2)
    with open(run_dir / "experiment_metadata.json", "w", encoding="utf-8") as f:
        json.dump(json_safe({
            "experiment_name": base_config["experiment_name"],
            "base_config": base_config,
            "schedule_configs": schedule_configs,
            "optional_extra_schedules": OPTIONAL_EXTRA_SCHEDULES,
            "baseline_schedule": BASELINE_SCHEDULE_ID,
            "seeds": seeds,
            "kuhn_game_value_player_0": KUHN_GAME_VALUE_PLAYER_0,
            "tensorflow_version": tf.__version__,
            "pyspiel_version": getattr(pyspiel, "__version__", "unknown"),
        }), f, indent=2)

    if failures:
        with open(run_dir / "failed_runs.json", "w", encoding="utf-8") as f:
            json.dump(json_safe(failures), f, indent=2)

    if results:
        np.savez_compressed(
            run_dir / "lr_schedule_curves.npz",
            seeds=np.asarray(seeds),
            schedules=np.asarray(_schedule_order(schedule_configs)),
            seed_summary=np.asarray(json_safe(summary_rows), dtype=object),
            checkpoint_curves=np.asarray(json_safe(curve_rows), dtype=object),
        )
    _plot_outputs(results, curve_rows, summary_rows, paired_rows, schedule_configs, run_dir)

    _LOGGER.info("Saved outputs to: %s", run_dir.resolve())
    return 0 if not failures else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
