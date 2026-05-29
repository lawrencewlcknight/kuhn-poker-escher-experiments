"""CLI entry point for the constrained ESCHER hyperparameter search."""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Any, Dict, Iterable, List, Optional, Union

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

from escher_poker.constants import (  # noqa: E402
    AVERAGE_POLICY_VALUE_TARGET_LABEL,
    KUHN_GAME_VALUE_PLAYER_0,
    NASH_EXPLOITABILITY_TARGET,
    NASH_EXPLOITABILITY_TARGET_LABEL,
)
from escher_poker.experiment_utils import create_run_dir, json_safe  # noqa: E402
from escher_poker.hyperparameter_search import (  # noqa: E402
    aggregate_summaries,
    config_subset,
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
        "clear_value_buffer": args.clear_value_buffer,
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


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_safe(payload), f, indent=2)


def _append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(json_safe(payload), sort_keys=True))
        f.write("\n")


def _safe_float(value: Any) -> float:
    if value is None:
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _stage_iterations(config: Dict[str, Any]) -> List[int]:
    num_iterations = int(config["num_iterations"])
    interval = max(1, int(config["check_exploitability_every"]))
    iterations = list(range(0, num_iterations + 1, interval))
    if not iterations or iterations[-1] != num_iterations:
        iterations.append(num_iterations)
    return iterations


def _failed_result(
    seed: int,
    config: Dict[str, Any],
    stage_name: str,
    error: str,
) -> Dict[str, Any]:
    iterations = _stage_iterations(config)
    n_points = len(iterations)
    summary = {
        "stage": stage_name,
        "variant_id": config["variant_id"],
        "seed": int(seed),
        "status": "failed",
        "error": error,
        "final_exploitability": np.nan,
        "best_exploitability": np.nan,
        "final_window_mean_exploitability": np.nan,
        "exploitability_auc": np.nan,
        "final_policy_value": np.nan,
        "final_policy_value_error": np.nan,
        "best_policy_value_error": np.nan,
        "final_nodes_touched": np.nan,
        "final_wall_clock_seconds": np.nan,
        "nodes_to_exploitability_threshold": np.nan,
        "seconds_to_exploitability_threshold": np.nan,
        "final_policy_loss": np.nan,
        "final_value_loss": np.nan,
        "final_value_test_loss": np.nan,
        "final_regret_loss_player_0": np.nan,
        "final_regret_loss_player_1": np.nan,
        "final_nash_conv_recomputed": np.nan,
    }
    summary.update({
        f"hp_{key}": json.dumps(value) if isinstance(value, list) else value
        for key, value in config_subset(config).items()
    })
    return {
        "stage": stage_name,
        "variant_id": config["variant_id"],
        "seed": int(seed),
        "config": dict(config),
        "iterations": iterations,
        "nodes_touched": [np.nan] * n_points,
        "wall_clock_seconds": [np.nan] * n_points,
        "exploitability": [np.nan] * n_points,
        "average_policy_value": [np.nan] * n_points,
        "policy_value_error": [np.nan] * n_points,
        "diagnostics": {},
        "summary": summary,
    }


def _normalise_failed_result_shape(result: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    if result.get("summary", {}).get("status") != "failed":
        return result
    iterations = _stage_iterations(config)
    n_points = len(iterations)
    result["iterations"] = iterations
    for key in [
        "nodes_touched",
        "wall_clock_seconds",
        "exploitability",
        "average_policy_value",
        "policy_value_error",
    ]:
        result[key] = [np.nan] * n_points
    result.setdefault("diagnostics", {})
    return result


def _append_partial_result(run_dir: Path, result: Dict[str, Any]) -> None:
    stage_name = result["stage"]
    _append_jsonl(run_dir / f"partial_{stage_name}_seed_summary.jsonl", result["summary"])
    for row in _curve_rows([result]):
        _append_jsonl(run_dir / "partial_checkpoint_curves.jsonl", row)


def _write_failures(run_dir: Path, failures: List[Dict[str, Any]]) -> None:
    _write_json(run_dir / "failed_runs.json", failures)


def _record_failure(
    run_dir: Path,
    failures: List[Dict[str, Any]],
    *,
    config: Dict[str, Any],
    seed: int,
    stage_name: str,
    error: str,
    traceback_text: str = "",
) -> None:
    failures.append({
        "stage": stage_name,
        "variant_id": config["variant_id"],
        "seed": int(seed),
        "error": error,
        "traceback": traceback_text,
    })
    _write_failures(run_dir, failures)


def _worker_stem(stage_name: str, variant_id: str, seed: int) -> str:
    return f"{stage_name}_{variant_id}_seed_{int(seed)}"


def _run_worker(
    worker_input_json: Union[str, Path],
    worker_output_json: Union[str, Path],
) -> int:
    with open(worker_input_json, "r", encoding="utf-8") as f:
        payload = json.load(f)
    result = run_single_hyperparameter_seed(
        int(payload["seed"]),
        payload["config"],
        str(payload["stage_name"]),
    )
    _write_json(Path(worker_output_json), result)
    return 0


def _run_hyperparameter_seed_subprocess(
    seed: int,
    config: Dict[str, Any],
    stage_name: str,
    run_dir: Path,
) -> Dict[str, Any]:
    stem = _worker_stem(stage_name, str(config["variant_id"]), seed)
    worker_input = run_dir / "worker_inputs" / f"{stem}.json"
    worker_output = run_dir / "worker_results" / f"{stem}.json"
    worker_log = run_dir / "worker_logs" / f"{stem}.log"
    _write_json(worker_input, {
        "seed": int(seed),
        "stage_name": stage_name,
        "config": config,
    })

    command = [
        sys.executable,
        "-m",
        "experiments.kuhn_poker.escher_constrained_hyperparameter_search.run",
        "--worker-input-json",
        str(worker_input),
        "--worker-output-json",
        str(worker_output),
    ]
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    worker_log.parent.mkdir(parents=True, exist_ok=True)
    with open(worker_log, "w", encoding="utf-8") as log_file:
        completed = subprocess.run(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Worker failed with exit code {completed.returncode}. "
            f"See {worker_log} for details."
        )
    if not worker_output.exists():
        raise RuntimeError(f"Worker completed without writing {worker_output}")
    with open(worker_output, "r", encoding="utf-8") as f:
        result = json.load(f)
    return _normalise_failed_result_shape(result, config)


def _run_hyperparameter_seed(
    seed: int,
    config: Dict[str, Any],
    stage_name: str,
    run_dir: Path,
    *,
    subprocess_isolation_enabled: bool,
) -> Dict[str, Any]:
    if subprocess_isolation_enabled:
        return _run_hyperparameter_seed_subprocess(seed, config, stage_name, run_dir)
    result = run_single_hyperparameter_seed(seed, config, stage_name)
    return _normalise_failed_result_shape(result, config)


def _run_stage(
    configs: List[Dict[str, Any]],
    seeds: List[int],
    stage_name: str,
    num_iterations: int,
    evaluation_interval: int,
    run_dir: Path,
    failures: List[Dict[str, Any]],
    *,
    subprocess_isolation_enabled: bool,
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
            try:
                result = _run_hyperparameter_seed(
                    seed,
                    stage_config,
                    stage_name,
                    run_dir,
                    subprocess_isolation_enabled=subprocess_isolation_enabled,
                )
            except Exception as exc:  # pragma: no cover - operational robustness
                error = str(exc)
                _LOGGER.error(
                    "variant=%s seed=%s stage=%s failed: %s",
                    stage_config["variant_id"],
                    seed,
                    stage_name,
                    error,
                )
                _record_failure(
                    run_dir,
                    failures,
                    config=stage_config,
                    seed=seed,
                    stage_name=stage_name,
                    error=error,
                    traceback_text=traceback.format_exc(),
                )
                result = _failed_result(seed, stage_config, stage_name, error)
            else:
                if result["summary"].get("status") != "completed":
                    _record_failure(
                        run_dir,
                        failures,
                        config=stage_config,
                        seed=seed,
                        stage_name=stage_name,
                        error=str(result["summary"].get("error", "solver reported failure")),
                    )
            stage_results.append(result)
            _append_partial_result(run_dir, result)
    return stage_results


def _curve_rows(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for result in results:
        diagnostics = result.get("diagnostics") or {}
        for idx, iteration in enumerate(result["iterations"]):
            row = {
                "stage": result["stage"],
                "variant_id": result["variant_id"],
                "seed": result["seed"],
                "iteration": int(iteration),
                "nodes_touched": (
                    _safe_float(result["nodes_touched"][idx])
                    if idx < len(result["nodes_touched"])
                    else np.nan
                ),
                "wall_clock_seconds": (
                    _safe_float(result["wall_clock_seconds"][idx])
                    if idx < len(result["wall_clock_seconds"])
                    else np.nan
                ),
                "exploitability": _safe_float(result["exploitability"][idx]),
                "average_policy_value": (
                    _safe_float(result["average_policy_value"][idx])
                    if idx < len(result["average_policy_value"])
                    else np.nan
                ),
                "policy_value_error": (
                    _safe_float(result["policy_value_error"][idx])
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
            ]:
                arr = diagnostics.get(key)
                if arr is not None and len(arr) > idx:
                    row[key] = _safe_float(arr[idx])
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
    ax.axhline(
        NASH_EXPLOITABILITY_TARGET,
        linestyle="--",
        label=NASH_EXPLOITABILITY_TARGET_LABEL,
    )
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Exploitability (NashConv/2)")
    ax.set_title(f"ESCHER Hyperparameter Search: {stage_name.title()} Exploitability")
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
    ax.axhline(
        NASH_EXPLOITABILITY_TARGET,
        linestyle="--",
        label=NASH_EXPLOITABILITY_TARGET_LABEL,
    )
    ax.set_xlabel("Nodes touched")
    ax.set_ylabel("Exploitability (NashConv/2)")
    ax.set_title(f"ESCHER Hyperparameter Search: {stage_name.title()} Exploitability by Nodes")
    ax.grid(True)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(run_dir / f"{stage_name}_exploitability_by_nodes.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for variant_id, variant_results in grouped.items():
        xs = variant_results[0]["iterations"]
        ymat = np.vstack([result["average_policy_value"] for result in variant_results])
        if np.all(~np.isfinite(ymat)):
            continue
        mean = np.nanmean(ymat, axis=0)
        ax.plot(xs, mean, linewidth=2, label=variant_id)
    ax.axhline(
        float(DEFAULT_CONFIG["average_policy_value_target"]),
        linestyle="--",
        label=AVERAGE_POLICY_VALUE_TARGET_LABEL,
    )
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Average policy value")
    ax.set_title(f"ESCHER Hyperparameter Search: {stage_name.title()} Average Policy Value")
    ax.grid(True)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(run_dir / f"{stage_name}_average_policy_value_by_iteration.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for variant_id, variant_results in grouped.items():
        xmat = np.vstack([result["nodes_touched"] for result in variant_results])
        ymat = np.vstack([result["average_policy_value"] for result in variant_results])
        if np.all(~np.isfinite(xmat)) or np.all(~np.isfinite(ymat)):
            continue
        ax.plot(np.nanmean(xmat, axis=0), np.nanmean(ymat, axis=0), linewidth=2, label=variant_id)
    ax.axhline(
        float(DEFAULT_CONFIG["average_policy_value_target"]),
        linestyle="--",
        label=AVERAGE_POLICY_VALUE_TARGET_LABEL,
    )
    ax.set_xlabel("Nodes touched")
    ax.set_ylabel("Average policy value")
    ax.set_title(f"ESCHER Hyperparameter Search: {stage_name.title()} Average Policy Value by Nodes")
    ax.grid(True)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(run_dir / f"{stage_name}_average_policy_value_by_nodes.png", dpi=200, bbox_inches="tight")
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
    if metric in {"final_exploitability", "best_exploitability", "final_window_mean_exploitability"}:
        ax.axhline(
            NASH_EXPLOITABILITY_TARGET,
            linestyle="--",
            linewidth=1,
            label=NASH_EXPLOITABILITY_TARGET_LABEL,
        )
        ax.legend(fontsize=8)
    if metric in {"final_policy_value", "best_policy_value", "final_window_mean_policy_value"}:
        ax.axhline(
            float(DEFAULT_CONFIG["average_policy_value_target"]),
            linestyle="--",
            linewidth=1,
            label=AVERAGE_POLICY_VALUE_TARGET_LABEL,
        )
        ax.legend(fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Variant")
    ax.set_title(f"ESCHER Hyperparameter Search: {stage_name.title()} {ylabel}")
    ax.grid(True, axis="y")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(run_dir / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_outputs(
    screening_results: List[Dict[str, Any]],
    confirmation_results: List[Dict[str, Any]],
    screening_summary_rows: List[Dict[str, Any]],
    confirmation_summary_rows: List[Dict[str, Any]],
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
        screening_summary_rows,
        "screening",
        "final_policy_value",
        "Final average policy value",
        "screening_final_average_policy_value_by_variant.png",
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
        "final_policy_value",
        "Final average policy value",
        "confirmation_final_average_policy_value_by_variant.png",
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


def _write_metadata(
    run_dir: Path,
    *,
    base_config: Dict[str, Any],
    screening_configs: List[Dict[str, Any]],
    confirmation_configs: List[Dict[str, Any]],
    selected_variant_ids: List[str],
    screening_seeds: List[int],
    confirmation_seeds: List[int],
    screening_iterations: int,
    screening_evaluation_interval: int,
    confirmation_iterations: int,
    confirmation_evaluation_interval: int,
    confirmation_top_k: int,
    random_search_seed: int,
    subprocess_isolation_enabled: bool,
    status: str,
) -> None:
    metadata = {
        "experiment_name": base_config["experiment_name"],
        "baseline_variant_id": BASELINE_VARIANT_ID,
        "baseline_config": base_config,
        "screening_iterations": screening_iterations,
        "screening_evaluation_interval": screening_evaluation_interval,
        "screening_seeds": screening_seeds,
        "confirmation_iterations": confirmation_iterations,
        "confirmation_evaluation_interval": confirmation_evaluation_interval,
        "confirmation_seeds": confirmation_seeds,
        "confirmation_top_k": confirmation_top_k,
        "selected_variant_ids": selected_variant_ids,
        "screening_configs": screening_configs,
        "confirmation_configs": confirmation_configs,
        "random_search_seed": random_search_seed,
        "subprocess_isolation_enabled": bool(subprocess_isolation_enabled),
        "incremental_outputs": {
            "partial_screening_seed_summary_jsonl": "partial_screening_seed_summary.jsonl",
            "partial_confirmation_seed_summary_jsonl": "partial_confirmation_seed_summary.jsonl",
            "partial_checkpoint_curves_jsonl": "partial_checkpoint_curves.jsonl",
            "worker_results_dir": "worker_results",
            "worker_logs_dir": "worker_logs",
        },
        "status": status,
        "kuhn_game_value_player_0": KUHN_GAME_VALUE_PLAYER_0,
        "tensorflow_version": tf.__version__,
        "pyspiel_version": getattr(pyspiel, "__version__", "unknown"),
    }
    _write_json(run_dir / "experiment_metadata.json", metadata)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Kuhn poker ESCHER constrained hyperparameter search."
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
    parser.add_argument("--clear-value-buffer", type=_str2bool, default=None)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    parser.add_argument(
        "--disable-subprocess-isolation",
        action="store_true",
        help=(
            "Run config/seed jobs in the parent process. By default each one runs "
            "in a fresh Python worker so TensorFlow memory is released."
        ),
    )
    parser.add_argument("--worker-input-json", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output-json", default=None, help=argparse.SUPPRESS)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    if args.worker_input_json or args.worker_output_json:
        if not args.worker_input_json or not args.worker_output_json:
            parser.error("--worker-input-json and --worker-output-json must be used together")
        return _run_worker(args.worker_input_json, args.worker_output_json)

    subprocess_isolation_enabled = not args.disable_subprocess_isolation
    base_config = build_base_config(args)
    screening_seeds = parse_seeds(args.screening_seeds, SCREENING_SEEDS)
    confirmation_seeds = parse_seeds(args.confirmation_seeds, CONFIRMATION_SEEDS)
    screening_configs = build_screening_configs(
        base_config,
        n_random_candidates=args.n_random_candidates,
        random_search_seed=args.random_search_seed,
    )

    run_dir = create_run_dir(args.output_root, base_config["experiment_name"])
    _configure_logging(run_dir, args.verbose)
    _write_metadata(
        run_dir,
        base_config=base_config,
        screening_configs=screening_configs,
        confirmation_configs=[],
        selected_variant_ids=[],
        screening_seeds=screening_seeds,
        confirmation_seeds=confirmation_seeds,
        screening_iterations=args.screening_iterations,
        screening_evaluation_interval=args.screening_evaluation_interval,
        confirmation_iterations=args.confirmation_iterations,
        confirmation_evaluation_interval=args.confirmation_evaluation_interval,
        confirmation_top_k=args.confirmation_top_k,
        random_search_seed=args.random_search_seed,
        subprocess_isolation_enabled=subprocess_isolation_enabled,
        status="running",
    )
    _LOGGER.info("Export directory: %s", run_dir.resolve())
    _LOGGER.info("Screening variants: %s", [config["variant_id"] for config in screening_configs])
    _LOGGER.info("Screening seeds: %s", screening_seeds)
    _LOGGER.info("Confirmation seeds: %s", confirmation_seeds)
    _LOGGER.info("Subprocess isolation enabled: %s", subprocess_isolation_enabled)

    failures: List[Dict[str, Any]] = []
    screening_results = _run_stage(
        screening_configs,
        screening_seeds,
        "screening",
        args.screening_iterations,
        args.screening_evaluation_interval,
        run_dir,
        failures,
        subprocess_isolation_enabled=subprocess_isolation_enabled,
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
            run_dir,
            failures,
            subprocess_isolation_enabled=subprocess_isolation_enabled,
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
        "final_policy_value",
    ])
    _write_csv(run_dir / "screening_aggregate_by_variant.csv", screening_aggregate_rows, [
        "variant_id", "final_exploitability_mean", "final_window_mean_exploitability_mean",
        "exploitability_auc_mean", "final_policy_value_mean", "n_runs", "n_completed",
    ])
    _write_csv(run_dir / "confirmation_seed_summary.csv", confirmation_summary_rows, [
        "stage", "variant_id", "seed", "status", "final_exploitability",
        "best_exploitability", "final_window_mean_exploitability", "exploitability_auc",
        "final_policy_value",
    ])
    _write_csv(run_dir / "confirmation_aggregate_by_variant.csv", confirmation_aggregate_rows, [
        "variant_id", "final_exploitability_mean", "final_window_mean_exploitability_mean",
        "exploitability_auc_mean", "final_policy_value_mean", "n_runs", "n_completed",
    ])
    _write_csv(run_dir / "confirmation_paired_differences_vs_baseline.csv", paired_rows, [
        "variant_id", "seed", "delta_final_exploitability",
        "delta_final_window_mean_exploitability", "delta_exploitability_auc",
        "delta_final_policy_value",
    ])
    _write_csv(run_dir / "confirmation_paired_difference_summary.csv", paired_aggregate_rows, [
        "variant_id", "delta_final_exploitability_mean",
        "delta_final_window_mean_exploitability_mean", "delta_exploitability_auc_mean",
        "delta_final_policy_value_mean",
    ])
    _write_csv(run_dir / "checkpoint_curves.csv", _curve_rows(screening_results + confirmation_results), [
        "stage", "variant_id", "seed", "iteration", "nodes_touched",
        "wall_clock_seconds", "exploitability", "average_policy_value", "policy_value_error",
    ])

    with open(run_dir / "aggregate_summary.json", "w", encoding="utf-8") as f:
        json.dump(json_safe({
            "screening": screening_aggregate_rows,
            "confirmation": confirmation_aggregate_rows,
            "paired_confirmation_vs_baseline": paired_aggregate_rows,
        }), f, indent=2)

    _write_metadata(
        run_dir,
        base_config=base_config,
        screening_configs=screening_configs,
        confirmation_configs=confirmation_configs,
        selected_variant_ids=selected_variant_ids,
        screening_seeds=screening_seeds,
        confirmation_seeds=confirmation_seeds,
        screening_iterations=args.screening_iterations,
        screening_evaluation_interval=args.screening_evaluation_interval,
        confirmation_iterations=args.confirmation_iterations,
        confirmation_evaluation_interval=args.confirmation_evaluation_interval,
        confirmation_top_k=args.confirmation_top_k,
        random_search_seed=args.random_search_seed,
        subprocess_isolation_enabled=subprocess_isolation_enabled,
        status="completed",
    )

    _plot_outputs(
        screening_results,
        confirmation_results,
        screening_summary_rows,
        confirmation_summary_rows,
        run_dir,
    )
    _LOGGER.info("Selected for confirmation: %s", selected_variant_ids)
    _LOGGER.info("Saved outputs to: %s", run_dir.resolve())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
