"""Shared experiment utilities for ESCHER runs."""

from __future__ import annotations

import csv
from datetime import datetime
import json
from pathlib import Path
import pickle
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from scipy import stats

import pyspiel
from open_spiel.python import policy
from open_spiel.python.algorithms import exploitability, expected_game_score

from .constants import DEFAULT_FINAL_WINDOW, KUHN_GAME_VALUE_PLAYER_0
from .seeding import set_seed_tf
from .solver import ESCHERSolver


def make_escher_solver(game, config: Dict[str, Any]) -> ESCHERSolver:
    """Construct an ``ESCHERSolver`` from a plain experiment config dict."""
    return ESCHERSolver(
        game,
        policy_network_layers=tuple(config["policy_network_layers"]),
        regret_network_layers=tuple(config["regret_network_layers"]),
        value_network_layers=tuple(config["value_network_layers"]),
        num_iterations=int(config["num_iterations"]),
        num_traversals=int(config["num_traversals"]),
        num_val_fn_traversals=int(config["num_val_fn_traversals"]),
        learning_rate=float(config["learning_rate"]),
        batch_size_regret=int(config["batch_size_regret"]),
        batch_size_value=int(config["batch_size_value"]),
        batch_size_average_policy=int(config["batch_size_average_policy"]),
        memory_capacity=int(config["memory_capacity"]),
        policy_network_train_steps=int(config["policy_network_train_steps"]),
        regret_network_train_steps=int(config["regret_network_train_steps"]),
        value_network_train_steps=int(config["value_network_train_steps"]),
        check_exploitability_every=int(config["check_exploitability_every"]),
        compute_exploitability=bool(config["compute_exploitability"]),
        reinitialize_regret_networks=bool(config["reinitialize_regret_networks"]),
        reinitialize_value_network=bool(config["reinitialize_value_network"]),
        save_policy_weights=bool(config["save_policy_weights"]),
        train_device=config["train_device"],
        infer_device=config["infer_device"],
        verbose=bool(config["verbose"]),
    )


def create_run_dir(output_root: str | Path, experiment_name: str) -> Path:
    """Create a timestamped run directory under ``output_root``."""
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_root) / f"{experiment_name}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def json_safe(value: Any) -> Any:
    """Convert common NumPy/scalar types to JSON-serialisable values."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        val = float(value)
        return None if not np.isfinite(val) else val
    if isinstance(value, np.ndarray):
        return [json_safe(x) for x in value.tolist()]
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(x) for x in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def first_nodes_to_threshold(nodes: Iterable[float], metric: Iterable[float], threshold: float) -> float:
    idx = np.where(np.asarray(metric) <= threshold)[0]
    return np.nan if len(idx) == 0 else float(np.asarray(nodes)[idx[0]])


def first_time_to_threshold(times: Iterable[float], metric: Iterable[float], threshold: float) -> float:
    idx = np.where(np.asarray(metric) <= threshold)[0]
    return np.nan if len(idx) == 0 else float(np.asarray(times)[idx[0]])


def final_window_mean(values: Iterable[float], window: int = DEFAULT_FINAL_WINDOW) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return np.nan
    return float(np.mean(values[-min(window, values.size):]))


def safe_stats(values: Iterable[float]) -> Dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"mean": np.nan, "std": np.nan, "se": np.nan, "n_finite": 0}
    return {
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
        "se": float(stats.sem(finite)) if finite.size > 1 else 0.0,
        "n_finite": int(finite.size),
    }


def run_single_seed(seed: int, config: Dict[str, Any], export_dir: Optional[str | Path] = None) -> Dict[str, Any]:
    """Run one ESCHER seed and return curves plus a compact summary."""
    set_seed_tf(seed)
    game = pyspiel.load_game(config["game_name"])
    solver = make_escher_solver(game, config)

    _regret_losses, _policy_loss, convs, nodes_touched, avg_policy_values, diagnostics = solver.solve()

    exploitability_curve = np.asarray(convs, dtype=np.float64) / 2.0
    nodes_touched = np.asarray(nodes_touched, dtype=np.float64)
    avg_policy_values = np.asarray(avg_policy_values, dtype=np.float64)
    value_error = np.abs(avg_policy_values - KUHN_GAME_VALUE_PLAYER_0)
    diagnostics = {k: np.asarray(v) for k, v in diagnostics.items()}
    iterations = diagnostics["iteration"].astype(int)
    wall_clock = diagnostics["wall_clock_seconds"].astype(float)

    final_policy = policy.tabular_policy_from_callable(game, solver.action_probabilities)
    final_nash_conv = exploitability.nash_conv(game, final_policy)
    final_policy_value = expected_game_score.policy_value(
        game.new_initial_state(), [final_policy] * game.num_players()
    )[0]

    summary = {
        "seed": int(seed),
        "final_exploitability": float(exploitability_curve[-1]),
        "best_exploitability": float(np.min(exploitability_curve)),
        "final_window_mean_exploitability": final_window_mean(exploitability_curve),
        "final_policy_value": float(final_policy_value),
        "final_policy_value_error": float(abs(final_policy_value - KUHN_GAME_VALUE_PLAYER_0)),
        "best_policy_value_error": float(np.min(value_error)),
        "final_nodes_touched": float(nodes_touched[-1]),
        "final_wall_clock_seconds": float(wall_clock[-1]),
        "nodes_to_exploitability_threshold": first_nodes_to_threshold(
            nodes_touched, exploitability_curve, config["exploitability_threshold"]
        ),
        "seconds_to_exploitability_threshold": first_time_to_threshold(
            wall_clock, exploitability_curve, config["exploitability_threshold"]
        ),
        "final_policy_loss": float(diagnostics["policy_loss"][-1]),
        "final_value_loss": float(diagnostics["value_loss"][-1]),
        "final_value_test_loss": float(diagnostics["value_test_loss"][-1]),
        "final_regret_loss_player_0": float(diagnostics["regret_loss_player_0"][-1]),
        "final_regret_loss_player_1": float(diagnostics["regret_loss_player_1"][-1]),
        "final_average_policy_buffer_size": int(diagnostics["average_policy_buffer_size"][-1]),
        "final_regret_buffer_size_player_0": int(diagnostics["regret_buffer_size_player_0"][-1]),
        "final_regret_buffer_size_player_1": int(diagnostics["regret_buffer_size_player_1"][-1]),
        "final_nash_conv_recomputed": float(final_nash_conv),
    }

    result = {
        "seed": int(seed),
        "iterations": iterations,
        "nodes_touched": nodes_touched,
        "wall_clock_seconds": wall_clock,
        "exploitability": exploitability_curve,
        "average_policy_value": avg_policy_values,
        "policy_value_error": value_error,
        "diagnostics": diagnostics,
        "summary": summary,
    }

    if bool(config.get("save_final_checkpoints", False)) and export_dir is not None:
        checkpoint_dir = Path(export_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        with open(checkpoint_dir / f"seed_{seed}_final_model.pkl", "wb") as f:
            pickle.dump(solver.extract_full_model(), f)

    return result


def export_metadata(run_dir: Path, config: Dict[str, Any], seeds: List[int]) -> None:
    metadata = {
        "config": config,
        "seeds": seeds,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(run_dir / "experiment_metadata.json", "w", encoding="utf-8") as f:
        json.dump(json_safe(metadata), f, indent=2)


def export_seed_summary(run_dir: Path, results: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary_rows = [result["summary"] for result in results]
    summary_csv = run_dir / "seed_summary.csv"
    fields = list(summary_rows[0].keys())
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    aggregate = {}
    for field in fields:
        if field == "seed":
            continue
        aggregate[field] = safe_stats([row[field] for row in summary_rows])

    with open(run_dir / "aggregate_summary.json", "w", encoding="utf-8") as f:
        json.dump(json_safe(aggregate), f, indent=2)
    return aggregate


def export_checkpoint_curves(run_dir: Path, results: List[Dict[str, Any]]) -> None:
    curve_csv = run_dir / "checkpoint_curves.csv"
    curve_fields = [
        "seed", "iteration", "nodes_touched", "wall_clock_seconds", "exploitability",
        "average_policy_value", "policy_value_error", "policy_loss", "value_loss",
        "value_test_loss", "regret_loss_player_0", "regret_loss_player_1",
        "average_policy_buffer_size", "regret_buffer_size_player_0",
        "regret_buffer_size_player_1", "value_buffer_size", "value_test_buffer_size",
    ]
    with open(curve_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=curve_fields)
        writer.writeheader()
        for result in results:
            diag = result["diagnostics"]
            for i, iteration in enumerate(result["iterations"]):
                writer.writerow({
                    "seed": result["seed"],
                    "iteration": int(iteration),
                    "nodes_touched": float(result["nodes_touched"][i]),
                    "wall_clock_seconds": float(result["wall_clock_seconds"][i]),
                    "exploitability": float(result["exploitability"][i]),
                    "average_policy_value": float(result["average_policy_value"][i]),
                    "policy_value_error": float(result["policy_value_error"][i]),
                    "policy_loss": float(diag["policy_loss"][i]),
                    "value_loss": float(diag["value_loss"][i]),
                    "value_test_loss": float(diag["value_test_loss"][i]),
                    "regret_loss_player_0": float(diag["regret_loss_player_0"][i]),
                    "regret_loss_player_1": float(diag["regret_loss_player_1"][i]),
                    "average_policy_buffer_size": int(diag["average_policy_buffer_size"][i]),
                    "regret_buffer_size_player_0": int(diag["regret_buffer_size_player_0"][i]),
                    "regret_buffer_size_player_1": int(diag["regret_buffer_size_player_1"][i]),
                    "value_buffer_size": int(diag["value_buffer_size"][i]),
                    "value_test_buffer_size": int(diag["value_test_buffer_size"][i]),
                })
