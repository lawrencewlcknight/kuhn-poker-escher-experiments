"""CLI entry point for the ESCHER checkpoint-stability experiment."""

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

import numpy as np  # noqa: E402
import pyspiel  # noqa: E402
import tensorflow as tf  # noqa: E402
from open_spiel.python import policy  # noqa: E402
from open_spiel.python.algorithms import exploitability, expected_game_score  # noqa: E402
from tqdm import tqdm  # noqa: E402

from escher_poker.checkpoint_analysis import analyze_checkpoint_snapshots  # noqa: E402
from escher_poker.checkpoint_plotting import (  # noqa: E402
    plot_checkpoint_head_to_head_outputs,
    plot_checkpoint_training_summary,
)
from escher_poker.constants import KUHN_GAME_VALUE_PLAYER_0  # noqa: E402
from escher_poker.experiment_utils import (  # noqa: E402
    create_run_dir,
    final_window_mean,
    json_safe,
    make_escher_solver,
)
from escher_poker.policy_snapshots import (  # noqa: E402
    discover_policy_snapshots,
    full_checkpoint_path,
    load_pickle,
    policy_snapshot_path,
    save_pickle,
    save_policy_snapshot,
)
from escher_poker.seeding import set_seed_tf  # noqa: E402

from .config import (  # noqa: E402
    ANNOTATE_HEATMAP,
    CHECKPOINT_SCHEDULE,
    DEFAULT_CONFIG,
    DEFAULT_SEEDS,
    EQUIVALENCE_EPSILON,
    FINAL_ITERATION,
    RUN_CHECKPOINTED_ARM,
    RUN_CONTINUOUS_BASELINE_ARM,
    SAVE_FULL_CHECKPOINTS,
    SAVE_POLICY_SNAPSHOTS,
    parse_checkpoint_schedule,
)

_LOGGER = logging.getLogger("escher_poker.experiment")


def parse_seeds(seed_string: Optional[str]) -> List[int]:
    if not seed_string:
        return list(DEFAULT_SEEDS)
    return [int(item.strip()) for item in seed_string.split(",") if item.strip()]


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


def solver_num_iterations_for_requested_updates(requested_updates: int) -> int:
    """ESCHERSolver.solve loops over ``range(num_iterations + 1)``."""
    requested_updates = int(requested_updates)
    if requested_updates < 1:
        raise ValueError("requested_updates must be at least 1.")
    return requested_updates - 1


def safe_last(values: Iterable[float]) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(values[-1]) if values.size else np.nan


def safe_min(values: Iterable[float]) -> float:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return float(np.min(finite)) if finite.size else np.nan


def compute_solver_policy_metrics(game, solver) -> Dict[str, float]:
    final_policy = policy.tabular_policy_from_callable(game, solver.action_probabilities)
    nash_conv = exploitability.nash_conv(game, final_policy)
    policy_value = expected_game_score.policy_value(
        game.new_initial_state(),
        [final_policy] * game.num_players(),
    )[0]
    return {
        "nash_conv_recomputed": float(nash_conv),
        "exploitability_recomputed": float(nash_conv / 2.0),
        "policy_value_recomputed": float(policy_value),
        "policy_value_error_recomputed": float(abs(policy_value - KUHN_GAME_VALUE_PLAYER_0)),
    }


def extract_stage_curves(
    seed: int,
    arm: str,
    target_iteration: int,
    result_tuple,
    diagnostics: Dict[str, Any],
) -> List[Dict[str, Any]]:
    _regret_losses, policy_loss, convs, nodes_touched, avg_policy_values, _diagnostics = result_tuple
    exploitability_curve = np.asarray(convs, dtype=np.float64) / 2.0
    nodes_curve = np.asarray(nodes_touched, dtype=np.float64)
    value_curve = np.asarray(avg_policy_values, dtype=np.float64)
    diag = {key: np.asarray(value) for key, value in diagnostics.items()}
    if "iteration" in diag and len(diag["iteration"]) == len(exploitability_curve):
        iterations = diag["iteration"].astype(int)
    else:
        iterations = np.arange(len(exploitability_curve), dtype=int)

    rows = []
    for idx in range(len(exploitability_curve)):
        row = {
            "seed": int(seed),
            "arm": arm,
            "target_checkpoint_iteration": int(target_iteration),
            "checkpoint_index": int(idx),
            "iteration_within_stage": int(iterations[idx]),
            "nodes_touched": float(nodes_curve[idx]) if idx < len(nodes_curve) else np.nan,
            "exploitability": float(exploitability_curve[idx]),
            "average_policy_value": float(value_curve[idx]) if idx < len(value_curve) else np.nan,
            "policy_value_error": (
                float(abs(value_curve[idx] - KUHN_GAME_VALUE_PLAYER_0))
                if idx < len(value_curve)
                else np.nan
            ),
            "policy_loss_final_stage": float(np.asarray(policy_loss)) if policy_loss is not None else np.nan,
        }
        for key, arr in diag.items():
            if len(arr) > idx:
                try:
                    row[key] = float(arr[idx])
                except Exception:
                    pass
        rows.append(row)
    return rows


def _run_checkpointed_arm(
    seed: int,
    config: Dict[str, Any],
    game,
    checkpoint_schedule: List[int],
    checkpoint_dir: Path,
    snapshot_dir: Path,
    save_full_checkpoints: bool,
    save_policy_snapshots: bool,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    all_curve_rows = []
    checkpoint_rows = []
    previous_target = 0
    previous_state = None

    for stage_idx, target in enumerate(checkpoint_schedule, start=1):
        additional_updates = int(target) - int(previous_target)
        _LOGGER.info(
            "Checkpointed arm seed=%s stage=%s target=%s additional_updates=%s",
            seed,
            stage_idx,
            target,
            additional_updates,
        )
        solver = make_escher_solver(
            game,
            config,
            num_iterations=solver_num_iterations_for_requested_updates(additional_updates),
        )
        if previous_state is not None:
            solver.load_full_model(previous_state)

        start_time = time.time()
        result_tuple = solver.solve()
        elapsed = time.time() - start_time
        _regret_losses, _policy_loss, convs, _nodes, avg_policy_values, diagnostics = result_tuple
        all_curve_rows.extend(
            extract_stage_curves(seed, "checkpointed", target, result_tuple, diagnostics)
        )

        snapshot_path = policy_snapshot_path(snapshot_dir, seed, target, "checkpointed")
        checkpoint_path = full_checkpoint_path(checkpoint_dir, seed, target, "checkpointed")
        metrics = compute_solver_policy_metrics(game, solver)
        row = {
            "seed": int(seed),
            "arm": "checkpointed",
            "checkpoint_iteration": int(target),
            "stage_index": int(stage_idx),
            "additional_updates": int(additional_updates),
            "solver_internal_iteration": int(getattr(solver, "_iteration", -1)),
            "nodes_visited": int(solver.get_num_nodes()),
            "stage_wall_clock_seconds": float(elapsed),
            "final_stage_exploitability": safe_last(np.asarray(convs, dtype=float) / 2.0),
            "best_stage_exploitability": safe_min(np.asarray(convs, dtype=float) / 2.0),
            "final_stage_policy_value": safe_last(avg_policy_values),
            "final_stage_policy_value_error": float(
                abs(safe_last(avg_policy_values) - KUHN_GAME_VALUE_PLAYER_0)
            ),
            "policy_snapshot_path": str(snapshot_path) if save_policy_snapshots else "",
            "full_checkpoint_path": str(checkpoint_path) if save_full_checkpoints else "",
            **metrics,
        }
        checkpoint_rows.append(row)

        if save_policy_snapshots:
            save_policy_snapshot(
                solver,
                snapshot_path,
                seed=seed,
                iteration=target,
                arm="checkpointed",
                config=config,
                stage_label=f"stage_{stage_idx}_to_{target}",
            )

        previous_state = solver.extract_full_model()
        if save_full_checkpoints:
            save_pickle(previous_state, checkpoint_path)

        previous_target = target
        del solver
        cleanup_memory()

    return all_curve_rows, checkpoint_rows


def _run_continuous_baseline_arm(
    seed: int,
    config: Dict[str, Any],
    game,
    final_iteration: int,
    checkpoint_dir: Path,
    snapshot_dir: Path,
    save_full_checkpoints: bool,
    save_policy_snapshots: bool,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    _LOGGER.info("Continuous baseline arm seed=%s final_iteration=%s", seed, final_iteration)
    solver = make_escher_solver(
        game,
        config,
        num_iterations=solver_num_iterations_for_requested_updates(final_iteration),
    )
    start_time = time.time()
    result_tuple = solver.solve()
    elapsed = time.time() - start_time
    _regret_losses, _policy_loss, convs, _nodes, avg_policy_values, diagnostics = result_tuple

    curve_rows = extract_stage_curves(
        seed,
        "continuous_baseline",
        final_iteration,
        result_tuple,
        diagnostics,
    )
    snapshot_path = policy_snapshot_path(snapshot_dir, seed, final_iteration, "continuous_baseline")
    checkpoint_path = full_checkpoint_path(checkpoint_dir, seed, final_iteration, "continuous_baseline")
    metrics = compute_solver_policy_metrics(game, solver)
    baseline_row = {
        "seed": int(seed),
        "arm": "continuous_baseline",
        "checkpoint_iteration": int(final_iteration),
        "solver_internal_iteration": int(getattr(solver, "_iteration", -1)),
        "nodes_visited": int(solver.get_num_nodes()),
        "wall_clock_seconds": float(elapsed),
        "final_exploitability": safe_last(np.asarray(convs, dtype=float) / 2.0),
        "best_exploitability": safe_min(np.asarray(convs, dtype=float) / 2.0),
        "final_window_mean_exploitability": final_window_mean(
            np.asarray(convs, dtype=float) / 2.0
        ),
        "final_policy_value": safe_last(avg_policy_values),
        "final_policy_value_error": float(
            abs(safe_last(avg_policy_values) - KUHN_GAME_VALUE_PLAYER_0)
        ),
        "policy_snapshot_path": str(snapshot_path) if save_policy_snapshots else "",
        "full_checkpoint_path": str(checkpoint_path) if save_full_checkpoints else "",
        **metrics,
    }

    if save_policy_snapshots:
        save_policy_snapshot(
            solver,
            snapshot_path,
            seed=seed,
            iteration=final_iteration,
            arm="continuous_baseline",
            config=config,
            stage_label="continuous_baseline_final",
        )
    if save_full_checkpoints:
        save_pickle(solver.extract_full_model(), checkpoint_path)

    del solver
    cleanup_memory()
    return curve_rows, baseline_row


def _write_metadata(
    run_dir: Path,
    config: Dict[str, Any],
    checkpoint_schedule: List[int],
    seeds: List[int],
    args,
) -> None:
    metadata = {
        "experiment_name": config["experiment_name"],
        "config": config,
        "checkpoint_schedule": checkpoint_schedule,
        "final_iteration": int(checkpoint_schedule[-1]),
        "seeds": seeds,
        "run_checkpointed_arm": bool(args.run_checkpointed_arm),
        "run_continuous_baseline_arm": bool(args.run_continuous_baseline_arm),
        "save_full_checkpoints": bool(args.save_full_checkpoints),
        "save_policy_snapshots": bool(args.save_policy_snapshots),
        "equivalence_epsilon": float(args.equivalence_epsilon),
        "kuhn_game_value_player_0": KUHN_GAME_VALUE_PLAYER_0,
        "tensorflow_version": tf.__version__,
        "pyspiel_version": getattr(pyspiel, "__version__", "unknown"),
    }
    with open(run_dir / "experiment_metadata.json", "w", encoding="utf-8") as f:
        json.dump(json_safe(metadata), f, indent=2)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Kuhn poker ESCHER checkpoint-stability experiment.")
    parser.add_argument("--output-root", default="outputs", help="Root directory for outputs.")
    parser.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated seed list. Defaults to 10 fixed seeds.",
    )
    parser.add_argument(
        "--checkpoint-schedule",
        default=None,
        help="Comma-separated checkpoint iterations. Defaults to the thesis schedule.",
    )
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
    parser.add_argument("--policy-network-layers", default=None, help="Comma-separated hidden sizes.")
    parser.add_argument("--regret-network-layers", default=None, help="Comma-separated hidden sizes.")
    parser.add_argument("--value-network-layers", default=None, help="Comma-separated hidden sizes.")
    parser.add_argument("--reinitialize-regret-networks", type=_str2bool, default=None)
    parser.add_argument("--reinitialize-value-network", type=_str2bool, default=None)
    parser.add_argument("--run-checkpointed-arm", type=_str2bool, default=RUN_CHECKPOINTED_ARM)
    parser.add_argument(
        "--run-continuous-baseline-arm",
        type=_str2bool,
        default=RUN_CONTINUOUS_BASELINE_ARM,
    )
    parser.add_argument("--save-full-checkpoints", type=_str2bool, default=SAVE_FULL_CHECKPOINTS)
    parser.add_argument("--save-policy-snapshots", type=_str2bool, default=SAVE_POLICY_SNAPSHOTS)
    parser.add_argument("--run-analysis", type=_str2bool, default=True)
    parser.add_argument("--equivalence-epsilon", type=float, default=EQUIVALENCE_EPSILON)
    parser.add_argument("--annotate-heatmap", type=_str2bool, default=ANNOTATE_HEATMAP)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    config = build_config(args)
    checkpoint_schedule = parse_checkpoint_schedule(args.checkpoint_schedule) or list(CHECKPOINT_SCHEDULE)
    final_iteration = int(checkpoint_schedule[-1])
    config["num_iterations"] = final_iteration
    seeds = parse_seeds(args.seeds)

    run_dir = create_run_dir(args.output_root, config["experiment_name"])
    checkpoint_dir = run_dir / "checkpoints"
    snapshot_dir = run_dir / "policy_snapshots"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    _configure_logging(run_dir, args.verbose)
    _write_metadata(run_dir, config, checkpoint_schedule, seeds, args)

    _LOGGER.info("Export directory: %s", run_dir.resolve())
    _LOGGER.info("Checkpoint schedule: %s", checkpoint_schedule)
    _LOGGER.info("Running seeds: %s", seeds)

    all_curve_rows = []
    checkpoint_rows = []
    baseline_rows = []
    failures = []

    for seed in tqdm(seeds, desc="ESCHER checkpoint-stability seeds"):
        try:
            if args.run_checkpointed_arm:
                set_seed_tf(seed)
                game = pyspiel.load_game(config["game_name"])
                curve_rows, stage_rows = _run_checkpointed_arm(
                    seed,
                    config,
                    game,
                    checkpoint_schedule,
                    checkpoint_dir,
                    snapshot_dir,
                    args.save_full_checkpoints,
                    args.save_policy_snapshots,
                )
                all_curve_rows.extend(curve_rows)
                checkpoint_rows.extend(stage_rows)

            if args.run_continuous_baseline_arm:
                set_seed_tf(seed)
                game = pyspiel.load_game(config["game_name"])
                curve_rows, baseline_row = _run_continuous_baseline_arm(
                    seed,
                    config,
                    game,
                    final_iteration,
                    checkpoint_dir,
                    snapshot_dir,
                    args.save_full_checkpoints,
                    args.save_policy_snapshots,
                )
                all_curve_rows.extend(curve_rows)
                baseline_rows.append(baseline_row)
        except Exception as exc:  # pragma: no cover - operational robustness
            _LOGGER.error("Seed %s failed: %s", seed, exc)
            failures.append({"seed": seed, "error": str(exc), "traceback": traceback.format_exc()})

    if failures:
        with open(run_dir / "failed_runs.json", "w", encoding="utf-8") as f:
            json.dump(failures, f, indent=2)

    _write_csv(run_dir / "checkpoint_training_curves.csv", all_curve_rows, [
        "seed", "arm", "target_checkpoint_iteration", "checkpoint_index",
        "iteration_within_stage", "nodes_touched", "exploitability",
        "average_policy_value", "policy_value_error", "wall_clock_seconds",
    ])
    _write_csv(run_dir / "checkpoint_stage_summary.csv", checkpoint_rows, [
        "seed", "arm", "checkpoint_iteration", "stage_index", "additional_updates",
        "exploitability_recomputed", "policy_value_error_recomputed",
        "nodes_visited", "stage_wall_clock_seconds", "policy_snapshot_path",
        "full_checkpoint_path",
    ])
    _write_csv(run_dir / "continuous_baseline_summary.csv", baseline_rows, [
        "seed", "arm", "checkpoint_iteration", "exploitability_recomputed",
        "policy_value_error_recomputed", "nodes_visited", "wall_clock_seconds",
        "policy_snapshot_path", "full_checkpoint_path",
    ])

    plot_checkpoint_training_summary(
        checkpoint_rows,
        baseline_rows,
        final_iteration,
        run_dir,
    )

    if args.run_analysis and args.save_policy_snapshots:
        snapshot_rows = discover_policy_snapshots(snapshot_dir)
        _write_csv(run_dir / "snapshot_inventory.csv", snapshot_rows, [
            "seed", "arm", "iteration", "path", "size_mb",
        ])
        if snapshot_rows:
            game = pyspiel.load_game(config["game_name"])
            analysis = analyze_checkpoint_snapshots(
                game,
                snapshot_rows,
                checkpoint_schedule,
                final_iteration,
                args.equivalence_epsilon,
            )
            _write_csv(run_dir / "loaded_policy_inventory.csv", analysis["loaded_policy_inventory"], [
                "seed", "arm", "checkpoint", "nodes_visited", "policy_layers",
                "input_size", "num_actions", "path",
            ])
            _write_csv(
                run_dir / "checkpoint_exploitability_metrics.csv",
                analysis["checkpoint_exploitability_metrics"],
                ["seed", "arm", "checkpoint", "nash_conv", "exploitability", "policy_value"],
            )
            _write_csv(run_dir / "head_to_head_exact_pairwise.csv", analysis["head_to_head_exact_pairwise"], [
                "seed", "checkpoint_A", "checkpoint_B", "A_EV_as_player0",
                "A_EV_as_player1", "A_EV_seat_averaged",
            ])
            _write_csv(
                run_dir / "head_to_head_exact_mean_matrix.csv",
                analysis["head_to_head_exact_mean_matrix"],
                ["checkpoint", *[str(item) for item in checkpoint_schedule]],
            )
            _write_csv(
                run_dir / "head_to_head_seed_win_fraction_matrix.csv",
                analysis["head_to_head_seed_win_fraction_matrix"],
                ["checkpoint", *[str(item) for item in checkpoint_schedule]],
            )
            _write_csv(
                run_dir / "head_to_head_monotonicity_summary_by_seed.csv",
                analysis["head_to_head_monotonicity_summary_by_seed"],
                ["seed", "num_later_vs_earlier_pairs", "mean_later_vs_earlier_ev"],
            )
            _write_csv(
                run_dir / "head_to_head_strength_with_metrics.csv",
                analysis["head_to_head_strength_with_metrics"],
                ["seed", "checkpoint", "mean_EV_vs_all_other_checkpoints", "exploitability"],
            )
            _write_csv(
                run_dir / "head_to_head_aggregate_strength_summary.csv",
                analysis["head_to_head_aggregate_strength_summary"],
                ["checkpoint", "mean_EV_vs_earlier_mean", "EV_vs_previous_mean"],
            )
            _write_csv(run_dir / "best_checkpoint_summary.csv", analysis["best_checkpoint_summary"], [
                "seed", "best_checkpoint_by_head_to_head",
                "best_checkpoint_by_exploitability", "final_checkpoint_exploitability",
            ])
            _write_csv(
                run_dir / "final_checkpoint_vs_continuous_baseline.csv",
                analysis["final_checkpoint_vs_continuous_baseline"],
                ["seed", "delta_exploitability_checkpointed_minus_baseline"],
            )
            with open(run_dir / "analysis_metadata.json", "w", encoding="utf-8") as f:
                json.dump(json_safe({
                    "snapshot_dir": str(snapshot_dir),
                    "output_dir": str(run_dir),
                    "checkpoint_schedule": checkpoint_schedule,
                    "final_iteration": final_iteration,
                    "equivalence_epsilon": args.equivalence_epsilon,
                    "seeds": sorted({int(row["seed"]) for row in snapshot_rows}),
                    "has_continuous_baseline": any(
                        row["arm"] == "continuous_baseline" for row in snapshot_rows
                    ),
                }), f, indent=2)
            plot_checkpoint_head_to_head_outputs(
                analysis,
                final_iteration,
                args.equivalence_epsilon,
                run_dir,
                annotate_heatmap=args.annotate_heatmap,
            )

    _LOGGER.info("Saved outputs to: %s", run_dir.resolve())
    return 0 if (checkpoint_rows or baseline_rows) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
