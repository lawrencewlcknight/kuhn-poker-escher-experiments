"""CLI entry point for the ESCHER intermediate policy-training ablation."""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import json
import logging
import os
from pathlib import Path
import sys
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
from tqdm import tqdm  # noqa: E402

from escher_poker.ablation_plotting import plot_policy_training_ablation  # noqa: E402
from escher_poker.constants import KUHN_GAME_VALUE_PLAYER_0  # noqa: E402
from escher_poker.experiment_utils import (  # noqa: E402
    create_run_dir,
    json_safe,
    safe_stats,
    run_single_seed_variant,
)

from .config import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_SEEDS,
    DEFAULT_VARIANT_IDS,
    DEVELOPMENT_SEEDS,
    REFERENCE_VARIANT_ID,
    SMOKE_TEST_SEEDS,
    baseline_total_policy_events,
    build_policy_training_variants,
    count_intermediate_policy_events,
    make_variant_config,
    matched_total_policy_steps,
)

_LOGGER = logging.getLogger("escher_poker.experiment")

METRICS_TO_SUMMARISE = [
    "final_exploitability",
    "final_policy_value_error",
    "intermediate_final_exploitability",
    "intermediate_best_exploitability",
    "intermediate_final_window_mean_exploitability",
    "intermediate_exploitability_normalised_auc_nodes",
    "elapsed_seconds",
    "final_wall_clock_seconds",
    "final_nodes_touched",
    "final_policy_loss",
    "policy_gradient_steps_expected",
]

PAIRED_METRICS = [
    "delta_final_exploitability_vs_baseline",
    "delta_final_policy_value_error_vs_baseline",
    "delta_elapsed_seconds_vs_baseline",
    "delta_policy_gradient_steps_vs_baseline",
]


def parse_seeds(seed_string: Optional[str]) -> List[int]:
    if not seed_string:
        return list(DEFAULT_SEEDS)
    return [int(item.strip()) for item in seed_string.split(",") if item.strip()]


def parse_variant_ids(variant_string: Optional[str]) -> List[str]:
    if not variant_string:
        return list(DEFAULT_VARIANT_IDS)
    return [item.strip() for item in variant_string.split(",") if item.strip()]


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
        "save_final_checkpoints": args.save_final_checkpoints,
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


def _summarise_by_variant(summary_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    aggregate_rows = []
    variant_ids = sorted({row["variant_id"] for row in summary_rows})
    for variant_id in variant_ids:
        variant_rows = [row for row in summary_rows if row["variant_id"] == variant_id]
        for metric in METRICS_TO_SUMMARISE:
            aggregate_rows.append({
                "variant_id": variant_id,
                "metric": metric,
                **safe_stats([row.get(metric, np.nan) for row in variant_rows]),
            })
    return aggregate_rows


def _aggregate_json(aggregate_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    aggregate = {}
    for row in aggregate_rows:
        aggregate.setdefault(row["variant_id"], {})[row["metric"]] = {
            "mean": row["mean"],
            "std": row["std"],
            "se": row["se"],
            "n_finite": row["n_finite"],
        }
    return aggregate


def _paired_differences(summary_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_variant_seed = {
        (row["variant_id"], int(row["seed"])): row
        for row in summary_rows
    }
    reference_seeds = {
        int(row["seed"])
        for row in summary_rows
        if row["variant_id"] == REFERENCE_VARIANT_ID
    }
    paired_rows = []
    variant_ids = sorted({
        row["variant_id"]
        for row in summary_rows
        if row["variant_id"] != REFERENCE_VARIANT_ID
    })
    for variant_id in variant_ids:
        variant_seeds = {
            int(row["seed"])
            for row in summary_rows
            if row["variant_id"] == variant_id
        }
        for seed in sorted(reference_seeds & variant_seeds):
            reference = by_variant_seed[(REFERENCE_VARIANT_ID, seed)]
            variant = by_variant_seed[(variant_id, seed)]
            paired_rows.append({
                "variant_id": variant_id,
                "seed": int(seed),
                "delta_final_exploitability_vs_baseline": float(
                    variant["final_exploitability"] - reference["final_exploitability"]
                ),
                "delta_final_policy_value_error_vs_baseline": float(
                    variant["final_policy_value_error"] - reference["final_policy_value_error"]
                ),
                "delta_elapsed_seconds_vs_baseline": float(
                    variant["elapsed_seconds"] - reference["elapsed_seconds"]
                ),
                "delta_policy_gradient_steps_vs_baseline": float(
                    variant["policy_gradient_steps_expected"]
                    - reference["policy_gradient_steps_expected"]
                ),
            })
    return paired_rows


def _paired_summary(paired_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for variant_id in sorted({row["variant_id"] for row in paired_rows}):
        variant_rows = [row for row in paired_rows if row["variant_id"] == variant_id]
        for metric in PAIRED_METRICS:
            rows.append({
                "variant_id": variant_id,
                "metric": metric,
                **safe_stats([row.get(metric, np.nan) for row in variant_rows]),
            })
    return rows


def _export_metadata(
    run_dir: Path,
    base_config: Dict[str, Any],
    variants: List[Dict[str, Any]],
    seeds: List[int],
) -> None:
    metadata = {
        "config": base_config,
        "policy_training_variants": variants,
        "reference_variant_id": REFERENCE_VARIANT_ID,
        "seeds": seeds,
        "smoke_test_seeds": SMOKE_TEST_SEEDS,
        "development_seeds": DEVELOPMENT_SEEDS,
        "kuhn_game_value_player_0": KUHN_GAME_VALUE_PLAYER_0,
        "tensorflow_version": tf.__version__,
        "pyspiel_version": getattr(pyspiel, "__version__", "unknown"),
        "intermediate_policy_events": count_intermediate_policy_events(base_config),
        "baseline_total_policy_events": baseline_total_policy_events(base_config),
        "matched_total_policy_steps": matched_total_policy_steps(base_config),
        "experiment_note": (
            "Controlled ablation comparing the ESCHER baseline's intermediate "
            "policy-network training/evaluation path with final-only average-policy "
            "extraction variants."
        ),
    }
    with open(run_dir / "experiment_metadata.json", "w", encoding="utf-8") as f:
        json.dump(json_safe(metadata), f, indent=2)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Kuhn poker ESCHER intermediate policy-training ablation."
    )
    parser.add_argument("--output-root", default="outputs", help="Root directory for outputs.")
    parser.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated seed list. Defaults to 10 fixed seeds.",
    )
    parser.add_argument(
        "--variant-ids",
        default=None,
        help=(
            "Comma-separated variants to run. Options: "
            f"{', '.join(DEFAULT_VARIANT_IDS)}. Defaults to all variants."
        ),
    )
    parser.add_argument("--iterations", type=int, default=None)
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
    parser.add_argument(
        "--policy-network-layers",
        default=None,
        help="Comma-separated hidden sizes, e.g. 256,128",
    )
    parser.add_argument(
        "--regret-network-layers",
        default=None,
        help="Comma-separated hidden sizes, e.g. 256,128",
    )
    parser.add_argument(
        "--value-network-layers",
        default=None,
        help="Comma-separated hidden sizes, e.g. 256,128",
    )
    parser.add_argument("--reinitialize-regret-networks", type=_str2bool, default=None)
    parser.add_argument("--reinitialize-value-network", type=_str2bool, default=None)
    parser.add_argument("--save-final-checkpoints", type=_str2bool, default=None)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    base_config = build_config(args)
    all_variants = build_policy_training_variants(base_config)
    variant_ids = parse_variant_ids(args.variant_ids)
    unknown_variants = sorted(
        set(variant_ids) - {variant["variant_id"] for variant in all_variants}
    )
    if unknown_variants:
        raise ValueError(f"Unknown variant id(s): {unknown_variants}")
    variants = [variant for variant in all_variants if variant["variant_id"] in set(variant_ids)]
    seeds = parse_seeds(args.seeds)

    run_dir = create_run_dir(args.output_root, base_config["experiment_name"])
    _configure_logging(run_dir, args.verbose)
    _export_metadata(run_dir, base_config, variants, seeds)

    _LOGGER.info("Export directory: %s", run_dir.resolve())
    _LOGGER.info("Running seeds: %s", seeds)
    _LOGGER.info("Running variants: %s", [variant["variant_id"] for variant in variants])
    _LOGGER.info("Base config: %s", base_config)

    results = []
    failures = []
    for variant in variants:
        config = make_variant_config(base_config, variant)
        _LOGGER.info(
            "Running %s (%s expected policy-gradient steps)",
            config["variant_id"],
            config["policy_gradient_steps_expected"],
        )
        for seed in tqdm(seeds, desc=config["variant_id"]):
            try:
                results.append(run_single_seed_variant(seed, config, export_dir=run_dir))
            except Exception as exc:  # pragma: no cover - operational robustness
                _LOGGER.error("Variant %s seed %s failed: %s", config["variant_id"], seed, exc)
                failures.append({
                    "variant_id": config["variant_id"],
                    "seed": seed,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                })

    if failures:
        with open(run_dir / "failed_runs.json", "w", encoding="utf-8") as f:
            json.dump(failures, f, indent=2)

    if not results:
        _LOGGER.error("No variant/seed runs completed successfully.")
        return 1

    summary_rows = [result["summary"] for result in results]
    curve_rows = [row for result in results for row in result["curves"]]
    aggregate_rows = _summarise_by_variant(summary_rows)
    paired_rows = _paired_differences(summary_rows)
    paired_summary_rows = _paired_summary(paired_rows)

    _write_csv(run_dir / "seed_summary.csv", summary_rows, [
        "variant_id", "variant_label", "seed", "final_exploitability",
        "final_policy_value_error", "final_policy_value", "final_nash_conv_recomputed",
        "intermediate_final_exploitability", "intermediate_best_exploitability",
        "intermediate_final_window_mean_exploitability", "elapsed_seconds",
        "final_wall_clock_seconds", "final_nodes_touched", "final_policy_loss",
        "policy_gradient_steps_expected",
    ])
    _write_csv(run_dir / "checkpoint_curves.csv", curve_rows, [
        "variant_id", "variant_label", "seed", "checkpoint_index", "iteration",
        "nodes_touched", "wall_clock_seconds", "exploitability", "average_policy_value",
        "policy_value_error", "is_final_policy_evaluation", "policy_loss", "value_loss",
        "value_test_loss", "regret_loss_player_0", "regret_loss_player_1",
        "average_policy_buffer_size", "regret_buffer_size_player_0",
        "regret_buffer_size_player_1", "value_buffer_size", "value_test_buffer_size",
    ])
    _write_csv(run_dir / "variant_aggregate_summary.csv", aggregate_rows, [
        "variant_id", "metric", "mean", "std", "se", "n_finite",
    ])
    _write_csv(run_dir / "paired_differences_vs_baseline.csv", paired_rows, [
        "variant_id", "seed", *PAIRED_METRICS,
    ])
    _write_csv(run_dir / "paired_difference_summary.csv", paired_summary_rows, [
        "variant_id", "metric", "mean", "std", "se", "n_finite",
    ])

    with open(run_dir / "aggregate_summary.json", "w", encoding="utf-8") as f:
        json.dump(json_safe(_aggregate_json(aggregate_rows)), f, indent=2)
    with open(run_dir / "paired_difference_summary.json", "w", encoding="utf-8") as f:
        json.dump(json_safe(_aggregate_json(paired_summary_rows)), f, indent=2)

    plot_policy_training_ablation(
        summary_rows,
        curve_rows,
        paired_rows,
        variants,
        REFERENCE_VARIANT_ID,
        run_dir,
    )

    _LOGGER.info("Saved all outputs to: %s", run_dir.resolve())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
