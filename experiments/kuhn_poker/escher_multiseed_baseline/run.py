"""CLI entry point for the aligned Kuhn poker ESCHER baseline experiment."""

from __future__ import annotations

import argparse
from copy import deepcopy
import logging
import os
from pathlib import Path
import sys
import traceback
from typing import List, Optional

# Keep execution CPU-only by default for reproducibility unless explicitly overridden.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")

from tqdm import tqdm  # noqa: E402

from escher_poker.experiment_utils import (  # noqa: E402
    create_run_dir,
    export_checkpoint_curves,
    export_metadata,
    export_seed_summary,
    run_single_seed,
)
from escher_poker.plotting import plot_diagnostics, plot_multiseed_results  # noqa: E402

from .config import DEFAULT_CONFIG, DEFAULT_SEEDS  # noqa: E402

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
        "compute_exploitability": args.compute_exploitability,
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


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the aligned Kuhn poker ESCHER multi-seed baseline.")
    parser.add_argument("--output-root", default="outputs", help="Root directory for outputs.")
    parser.add_argument("--seeds", default=None, help="Comma-separated seed list. Defaults to 10 fixed seeds.")
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
    parser.add_argument("--policy-network-layers", default=None, help="Comma-separated hidden sizes, e.g. 256,128")
    parser.add_argument("--regret-network-layers", default=None, help="Comma-separated hidden sizes, e.g. 256,128")
    parser.add_argument("--value-network-layers", default=None, help="Comma-separated hidden sizes, e.g. 256,128")
    parser.add_argument("--reinitialize-regret-networks", type=_str2bool, default=None)
    parser.add_argument("--reinitialize-value-network", type=_str2bool, default=None)
    parser.add_argument("--compute-exploitability", type=_str2bool, default=None)
    parser.add_argument("--save-final-checkpoints", type=_str2bool, default=None)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    config = build_config(args)
    seeds = parse_seeds(args.seeds)
    run_dir = create_run_dir(args.output_root, config["experiment_name"])
    _configure_logging(run_dir, args.verbose)
    export_metadata(run_dir, config, seeds)

    _LOGGER.info("Export directory: %s", run_dir.resolve())
    _LOGGER.info("Running seeds: %s", seeds)
    _LOGGER.info("Config: %s", config)

    results = []
    failures = []
    for seed in tqdm(seeds, desc="ESCHER seeds"):
        try:
            results.append(run_single_seed(seed, config, export_dir=run_dir))
        except Exception as exc:  # pragma: no cover - operational robustness
            _LOGGER.error("Seed %s failed: %s", seed, exc)
            failures.append({"seed": seed, "error": str(exc), "traceback": traceback.format_exc()})

    if failures:
        import json
        with open(run_dir / "failed_seeds.json", "w", encoding="utf-8") as f:
            json.dump(failures, f, indent=2)

    if not results:
        _LOGGER.error("No seeds completed successfully.")
        return 1

    aggregate = export_seed_summary(run_dir, results)
    export_checkpoint_curves(run_dir, results)
    plot_multiseed_results(results, run_dir)
    plot_diagnostics(results, run_dir)

    _LOGGER.info("Aggregate final exploitability: %s", aggregate.get("final_exploitability"))
    _LOGGER.info("Aggregate best exploitability: %s", aggregate.get("best_exploitability"))
    _LOGGER.info("Saved all outputs to: %s", run_dir.resolve())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
