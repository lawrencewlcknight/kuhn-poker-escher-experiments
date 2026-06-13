"""CLI entry point for quick ESCHER diagnostic hypothesis experiments."""

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
from typing import Dict, Iterable, List, Optional

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")
os.environ.setdefault("XDG_CACHE_HOME", str((Path("outputs") / ".cache").resolve()))
os.environ.setdefault("MPLCONFIGDIR", str((Path("outputs") / ".matplotlib_cache").resolve()))
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from tqdm import tqdm  # noqa: E402

from escher_poker.constants import (  # noqa: E402
    AVERAGE_POLICY_VALUE_TARGET_LABEL,
    KUHN_GAME_VALUE_PLAYER_0,
    NASH_EXPLOITABILITY_TARGET,
    NASH_EXPLOITABILITY_TARGET_LABEL,
)
from escher_poker.experiment_utils import (  # noqa: E402
    cleanup_tensorflow_memory,
    create_run_dir,
    json_safe,
    run_single_seed_variant,
)

from .config import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_SEED,
    VARIANTS,
    make_variant_config,
    parse_variant_ids,
    variant_lookup,
)

_LOGGER = logging.getLogger("escher_poker.experiment")

SUMMARY_HP_FIELDS = [
    "num_iterations",
    "num_traversals",
    "num_val_fn_traversals",
    "importance_sampling",
    "zero_regret_fallback",
    "all_actions",
    "expl",
    "val_expl",
    "policy_network_layers",
    "regret_network_layers",
    "value_network_layers",
    "batch_size_regret",
    "batch_size_value",
    "batch_size_average_policy",
    "policy_network_train_steps",
    "regret_network_train_steps",
    "value_network_train_steps",
    "reinitialize_regret_networks",
    "reinitialize_value_network",
]


def _str2bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"true", "t", "yes", "y", "1"}:
        return True
    if lowered in {"false", "f", "no", "n", "0"}:
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {value!r}")


def _parse_int_tuple(value: Optional[str]):
    if value is None:
        return None
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def build_base_config(args) -> Dict:
    config = deepcopy(DEFAULT_CONFIG)
    overrides = {
        "experiment_name": args.experiment_name,
        "num_iterations": args.iterations,
        "check_exploitability_every": args.evaluation_interval,
        "num_traversals": args.traversals,
        "num_val_fn_traversals": args.value_traversals,
        "policy_network_train_steps": args.policy_network_train_steps,
        "regret_network_train_steps": args.regret_network_train_steps,
        "value_network_train_steps": args.value_network_train_steps,
        "batch_size_regret": args.batch_size_regret,
        "batch_size_value": args.batch_size_value,
        "batch_size_average_policy": args.batch_size_average_policy,
        "policy_network_layers": _parse_int_tuple(args.policy_network_layers),
        "regret_network_layers": _parse_int_tuple(args.regret_network_layers),
        "value_network_layers": _parse_int_tuple(args.value_network_layers),
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


def _write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _add_config_columns(result: Dict, config: Dict) -> None:
    result["summary"]["variant_description"] = config.get("variant_description", "")
    for key in SUMMARY_HP_FIELDS:
        result["summary"][f"hp_{key}"] = json_safe(config.get(key))


def _plot_final_exploitability(run_dir: Path, summary_rows: List[Dict]) -> None:
    labels = [row["variant_id"] for row in summary_rows]
    y = [row["final_exploitability"] for row in summary_rows]
    y_intermediate = [row["intermediate_final_exploitability"] for row in summary_rows]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(max(9, 0.85 * len(labels)), 5))
    ax.bar(x - 0.18, y, width=0.36, label="Recomputed final policy")
    ax.bar(x + 0.18, y_intermediate, width=0.36, label="Last intermediate checkpoint")
    ax.axhline(
        NASH_EXPLOITABILITY_TARGET,
        color="black",
        linestyle="--",
        linewidth=1,
        label=NASH_EXPLOITABILITY_TARGET_LABEL,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Exploitability (NashConv / 2)")
    ax.set_title("Quick ESCHER diagnostic sweep: final exploitability")
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "final_exploitability_by_variant.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_curves(run_dir: Path, curve_rows: List[Dict]) -> None:
    intermediate_rows = [
        row for row in curve_rows if not bool(row.get("is_final_policy_evaluation", False))
    ]
    if not intermediate_rows:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    for variant_id in sorted({row["variant_id"] for row in intermediate_rows}):
        rows = [row for row in intermediate_rows if row["variant_id"] == variant_id]
        rows = sorted(rows, key=lambda row: row["iteration"])
        ax.plot(
            [row["iteration"] for row in rows],
            [row["exploitability"] for row in rows],
            marker="o",
            linewidth=1.8,
            label=variant_id,
        )
    ax.axhline(
        NASH_EXPLOITABILITY_TARGET,
        color="black",
        linestyle="--",
        linewidth=1,
        label=NASH_EXPLOITABILITY_TARGET_LABEL,
    )
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Exploitability (NashConv / 2)")
    ax.set_title("Intermediate exploitability checkpoints")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(run_dir / "intermediate_exploitability_by_iteration.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for variant_id in sorted({row["variant_id"] for row in intermediate_rows}):
        rows = [row for row in intermediate_rows if row["variant_id"] == variant_id]
        rows = sorted(rows, key=lambda row: row["iteration"])
        ax.plot(
            [row["iteration"] for row in rows],
            [row["average_policy_value"] for row in rows],
            marker="o",
            linewidth=1.8,
            label=variant_id,
        )
    ax.axhline(
        KUHN_GAME_VALUE_PLAYER_0,
        color="black",
        linestyle="--",
        linewidth=1,
        label=AVERAGE_POLICY_VALUE_TARGET_LABEL,
    )
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Average policy value")
    ax.set_title("Intermediate average-policy value checkpoints")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(run_dir / "average_policy_value_by_iteration.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run quick single-seed ESCHER diagnostic hypothesis experiments."
    )
    parser.add_argument("--output-root", default="outputs/diagnostic_sweeps")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--variant-ids",
        default=None,
        help="Comma-separated variant IDs. Defaults to all diagnostic variants.",
    )
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--evaluation-interval", type=int, default=None)
    parser.add_argument("--traversals", type=int, default=None)
    parser.add_argument("--value-traversals", type=int, default=None)
    parser.add_argument("--policy-network-train-steps", type=int, default=None)
    parser.add_argument("--regret-network-train-steps", type=int, default=None)
    parser.add_argument("--value-network-train-steps", type=int, default=None)
    parser.add_argument("--batch-size-regret", type=int, default=None)
    parser.add_argument("--batch-size-value", type=int, default=None)
    parser.add_argument("--batch-size-average-policy", type=int, default=None)
    parser.add_argument("--policy-network-layers", default=None)
    parser.add_argument("--regret-network-layers", default=None)
    parser.add_argument("--value-network-layers", default=None)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--continue-on-error", type=_str2bool, default=True)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    base_config = build_base_config(args)
    selected_ids = parse_variant_ids(args.variant_ids)
    lookup = variant_lookup()
    unknown = [variant_id for variant_id in selected_ids if variant_id not in lookup]
    if unknown:
        raise ValueError(f"Unknown diagnostic variant id(s): {unknown}")

    run_dir = create_run_dir(args.output_root, base_config["experiment_name"])
    _configure_logging(run_dir, args.verbose)

    selected_variants = [lookup[variant_id] for variant_id in selected_ids]
    metadata = {
        "base_config": base_config,
        "seed": int(args.seed),
        "selected_variant_ids": selected_ids,
        "available_variants": VARIANTS,
    }
    with open(run_dir / "experiment_metadata.json", "w", encoding="utf-8") as f:
        json.dump(json_safe(metadata), f, indent=2)

    _LOGGER.info("Export directory: %s", run_dir.resolve())
    _LOGGER.info("Running seed: %s", args.seed)
    _LOGGER.info("Selected variants: %s", selected_ids)

    results = []
    failures = []
    for variant in tqdm(selected_variants, desc="Diagnostic variants"):
        config = make_variant_config(base_config, variant)
        try:
            result = run_single_seed_variant(args.seed, config, export_dir=run_dir)
            _add_config_columns(result, config)
            results.append(result)
            _LOGGER.info(
                "%s final exploitability %.6f",
                config["variant_id"],
                result["summary"]["final_exploitability"],
            )
        except Exception as exc:  # pragma: no cover - operational robustness
            failure = {
                "variant_id": variant["variant_id"],
                "seed": int(args.seed),
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            failures.append(failure)
            _LOGGER.error("Variant %s failed: %s", variant["variant_id"], exc)
            if not args.continue_on_error:
                break
        finally:
            cleanup_tensorflow_memory()

    if failures:
        with open(run_dir / "failed_runs.json", "w", encoding="utf-8") as f:
            json.dump(json_safe(failures), f, indent=2)

    if not results:
        _LOGGER.error("No diagnostic variants completed successfully.")
        return 1

    summary_rows = [result["summary"] for result in results]
    curve_rows = [row for result in results for row in result["curves"]]
    _write_csv(run_dir / "variant_summary.csv", summary_rows)
    _write_csv(run_dir / "checkpoint_curves.csv", curve_rows)

    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(
            json_safe({
                "variant_summary": summary_rows,
                "failed_runs": failures,
            }),
            f,
            indent=2,
        )

    _plot_final_exploitability(run_dir, summary_rows)
    _plot_curves(run_dir, curve_rows)
    _LOGGER.info("Saved diagnostic outputs to: %s", run_dir.resolve())
    return 0 if not failures else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

