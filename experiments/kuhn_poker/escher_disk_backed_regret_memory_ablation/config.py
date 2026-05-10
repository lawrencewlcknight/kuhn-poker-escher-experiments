"""Configuration for the ESCHER disk-backed regret-memory ablation."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List

from experiments.kuhn_poker.escher_multiseed_baseline.config import (
    DEFAULT_CONFIG as BASELINE_DEFAULT_CONFIG,
    DEFAULT_SEEDS,
    DEVELOPMENT_SEEDS_5,
)

DEFAULT_CONFIG = deepcopy(BASELINE_DEFAULT_CONFIG)
DEFAULT_CONFIG.update({
    "experiment_name": "kuhn_poker_escher_disk_backed_regret_memory_ablation",
    "use_disk_average_policy_memory": True,
    "tfrecord_compression": None,
})

REFERENCE_VARIANT_ID = "in_memory_regret_baseline"
TREATMENT_VARIANT_ID = "disk_backed_regret_memory"

VARIANTS = [
    {
        "variant_id": REFERENCE_VARIANT_ID,
        "variant_label": "In-memory regret replay",
        "variant_description": (
            "Baseline ESCHER regret replay stored in ordinary in-memory "
            "reservoir buffers. Average-policy replay is disk-backed."
        ),
        "use_disk_regret_memory": False,
    },
    {
        "variant_id": TREATMENT_VARIANT_ID,
        "variant_label": "Disk-backed regret replay",
        "variant_description": (
            "Regret replay stored in TFRecord shards and streamed during "
            "regret-network training. Average-policy replay is disk-backed."
        ),
        "use_disk_regret_memory": True,
    },
]

DEFAULT_DISK_MEMORY_SEEDS = DEVELOPMENT_SEEDS_5
THESIS_SEEDS_10 = DEFAULT_SEEDS
SMOKE_TEST_SEEDS = [1234]


def parse_seeds(seed_string: str | None, default: List[int] | None = None) -> List[int]:
    if not seed_string:
        return list(DEFAULT_DISK_MEMORY_SEEDS if default is None else default)
    return [int(item.strip()) for item in seed_string.split(",") if item.strip()]


def make_variant_config(base_config: Dict[str, Any], variant: Dict[str, Any]) -> Dict[str, Any]:
    config = deepcopy(base_config)
    config.update(variant)
    config["variant_id"] = variant["variant_id"]
    config["variant_label"] = variant["variant_label"]
    config["variant_description"] = variant["variant_description"]
    config["use_disk_regret_memory"] = bool(variant["use_disk_regret_memory"])
    config["use_disk_average_policy_memory"] = bool(
        config.get("use_disk_average_policy_memory", True)
    )
    return config
