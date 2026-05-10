"""Configuration for the ESCHER value-trajectory reuse ablation."""

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
    "experiment_name": "kuhn_poker_escher_reuse_value_trajectory_ablation",
    "value_test_traversals": 20,
    "bootstrap_value_with_separate_traversal": False,
})

REFERENCE_VARIANT_ID = "separate_value_traversals_baseline"
TREATMENT_VARIANT_ID = "reuse_regret_traversals_for_value"

VARIANTS = [
    {
        "variant_id": REFERENCE_VARIANT_ID,
        "variant_label": "Separate value traversals",
        "variant_description": (
            "Baseline ESCHER: dedicated value traversals populate the "
            "history-value memory before regret-network training."
        ),
        "reuse_regret_traversals_for_value": False,
    },
    {
        "variant_id": TREATMENT_VARIANT_ID,
        "variant_label": "Reuse regret traversals",
        "variant_description": (
            "Reuse player-0 regret traversals to populate history-value memory; "
            "no dedicated value-training traversal pass is used."
        ),
        "reuse_regret_traversals_for_value": True,
    },
]

DEFAULT_REUSE_VALUE_TRAJECTORY_SEEDS = DEVELOPMENT_SEEDS_5
THESIS_SEEDS_10 = DEFAULT_SEEDS
SMOKE_TEST_SEEDS = [1234]


def parse_seeds(seed_string: str | None, default: List[int] | None = None) -> List[int]:
    if not seed_string:
        return list(DEFAULT_REUSE_VALUE_TRAJECTORY_SEEDS if default is None else default)
    return [int(item.strip()) for item in seed_string.split(",") if item.strip()]


def make_variant_config(base_config: Dict[str, Any], variant: Dict[str, Any]) -> Dict[str, Any]:
    config = deepcopy(base_config)
    config.update(variant)
    config["variant_id"] = variant["variant_id"]
    config["variant_label"] = variant["variant_label"]
    config["variant_description"] = variant["variant_description"]
    config["reuse_regret_traversals_for_value"] = bool(
        variant["reuse_regret_traversals_for_value"]
    )
    return config
