"""Configuration for the ESCHER on-policy joint-regret ablation."""

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
    "experiment_name": "kuhn_poker_escher_on_policy_joint_regret_ablation",
    "on_policy_joint_regret_updates": False,
})

REFERENCE_VARIANT_ID = "separate_player_regret_updates_baseline"
TREATMENT_VARIANT_ID = "on_policy_joint_regret_updates"

VARIANTS = [
    {
        "variant_id": REFERENCE_VARIANT_ID,
        "variant_label": "Separate player regret updates",
        "variant_description": (
            "Baseline ESCHER: each iteration collects separate regret traversal "
            "batches for player 0 and player 1."
        ),
        "on_policy_joint_regret_updates": False,
    },
    {
        "variant_id": TREATMENT_VARIANT_ID,
        "variant_label": "On-policy joint regret updates",
        "variant_description": (
            "Sample one batch of trajectories from the current joint regret-matching "
            "policy and write regret targets for the acting player at visited nodes."
        ),
        "on_policy_joint_regret_updates": True,
    },
]

DEFAULT_ON_POLICY_JOINT_REGRET_SEEDS = DEVELOPMENT_SEEDS_5
THESIS_SEEDS_10 = DEFAULT_SEEDS
SMOKE_TEST_SEEDS = [1234]


def parse_seeds(seed_string: str | None, default: List[int] | None = None) -> List[int]:
    if not seed_string:
        return list(DEFAULT_ON_POLICY_JOINT_REGRET_SEEDS if default is None else default)
    return [int(item.strip()) for item in seed_string.split(",") if item.strip()]


def make_variant_config(base_config: Dict[str, Any], variant: Dict[str, Any]) -> Dict[str, Any]:
    config = deepcopy(base_config)
    config.update(variant)
    config["variant_id"] = variant["variant_id"]
    config["variant_label"] = variant["variant_label"]
    config["variant_description"] = variant["variant_description"]
    config["on_policy_joint_regret_updates"] = bool(
        variant["on_policy_joint_regret_updates"]
    )
    return config
