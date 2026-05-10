"""Configuration for the ESCHER reach-probability weighting ablation."""

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
    "experiment_name": "kuhn_poker_escher_reach_weighting_ablation",
})

REFERENCE_VARIANT_ID = "iteration_only_baseline"
TREATMENT_VARIANT_ID = "iteration_plus_reach_weighted"

VARIANTS = [
    {
        "variant_id": REFERENCE_VARIANT_ID,
        "variant_label": "Iteration weighting only",
        "variant_description": (
            "ESCHER baseline: average-policy loss weighted by CFR iteration only."
        ),
        "use_reach_weighted_avg_policy_loss": False,
    },
    {
        "variant_id": TREATMENT_VARIANT_ID,
        "variant_label": "Iteration plus reach weighting",
        "variant_description": (
            "ESCHER with average-policy loss weighted by CFR iteration and "
            "acting-player reach probability."
        ),
        "use_reach_weighted_avg_policy_loss": True,
    },
]

DEFAULT_REACH_WEIGHTING_SEEDS = DEVELOPMENT_SEEDS_5
THESIS_SEEDS_10 = DEFAULT_SEEDS
SMOKE_TEST_SEEDS = [1234]


def parse_seeds(seed_string: str | None, default: List[int] | None = None) -> List[int]:
    if not seed_string:
        return list(DEFAULT_REACH_WEIGHTING_SEEDS if default is None else default)
    return [int(item.strip()) for item in seed_string.split(",") if item.strip()]


def make_variant_config(base_config: Dict[str, Any], variant: Dict[str, Any]) -> Dict[str, Any]:
    config = deepcopy(base_config)
    config.update(variant)
    config["variant_id"] = variant["variant_id"]
    config["variant_label"] = variant["variant_label"]
    config["variant_description"] = variant["variant_description"]
    config["use_reach_weighted_avg_policy_loss"] = bool(
        variant["use_reach_weighted_avg_policy_loss"]
    )
    return config
