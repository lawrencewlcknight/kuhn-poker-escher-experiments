"""Configuration for single-seed ESCHER network-size sweeps."""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Tuple

from experiments.kuhn_poker.escher_author_budget_multiseed.config import (
    DEFAULT_CONFIG as AUTHOR_BUDGET_DEFAULT_CONFIG,
)

DEFAULT_SEED = 1234

DEFAULT_CONFIG = deepcopy(AUTHOR_BUDGET_DEFAULT_CONFIG)
DEFAULT_CONFIG.update({
    "experiment_name": "kuhn_poker_escher_network_size_sweep",
})


def _network_variant(
    variant_id: str,
    variant_label: str,
    layers: Tuple[int, ...],
    variant_description: str,
) -> Dict:
    return {
        "variant_id": variant_id,
        "variant_label": variant_label,
        "variant_description": variant_description,
        "policy_network_layers": layers,
        "regret_network_layers": layers,
        "value_network_layers": layers,
    }


VARIANTS = [
    _network_variant(
        "tiny_32_32",
        "Tiny 32x32",
        (32, 32),
        "Very small two-layer approximators; tests whether ESCHER is capacity-limited.",
    ),
    _network_variant(
        "lightweight_64_64",
        "Lightweight 64x64",
        (64, 64),
        "Experiment 1 network size under the revised Experiment 13 training protocol.",
    ),
    _network_variant(
        "narrow_128_64",
        "Narrow 128x64",
        (128, 64),
        "Midpoint between the lightweight baseline and Experiment 13 reference.",
    ),
    _network_variant(
        "balanced_128_128",
        "Balanced 128x128",
        (128, 128),
        "Tests width increase without a second-layer bottleneck.",
    ),
    _network_variant(
        "exp13_reference_256_128",
        "Experiment 13 reference 256x128",
        (256, 128),
        "Reference architecture from the best Experiment 12 configuration and Experiment 13.",
    ),
    _network_variant(
        "wide_256_256",
        "Wide 256x256",
        (256, 256),
        "Tests whether removing the 256-to-128 bottleneck improves exploitability.",
    ),
    _network_variant(
        "very_wide_512_256",
        "Very wide 512x256",
        (512, 256),
        "High-capacity two-layer approximators; tests for benefit versus instability or cost.",
    ),
    _network_variant(
        "shallow_256",
        "Shallow 256",
        (256,),
        "Single-hidden-layer network; tests whether retained width is enough without depth.",
    ),
    _network_variant(
        "deep_128_128_64",
        "Deep 128x128x64",
        (128, 128, 64),
        "Moderately deep architecture with similar scale to the reference.",
    ),
    _network_variant(
        "deep_256_256_128",
        "Deep 256x256x128",
        (256, 256, 128),
        "Deeper high-capacity architecture; tests whether extra depth improves the revised ESCHER setup.",
    ),
]


def parse_variant_ids(value: str | None) -> List[str]:
    if not value:
        return [variant["variant_id"] for variant in VARIANTS]
    return [item.strip() for item in value.split(",") if item.strip()]


def variant_lookup() -> Dict[str, Dict]:
    return {variant["variant_id"]: dict(variant) for variant in VARIANTS}


def make_variant_config(base_config: Dict, variant: Dict) -> Dict:
    config = deepcopy(base_config)
    config.update(variant)

    interval = int(config["check_exploitability_every"])
    intermediate_events = (
        len(range(0, int(config["num_iterations"]) + 1, interval))
        if bool(config["compute_exploitability"])
        else 0
    )
    final_events = 1
    config["intermediate_policy_training_events_expected"] = int(intermediate_events)
    config["final_policy_training_events_expected"] = int(final_events)
    config["total_policy_training_events_expected"] = int(intermediate_events + final_events)
    config["policy_gradient_steps_expected"] = int(
        config["total_policy_training_events_expected"]
        * int(config["policy_network_train_steps"])
    )
    return config
