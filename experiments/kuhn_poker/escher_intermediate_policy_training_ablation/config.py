"""Configuration for the ESCHER intermediate policy-training ablation."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List

from experiments.kuhn_poker.escher_multiseed_baseline.config import (
    DEFAULT_CONFIG as BASELINE_DEFAULT_CONFIG,
    DEFAULT_SEEDS,
    DEVELOPMENT_SEEDS_3,
)

DEFAULT_CONFIG = deepcopy(BASELINE_DEFAULT_CONFIG)
DEFAULT_CONFIG.update({
    "experiment_name": "kuhn_poker_escher_intermediate_policy_training_ablation",
})

REFERENCE_VARIANT_ID = "intermediate_every_5_baseline"

DEFAULT_VARIANT_IDS = [
    REFERENCE_VARIANT_ID,
    "final_only_1000_steps",
    "final_only_matched_steps",
]

SMOKE_TEST_SEEDS = [1234]
DEVELOPMENT_SEEDS = DEVELOPMENT_SEEDS_3


def count_intermediate_policy_events(config: Dict[str, Any]) -> int:
    """Count solver checkpoints that trigger intermediate policy training."""
    interval = int(config["check_exploitability_every"])
    if interval <= 0:
        raise ValueError("check_exploitability_every must be positive")
    return len(range(0, int(config["num_iterations"]) + 1, interval))


def baseline_total_policy_events(config: Dict[str, Any]) -> int:
    """Intermediate policy-training events plus final policy extraction."""
    return count_intermediate_policy_events(config) + 1


def matched_total_policy_steps(config: Dict[str, Any]) -> int:
    """Total policy-gradient steps used by the baseline arm."""
    return int(baseline_total_policy_events(config) * int(config["policy_network_train_steps"]))


def build_policy_training_variants(base_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build ablation variants from the possibly CLI-overridden base config."""
    base_steps = int(base_config["policy_network_train_steps"])
    interval = int(base_config["check_exploitability_every"])
    matched_steps = matched_total_policy_steps(base_config)
    return [
        {
            "variant_id": REFERENCE_VARIANT_ID,
            "label": f"Baseline: intermediate every {interval}",
            "compute_exploitability": True,
            "check_exploitability_every": int(base_config["check_exploitability_every"]),
            "policy_network_train_steps": base_steps,
            "description": (
                "Equivalent to the ESCHER baseline: train/evaluate a policy network at "
                "intermediate exploitability checkpoints and train once more at the end."
            ),
        },
        {
            "variant_id": "final_only_1000_steps",
            "label": f"Final only: {base_steps} steps",
            "compute_exploitability": False,
            "check_exploitability_every": int(base_config["check_exploitability_every"]),
            "policy_network_train_steps": base_steps,
            "description": (
                "Collect ESCHER regret/value data without intermediate policy training; "
                "train the policy network once at the end for the usual single-event budget."
            ),
        },
        {
            "variant_id": "final_only_matched_steps",
            "label": f"Final only: matched steps ({matched_steps})",
            "compute_exploitability": False,
            "check_exploitability_every": int(base_config["check_exploitability_every"]),
            "policy_network_train_steps": matched_steps,
            "description": (
                "Collect ESCHER regret/value data without intermediate policy training; "
                "train the policy network once at the end with the same total policy-gradient "
                "step budget as the baseline."
            ),
        },
    ]


def make_variant_config(base_config: Dict[str, Any], variant: Dict[str, Any]) -> Dict[str, Any]:
    """Merge a variant treatment into the common ESCHER config."""
    config = deepcopy(base_config)
    config.update({
        "variant_id": variant["variant_id"],
        "variant_label": variant["label"],
        "variant_description": variant["description"],
        "compute_exploitability": bool(variant["compute_exploitability"]),
        "check_exploitability_every": int(variant["check_exploitability_every"]),
        "policy_network_train_steps": int(variant["policy_network_train_steps"]),
    })

    intermediate_events = (
        count_intermediate_policy_events(config)
        if config["compute_exploitability"]
        else 0
    )
    config["intermediate_policy_training_events_expected"] = int(intermediate_events)
    config["final_policy_training_events_expected"] = 1
    config["total_policy_training_events_expected"] = int(intermediate_events + 1)
    config["policy_gradient_steps_expected"] = int(
        config["total_policy_training_events_expected"] * config["policy_network_train_steps"]
    )
    return config
