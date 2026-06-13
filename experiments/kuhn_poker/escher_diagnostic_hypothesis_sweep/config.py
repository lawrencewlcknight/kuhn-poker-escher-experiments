"""Configuration for quick single-seed ESCHER diagnostic experiments."""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, List

from experiments.kuhn_poker.escher_multiseed_baseline.config import (
    DEFAULT_CONFIG as BASELINE_DEFAULT_CONFIG,
)

DEFAULT_SEED = 1234

DEFAULT_CONFIG = deepcopy(BASELINE_DEFAULT_CONFIG)
DEFAULT_CONFIG.update({
    "experiment_name": "kuhn_poker_escher_diagnostic_hypothesis_sweep",
    # Shorter common budget for quick diagnosis. Heavier author-style variants
    # override the per-iteration traversal/fitting budgets below.
    "num_iterations": 30,
    "check_exploitability_every": 10,
    "compute_exploitability": True,
    "importance_sampling": True,
    "zero_regret_fallback": "argmax",
    "all_actions": True,
    "expl": 1.0,
    "val_expl": 0.01,
})

BASELINE_POLICY_EVENTS = (
    len(range(0, DEFAULT_CONFIG["num_iterations"] + 1, DEFAULT_CONFIG["check_exploitability_every"]))
    + 1
)
BASELINE_MATCHED_FINAL_POLICY_STEPS = (
    BASELINE_POLICY_EVENTS * int(DEFAULT_CONFIG["policy_network_train_steps"])
)

AUTHOR_PUBLIC_KUHN_OVERRIDES = {
    "num_iterations": 30,
    "num_traversals": 500,
    "num_val_fn_traversals": 500,
    "policy_network_layers": (256, 128),
    "regret_network_layers": (256, 128),
    "value_network_layers": (256, 128),
    "batch_size_regret": 256,
    "batch_size_value": 256,
    "batch_size_average_policy": 10_000,
    "policy_network_train_steps": 1000,
    "regret_network_train_steps": 200,
    "value_network_train_steps": 200,
}

VARIANTS = [
    {
        "variant_id": "lightweight_current",
        "variant_label": "Current lightweight baseline",
        "variant_description": (
            "Current thesis baseline parameterisation on the shorter diagnostic "
            "iteration budget."
        ),
    },
    {
        "variant_id": "no_importance_sampling",
        "variant_label": "No importance sampling",
        "variant_description": (
            "Tests the paper-aligned ESCHER claim that regret targets should avoid "
            "importance-sampling corrections."
        ),
        "importance_sampling": False,
    },
    {
        "variant_id": "uniform_zero_regret_fallback",
        "variant_label": "Uniform zero-regret fallback",
        "variant_description": (
            "Uses standard uniform legal-action fallback when all positive regrets "
            "are zero, instead of the current deterministic argmax fallback."
        ),
        "zero_regret_fallback": "uniform",
    },
    {
        "variant_id": "no_is_uniform_fallback",
        "variant_label": "No IS plus uniform fallback",
        "variant_description": (
            "Combines the two most direct algorithmic suspicions: no importance "
            "sampling and uniform zero-regret fallback."
        ),
        "importance_sampling": False,
        "zero_regret_fallback": "uniform",
    },
    {
        "variant_id": "final_only_matched_policy_budget",
        "variant_label": "Final-only matched policy budget",
        "variant_description": (
            "Skips intermediate average-policy training and spends the matched "
            "total policy-step budget at final extraction."
        ),
        "compute_exploitability": False,
        "policy_network_train_steps": BASELINE_MATCHED_FINAL_POLICY_STEPS,
    },
    {
        "variant_id": "author_public_kuhn_budget",
        "variant_label": "Author-style Kuhn budget",
        "variant_description": (
            "Uses the larger public-repo Kuhn-style traversal, fitting, batch, and "
            "network sizes while preserving current importance sampling and "
            "argmax fallback behavior."
        ),
        **AUTHOR_PUBLIC_KUHN_OVERRIDES,
    },
    {
        "variant_id": "author_budget_no_is_uniform",
        "variant_label": "Author budget plus ESCHER fixes",
        "variant_description": (
            "Author-style Kuhn budget with no importance sampling and uniform "
            "zero-regret fallback."
        ),
        **AUTHOR_PUBLIC_KUHN_OVERRIDES,
        "importance_sampling": False,
        "zero_regret_fallback": "uniform",
    },
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

