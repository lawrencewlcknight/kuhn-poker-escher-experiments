"""Configuration for ESCHER policy-network width sweeps."""

from __future__ import annotations

from experiments.kuhn_poker.escher_architecture_base import (
    DEFAULT_SEED,
    make_default_config,
)

DEFAULT_CONFIG = make_default_config("kuhn_poker_escher_policy_network_width_sweep")


def _variant(variant_id, variant_label, layers, variant_description):
    return {
        "variant_id": variant_id,
        "variant_label": variant_label,
        "variant_description": variant_description,
        "policy_network_layers": layers,
        "regret_network_layers": (256, 128),
        "value_network_layers": (256, 128),
    }


VARIANTS = [
    _variant("policy_128_64", "Policy 128x64", (128, 64), "Lower-capacity average-policy net."),
    _variant("policy_256_128", "Policy 256x128", (256, 128), "Experiment 13 reference policy net."),
    _variant("policy_256_256", "Policy 256x256", (256, 256), "Removes the policy bottleneck."),
    _variant("policy_512_256", "Policy 512x256", (512, 256), "High-capacity policy net."),
    _variant("policy_256_256_128", "Policy 256x256x128", (256, 256, 128), "Deeper policy net."),
]

