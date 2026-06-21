"""Configuration for ESCHER regret-network width sweeps."""

from __future__ import annotations

from experiments.kuhn_poker.escher_architecture_base import (
    DEFAULT_SEED,
    make_default_config,
)

DEFAULT_CONFIG = make_default_config("kuhn_poker_escher_regret_network_width_sweep")


def _variant(variant_id, variant_label, layers, variant_description):
    return {
        "variant_id": variant_id,
        "variant_label": variant_label,
        "variant_description": variant_description,
        "policy_network_layers": (256, 128),
        "regret_network_layers": layers,
        "value_network_layers": (256, 128),
    }


VARIANTS = [
    _variant("regret_128_64", "Regret 128x64", (128, 64), "Lower-capacity regret net."),
    _variant("regret_256_128", "Regret 256x128", (256, 128), "Experiment 13 reference regret net."),
    _variant("regret_256_256", "Regret 256x256", (256, 256), "Removes the regret bottleneck."),
    _variant("regret_512_256", "Regret 512x256", (512, 256), "High-capacity regret net."),
    _variant("regret_256_256_128", "Regret 256x256x128", (256, 256, 128), "Deeper regret net."),
]

