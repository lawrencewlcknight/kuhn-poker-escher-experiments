"""Configuration for ESCHER bottleneck architecture sweeps."""

from __future__ import annotations

from experiments.kuhn_poker.escher_architecture_base import (
    DEFAULT_SEED,
    make_default_config,
)

DEFAULT_CONFIG = make_default_config("kuhn_poker_escher_bottleneck_architecture_sweep")


def _variant(variant_id, variant_label, layers, variant_description):
    return {
        "variant_id": variant_id,
        "variant_label": variant_label,
        "variant_description": variant_description,
        "policy_network_layers": layers,
        "regret_network_layers": layers,
        "value_network_layers": layers,
    }


VARIANTS = [
    _variant(
        "reference_256_128",
        "Reference 256x128",
        (256, 128),
        "Experiment 13 reference with a moderate 2:1 bottleneck.",
    ),
    _variant(
        "non_bottleneck_256_256",
        "Non-bottleneck 256x256",
        (256, 256),
        "Same first-layer width but no second-layer bottleneck.",
    ),
    _variant(
        "strong_bottleneck_256_64",
        "Strong bottleneck 256x64",
        (256, 64),
        "Tests whether compression regularises or discards useful signal.",
    ),
    _variant(
        "wide_bottleneck_512_128",
        "Wide bottleneck 512x128",
        (512, 128),
        "Larger first-layer expansion with the original final width.",
    ),
    _variant(
        "wide_non_bottleneck_512_512",
        "Wide non-bottleneck 512x512",
        (512, 512),
        "High-capacity non-bottleneck architecture.",
    ),
    _variant(
        "expanding_128_256",
        "Expanding 128x256",
        (128, 256),
        "Tests the opposite of a bottleneck: narrower first layer, wider final layer.",
    ),
]

