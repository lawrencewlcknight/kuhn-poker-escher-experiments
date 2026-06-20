from experiments.kuhn_poker.escher_network_size_sweep.config import (
    DEFAULT_CONFIG,
    DEFAULT_SEED,
    VARIANTS,
    make_variant_config,
)


def test_network_size_sweep_defaults_match_experiment_13_protocol():
    assert DEFAULT_SEED == 1234
    assert DEFAULT_CONFIG["num_iterations"] == 80
    assert DEFAULT_CONFIG["num_traversals"] == 500
    assert DEFAULT_CONFIG["num_val_fn_traversals"] == 500
    assert DEFAULT_CONFIG["importance_sampling"] is False
    assert DEFAULT_CONFIG["zero_regret_fallback"] == "uniform"


def test_network_size_sweep_variants_cover_width_and_depth():
    variant_layers = {
        variant["variant_id"]: variant["policy_network_layers"]
        for variant in VARIANTS
    }

    assert variant_layers == {
        "tiny_32_32": (32, 32),
        "lightweight_64_64": (64, 64),
        "narrow_128_64": (128, 64),
        "balanced_128_128": (128, 128),
        "exp13_reference_256_128": (256, 128),
        "wide_256_256": (256, 256),
        "very_wide_512_256": (512, 256),
        "shallow_256": (256,),
        "deep_128_128_64": (128, 128, 64),
        "deep_256_256_128": (256, 256, 128),
    }


def test_network_size_sweep_applies_same_architecture_to_all_networks():
    for variant in VARIANTS:
        config = make_variant_config(DEFAULT_CONFIG, variant)
        assert config["policy_network_layers"] == config["regret_network_layers"]
        assert config["policy_network_layers"] == config["value_network_layers"]
