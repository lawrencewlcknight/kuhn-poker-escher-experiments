from experiments.kuhn_poker.escher_author_budget_multiseed.config import (
    DEFAULT_CONFIG,
    DEFAULT_SEEDS,
)


def test_author_budget_multiseed_defaults_match_experiment_12_best_config():
    assert DEFAULT_SEEDS == [1234, 2025, 31415, 27182, 16180]
    assert DEFAULT_CONFIG["num_iterations"] == 80
    assert DEFAULT_CONFIG["num_traversals"] == 500
    assert DEFAULT_CONFIG["num_val_fn_traversals"] == 500
    assert DEFAULT_CONFIG["policy_network_layers"] == (256, 128)
    assert DEFAULT_CONFIG["regret_network_layers"] == (256, 128)
    assert DEFAULT_CONFIG["value_network_layers"] == (256, 128)
    assert DEFAULT_CONFIG["batch_size_average_policy"] == 10_000
    assert DEFAULT_CONFIG["policy_network_train_steps"] == 1000
    assert DEFAULT_CONFIG["regret_network_train_steps"] == 200
    assert DEFAULT_CONFIG["value_network_train_steps"] == 200
    assert DEFAULT_CONFIG["importance_sampling"] is False
    assert DEFAULT_CONFIG["zero_regret_fallback"] == "uniform"
