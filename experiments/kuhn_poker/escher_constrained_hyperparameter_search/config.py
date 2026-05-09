"""Configuration for the constrained ESCHER hyperparameter search."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List

from experiments.kuhn_poker.escher_multiseed_baseline.config import (
    DEFAULT_CONFIG as BASELINE_DEFAULT_CONFIG,
)
from escher_poker.hyperparameter_search import sample_candidate_configs

BASELINE_VARIANT_ID = "baseline_escher_exp1"

DEFAULT_CONFIG = deepcopy(BASELINE_DEFAULT_CONFIG)
DEFAULT_CONFIG.update({
    "experiment_name": "kuhn_poker_escher_constrained_hyperparameter_search",
    "variant_id": BASELINE_VARIANT_ID,
    "expl": 1.0,
    "val_expl": 0.01,
    "importance_sampling": True,
    "importance_sampling_threshold": 100.0,
    "clear_value_buffer": True,
    "val_bootstrap": False,
    "all_actions": True,
})

SCREENING_ITERATIONS = 80
SCREENING_EVALUATION_INTERVAL = 10
SCREENING_SEEDS = [1234, 2025]
N_RANDOM_CANDIDATES = 4

CONFIRMATION_ITERATIONS = 130
CONFIRMATION_EVALUATION_INTERVAL = 5
CONFIRMATION_SEEDS = [1234, 2025, 31415]
CONFIRMATION_TOP_K = 2

OPTIONAL_CONFIRMATION_SEEDS_5 = [1234, 2025, 31415, 27182, 16180]
OPTIONAL_CONFIRMATION_SEEDS_10 = [1234, 2025, 31415, 27182, 16180, 4242, 8675309, 7, 99, 1001]

RANDOM_SEARCH_SEED = 1729

SEARCH_SPACE = {
    "learning_rate": [3e-4, 5e-4, 1e-3, 2e-3],
    "num_traversals": [250, 500, 750, 1000],
    "num_val_fn_traversals": [500, 750, 1000, 1500],
    "regret_network_train_steps": [200, 400, 800],
    "value_network_train_steps": [200, 400, 800],
    "policy_network_train_steps": [1000, 1500, 2000],
    "policy_network_layers": [(128, 128), (256, 128), (256, 256), (512, 256)],
    "regret_network_layers": [(128, 128), (256, 128), (256, 256), (512, 256)],
    "value_network_layers": [(128, 128), (256, 128), (256, 256), (512, 256)],
    "batch_size_regret": [128, 256, 512],
    "batch_size_value": [128, 256, 512],
    "memory_capacity": [int(1e5), int(5e5)],
    "reinitialize_regret_networks": [True, False],
    "reinitialize_value_network": [True, False],
    "expl": [0.5, 1.0],
    "val_expl": [0.01, 0.05, 0.1],
    "clear_value_buffer": [True, False],
}


def make_targeted_candidates(base_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Candidate configurations that test plausible ESCHER failure modes."""
    candidates = []

    def add_targeted(name: str, **overrides) -> None:
        config = dict(base_config)
        config.update(overrides)
        config["variant_id"] = name
        candidates.append(config)

    add_targeted(
        "targeted_more_value_fitting",
        num_val_fn_traversals=1000,
        value_network_train_steps=800,
        learning_rate=5e-4,
    )
    add_targeted(
        "targeted_more_regret_fitting",
        num_traversals=750,
        regret_network_train_steps=800,
        learning_rate=5e-4,
    )
    add_targeted(
        "targeted_no_reinitialisation",
        reinitialize_regret_networks=False,
        reinitialize_value_network=False,
    )
    add_targeted(
        "targeted_larger_networks_lower_lr",
        policy_network_layers=(512, 256),
        regret_network_layers=(512, 256),
        value_network_layers=(512, 256),
        learning_rate=5e-4,
    )
    return candidates


def build_screening_configs(
    base_config: Dict[str, Any],
    n_random_candidates: int = N_RANDOM_CANDIDATES,
    random_search_seed: int = RANDOM_SEARCH_SEED,
) -> List[Dict[str, Any]]:
    """Build baseline, targeted, and random screening configs."""
    random_candidates = sample_candidate_configs(
        base_config,
        SEARCH_SPACE,
        n_random_candidates,
        random_search_seed,
    )
    return [dict(base_config)] + make_targeted_candidates(base_config) + random_candidates


def parse_seeds(seed_string: str | None, default: List[int]) -> List[int]:
    if not seed_string:
        return list(default)
    return [int(item.strip()) for item in seed_string.split(",") if item.strip()]
