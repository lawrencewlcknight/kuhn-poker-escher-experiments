"""Configuration for the ESCHER warm-start fair ablation."""

from __future__ import annotations

from copy import deepcopy
from typing import List

from experiments.kuhn_poker.escher_multiseed_baseline.config import (
    DEFAULT_CONFIG as BASELINE_DEFAULT_CONFIG,
    DEFAULT_SEEDS,
    DEVELOPMENT_SEEDS_3,
)

DEFAULT_CONFIG = deepcopy(BASELINE_DEFAULT_CONFIG)
DEFAULT_CONFIG.update({
    "experiment_name": "kuhn_poker_escher_warm_start_fair_ablation",
    "warm_start_boundary": 30,
})

ARM_CONTINUOUS = "baseline_continuous"
ARM_WARM_START = "warm_start"
ARM_LABELS = {
    ARM_CONTINUOUS: "Continuous baseline",
    ARM_WARM_START: "Warm-start resume",
}

SAVE_FULL_CHECKPOINTS = True
RESTORE_RNG_STATE = True
SMOKE_TEST_SEEDS = [1234]
DEVELOPMENT_SEEDS = DEVELOPMENT_SEEDS_3


def parse_seeds(seed_string: str | None, default: List[int] | None = None) -> List[int]:
    if not seed_string:
        return list(DEFAULT_SEEDS if default is None else default)
    return [int(item.strip()) for item in seed_string.split(",") if item.strip()]
