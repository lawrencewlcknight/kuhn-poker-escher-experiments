"""Configuration for the ESCHER checkpoint-stability experiment."""

from __future__ import annotations

from copy import deepcopy
from typing import List

from experiments.kuhn_poker.escher_multiseed_baseline.config import (
    DEFAULT_CONFIG as BASELINE_DEFAULT_CONFIG,
    DEFAULT_SEEDS,
    DEVELOPMENT_SEEDS_3,
    DEVELOPMENT_SEEDS_5 as BASELINE_DEVELOPMENT_SEEDS_5,
)

DEFAULT_CONFIG = deepcopy(BASELINE_DEFAULT_CONFIG)
DEFAULT_CONFIG.update({
    "experiment_name": "kuhn_poker_escher_checkpoint_stability",
})

CHECKPOINT_SCHEDULE = [10, 30, 50, 75, 100, 115, 130]
FINAL_ITERATION = CHECKPOINT_SCHEDULE[-1]
EQUIVALENCE_EPSILON = 0.001

RUN_CHECKPOINTED_ARM = True
RUN_CONTINUOUS_BASELINE_ARM = True
SAVE_FULL_CHECKPOINTS = True
SAVE_POLICY_SNAPSHOTS = True
ANNOTATE_HEATMAP = True

SMOKE_TEST_SEEDS = [1234]
DEVELOPMENT_SEEDS = DEVELOPMENT_SEEDS_3
DEVELOPMENT_SEEDS_5 = BASELINE_DEVELOPMENT_SEEDS_5


def parse_checkpoint_schedule(value: str | None) -> List[int] | None:
    if value is None:
        return None
    schedule = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not schedule:
        raise ValueError("Checkpoint schedule must not be empty.")
    if schedule != sorted(schedule):
        raise ValueError("Checkpoint schedule must be increasing.")
    if any(item <= 0 for item in schedule):
        raise ValueError("Checkpoint schedule entries must be positive.")
    return schedule
