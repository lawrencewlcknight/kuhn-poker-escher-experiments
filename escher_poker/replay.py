"""Replay buffers used by ESCHER experiments."""

from __future__ import annotations

import random
from typing import Iterable, List, Any

import numpy as np


class ReservoirBuffer(object):
    """Uniform reservoir sampler over a stream of data."""

    def __init__(self, reservoir_buffer_capacity: int):
        self._reservoir_buffer_capacity = int(reservoir_buffer_capacity)
        self._data: List[Any] = []
        self._add_calls = 0

    def add(self, element: Any) -> None:
        """Potentially add ``element`` using standard reservoir sampling."""
        if len(self._data) < self._reservoir_buffer_capacity:
            self._data.append(element)
        else:
            idx = np.random.randint(0, self._add_calls + 1)
            if idx < self._reservoir_buffer_capacity:
                self._data[idx] = element
        self._add_calls += 1

    def sample(self, num_samples: int) -> list[Any]:
        """Return ``num_samples`` uniformly sampled elements from the buffer."""
        if len(self._data) < num_samples:
            raise ValueError(f"{num_samples} elements could not be sampled from size {len(self._data)}")
        return random.sample(self._data, num_samples)

    def clear(self) -> None:
        self._data = []
        self._add_calls = 0

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterable[Any]:
        return iter(self._data)

    @property
    def data(self) -> list[Any]:
        return self._data

    def get_data(self) -> list[Any]:
        return self._data

    def shuffle_data(self) -> None:
        random.shuffle(self._data)

    def get_num_calls(self) -> int:
        return self._add_calls
