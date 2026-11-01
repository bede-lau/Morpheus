"""Stable contract between robot-specific policies and physics evaluation."""

from typing import Protocol

import numpy as np


class Policy(Protocol):
    def validate(self, model) -> dict: ...
    def reset(self, model, data) -> None: ...
    def step(self, model, data, command: np.ndarray) -> np.ndarray: ...
