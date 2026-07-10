"""Small Gymnasium compatibility layer used by the offline validation image.

The real experiment should install Gymnasium.  The fallback keeps Stage-0 unit
tests executable in restricted environments where third-party packages cannot
be installed.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

try:  # pragma: no cover - exercised when Gymnasium is installed
    import gymnasium as gym
    from gymnasium import spaces

    GYMNASIUM_AVAILABLE = True
except ImportError:  # pragma: no cover - fallback is tested indirectly
    GYMNASIUM_AVAILABLE = False

    class Env:
        metadata: dict[str, Any] = {}

        def reset(self, *, seed: int | None = None, options: dict | None = None):
            if seed is not None:
                self.np_random = np.random.default_rng(seed)

    class Box:
        def __init__(self, low, high, shape=None, dtype=np.float32):
            self.dtype = np.dtype(dtype)
            if shape is not None:
                self.low = np.full(shape, low, dtype=self.dtype)
                self.high = np.full(shape, high, dtype=self.dtype)
            else:
                self.low = np.asarray(low, dtype=self.dtype)
                self.high = np.asarray(high, dtype=self.dtype)
            self.shape = self.low.shape
            self._rng = np.random.default_rng()

        def seed(self, seed: int | None = None):
            self._rng = np.random.default_rng(seed)
            return [seed]

        def sample(self):
            return self._rng.uniform(self.low, self.high).astype(self.dtype)

        def contains(self, value) -> bool:
            array = np.asarray(value)
            return (
                array.shape == self.shape
                and np.all(np.isfinite(array))
                and np.all(array >= self.low)
                and np.all(array <= self.high)
            )

    gym = SimpleNamespace(Env=Env)
    spaces = SimpleNamespace(Box=Box)


__all__ = ["GYMNASIUM_AVAILABLE", "gym", "spaces"]

