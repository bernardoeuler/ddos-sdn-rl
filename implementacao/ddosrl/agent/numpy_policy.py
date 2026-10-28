"""Inferência da rede dueling em numpy (usada dentro do controlador)."""
from __future__ import annotations

from pathlib import Path

import numpy as np


class NumpyDuelingQ:
    def __init__(self, path: str | Path):
        z = np.load(Path(path).with_suffix(".npz"))
        self.mask = z["mask"].astype(bool)
        self.layers = []
        i = 0
        while f"body__{i}__weight" in z:
            self.layers.append((z[f"body__{i}__weight"], z[f"body__{i}__bias"]))
            i += 2                      # Linear, ReLU, Linear, ReLU ...
        self.v = (z["value__weight"], z["value__bias"])
        self.a = (z["adv__weight"], z["adv__bias"])

    def q_values(self, obs: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(obs).astype(np.float32)
        for w, b in self.layers:
            x = np.maximum(x @ w.T + b, 0.0)
        v = x @ self.v[0].T + self.v[1]
        a = x @ self.a[0].T + self.a[1]
        q = v + a - a.mean(axis=1, keepdims=True)
        return np.where(self.mask, q, -1e9)

    def act(self, obs: np.ndarray) -> np.ndarray:
        return self.q_values(obs).argmax(axis=1)
