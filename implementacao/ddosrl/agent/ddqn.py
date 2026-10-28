"""Double DQN com arquitetura dueling (PyTorch).

Cada origem é um "agente" com política compartilhada. Uma transição vai de uma
decisão da origem até a sua próxima decisão (k passos depois); com o bloqueio
temporário, k = T_b, e o alvo usa gamma^k com a recompensa descontada
acumulada (formulação semi-Markoviana).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..common import N_ACTIONS, OBS_DIM


class DuelingQNet(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM, n_actions: int = N_ACTIONS,
                 hidden=(128, 128)):
        super().__init__()
        layers, d = [], obs_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        self.body = nn.Sequential(*layers)
        self.value = nn.Linear(d, 1)
        self.adv = nn.Linear(d, n_actions)

    def forward(self, x):
        z = self.body(x)
        a = self.adv(z)
        return self.value(z) + a - a.mean(dim=-1, keepdim=True)


class ReplayBuffer:
    def __init__(self, size: int, obs_dim: int = OBS_DIM, seed: int = 0):
        self.size, self.i, self.full = size, 0, False
        self.s = np.zeros((size, obs_dim), np.float32)
        self.s2 = np.zeros((size, obs_dim), np.float32)
        self.a = np.zeros(size, np.int64)
        self.r = np.zeros(size, np.float32)
        self.g = np.zeros(size, np.float32)       # gamma^k
        self.d = np.zeros(size, np.float32)
        self.rng = np.random.default_rng(seed)

    def add(self, s, a, r, s2, gk, done):
        j = self.i
        self.s[j], self.a[j], self.r[j], self.s2[j], self.g[j], self.d[j] = s, a, r, s2, gk, done
        self.i = (j + 1) % self.size
        self.full = self.full or self.i == 0

    def __len__(self):
        return self.size if self.full else self.i

    def sample(self, n):
        idx = self.rng.integers(0, len(self), n)
        return (self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.g[idx], self.d[idx])


class DDQNAgent:
    def __init__(self, cfg: dict, seed: int = 0, action_mask=None):
        ag = cfg["agent"]
        torch.manual_seed(seed)
        self.cfg = cfg
        self.hidden = tuple(ag["hidden"])
        self.q = DuelingQNet(hidden=self.hidden)
        self.q_tgt = DuelingQNet(hidden=self.hidden)
        self.q_tgt.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=ag["lr"])
        self.buf = ReplayBuffer(ag["buffer_size"], seed=seed)
        self.batch = ag["batch_size"]
        self.target_update = ag["target_update"]
        self.updates = 0
        self.rng = np.random.default_rng(seed)
        # máscara de ações válidas (ablação {permitir, bloquear})
        self.mask = np.ones(N_ACTIONS, bool) if action_mask is None else np.asarray(action_mask, bool)
        self._mask_t = torch.as_tensor(np.where(self.mask, 0.0, -1e9), dtype=torch.float32)

    def act(self, obs: np.ndarray, eps: float = 0.0) -> np.ndarray:
        with torch.no_grad():
            q = self.q(torch.as_tensor(obs, dtype=torch.float32)) + self._mask_t
        a = q.argmax(dim=-1).numpy()
        if eps > 0:
            valid = np.flatnonzero(self.mask)
            rnd = self.rng.random(len(a)) < eps
            a[rnd] = self.rng.choice(valid, rnd.sum())
        return a

    def update(self) -> float | None:
        if len(self.buf) < max(self.batch, 1000):
            return None
        s, a, r, s2, g, d = (torch.as_tensor(x) for x in self.buf.sample(self.batch))
        with torch.no_grad():
            a2 = (self.q(s2) + self._mask_t).argmax(dim=1, keepdim=True)    # seleção: rede online
            q2 = self.q_tgt(s2).gather(1, a2).squeeze(1)                    # avaliação: rede-alvo
            y = r + g * (1 - d) * q2
        qsa = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)
        loss = nn.functional.smooth_l1_loss(qsa, y)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()
        self.updates += 1
        if self.updates % self.target_update == 0:
            self.q_tgt.load_state_dict(self.q.state_dict())
        return float(loss.detach())

    # --------------------------------------------------------------- export
    def save(self, path: str | Path, meta: dict | None = None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"q": self.q.state_dict(), "mask": self.mask.tolist(),
                    "hidden": self.hidden, "meta": meta or {}}, path.with_suffix(".pt"))
        # pesos em numpy para inferência no controlador (sem torch)
        arrs = {k: v.detach().numpy() for k, v in self.q.state_dict().items()}
        np.savez(path.with_suffix(".npz"), mask=self.mask, **{k.replace(".", "__"): v for k, v in arrs.items()})
        (path.with_suffix(".json")).write_text(json.dumps(meta or {}, indent=2), encoding="utf8")

    def load(self, path: str | Path):
        ck = torch.load(Path(path).with_suffix(".pt"), weights_only=False)
        self.q.load_state_dict(ck["q"])
        self.q_tgt.load_state_dict(ck["q"])
        self.mask = np.asarray(ck["mask"], bool)
        self._mask_t = torch.as_tensor(np.where(self.mask, 0.0, -1e9), dtype=torch.float32)
