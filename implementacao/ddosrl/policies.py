"""Políticas de mitigação: agente (DQN) e baselines B0-B4.

Todas recebem a matriz de observações (N, OBS_DIM) das origens que podem
decidir e devolvem uma ação por origem. Os baselines usam apenas as colunas de
atributos já presentes na observação, garantindo as mesmas entradas.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from .common import GLOBAL_FEATURES, LOCAL_FEATURES, N_LOCAL, ROOT, Action

I_PPS = LOCAL_FEATURES.index("pps_out")
I_ENT = N_LOCAL + GLOBAL_FEATURES.index("src_entropy")
BASELINE_FILE = ROOT / "models" / "baselines.pkl"


class Policy:
    name = "base"

    def decide(self, obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def q_values(self, obs):          # para auditoria; só o agente tem Q
        return None


class NoDefense(Policy):
    name = "B0"

    def decide(self, obs):
        return np.full(len(obs), Action.ALLOW)


class FixedThreshold(Policy):
    """B1: bloqueia temporariamente origens acima de um limiar de pps."""
    name = "B1"

    def __init__(self, thr_pps_norm: float):
        self.thr = thr_pps_norm

    def decide(self, obs):
        return np.where(obs[:, I_PPS] > self.thr, Action.BLOCK, Action.ALLOW)


class EntropyDetector(Policy):
    """B2: queda da entropia das origens rumo à vítima indica ataque; então
    bloqueia as origens com taxa acima de 1/N do total (adaptação de Mousavi e
    St-Hilaire, que usam a entropia de destino, ao cenário com um único servidor)."""
    name = "B2"

    def __init__(self, thr_entropy: float):
        self.thr = thr_entropy

    def decide(self, obs):
        acts = np.full(len(obs), Action.ALLOW)
        if len(obs) == 0 or obs[0, I_ENT] >= self.thr:
            return acts
        pps = obs[:, I_PPS]
        share = pps / max(pps.sum(), 1e-9)
        acts[share > 1.0 / max(len(obs), 1)] = Action.BLOCK
        return acts


class RandomForestPolicy(Policy):
    """B3: RF sobre os atributos locais; bloqueia se P(ataque) > 0,5.
    B4 (graduated=True): faixas de probabilidade mapeadas para as 4 ações."""

    BANDS = [(0.5, Action.ALLOW), (0.7, Action.LIMIT), (0.85, Action.REDIRECT)]

    def __init__(self, rf, graduated: bool = False):
        self.rf = rf
        self.graduated = graduated
        self.name = "B4" if graduated else "B3"

    def proba(self, obs):
        return self.rf.predict_proba(obs[:, :N_LOCAL])[:, 1]

    def decide(self, obs):
        if len(obs) == 0:
            return np.zeros(0, int)
        p = self.proba(obs)
        if not self.graduated:
            return np.where(p > 0.5, Action.BLOCK, Action.ALLOW)
        acts = np.full(len(p), Action.BLOCK)
        for thr, a in reversed(self.BANDS):
            acts[p <= thr] = a
        return acts


class DQNPolicy(Policy):
    def __init__(self, model_path: str | Path, name: str = "DQN"):
        from .agent.numpy_policy import NumpyDuelingQ
        self.net = NumpyDuelingQ(model_path)
        self.name = name

    def decide(self, obs):
        if len(obs) == 0:
            return np.zeros(0, int)
        return self.net.act(obs)

    def q_values(self, obs):
        return self.net.q_values(obs)


def load_policy(name: str, model_dir: str | Path | None = None) -> Policy:
    """Constrói a política pelo nome usado nos experimentos."""
    model_dir = Path(model_dir) if model_dir else ROOT / "models"
    if name == "B0":
        return NoDefense()
    if name in ("B1", "B2", "B3", "B4"):
        with open(model_dir / "baselines.pkl", "rb") as f:
            b = pickle.load(f)
        if name == "B1":
            return FixedThreshold(b["thr_pps"])
        if name == "B2":
            return EntropyDetector(b["thr_entropy"])
        return RandomForestPolicy(b["rf"], graduated=name == "B4")
    # agente e ablações: "DQN" -> ddqn_full_s1, "DQN2" -> two_actions, "DQNLF" -> label_free
    variant = {"DQN": "full", "DQN2": "two_actions", "DQNLF": "label_free"}[name]
    return DQNPolicy(model_dir / f"ddqn_{variant}_s1", name)
