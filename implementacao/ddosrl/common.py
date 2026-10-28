"""Definições compartilhadas por simulador, controlador, agente e baselines.

O layout do vetor de observação é o mesmo em todos os componentes; qualquer
mudança aqui precisa ser refletida em sim/env.py e sdn/controller.py.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"


class Action(IntEnum):
    ALLOW = 0
    LIMIT = 1
    REDIRECT = 2
    BLOCK = 3


ACTION_NAMES = ["permitir", "limitar", "redirecionar", "bloquear"]
N_ACTIONS = len(Action)
RESTRICTIVE = (Action.LIMIT, Action.REDIRECT, Action.BLOCK)

# Atributos locais x_t^i (por origem), todos normalizados para ~[0, 1]
LOCAL_FEATURES = [
    "pps_out",        # pacotes/s enviados pela origem / PPS_NORM
    "bps_out",        # bits/s enviados / capacidade do gargalo
    "pkt_size",       # tamanho médio de pacote / 1500
    "syn_ratio",      # fração de segmentos TCP só-SYN (sFlow)
    "udp_ratio",      # fração UDP (sFlow)
    "icmp_ratio",     # fração ICMP (sFlow)
    "dst_ports",      # portas de destino distintas (sFlow) / 64
    "conns",          # conexões (5-tuplas) distintas estimadas (sFlow) / 64
    "resp_ratio",     # bytes recebidos / (recebidos + enviados)
    "pps_trend",      # variação de pps na janela / PPS_NORM, em [-1, 1]
]
# Atributos globais g_t
GLOBAL_FEATURES = [
    "victim_util",    # utilização do enlace da vítima
    "probe_rtt",      # RTT das sondas / RTT_NORM_MS
    "src_entropy",    # entropia normalizada dos IPs de origem rumo à vítima
    "packet_in",      # packet-in/s / 100
    "analysis_util",  # utilização do enlace do nó de análise
    "restricted",     # fração das origens com ação restritiva
]
N_LOCAL = len(LOCAL_FEATURES)
N_GLOBAL = len(GLOBAL_FEATURES)
OBS_DIM = N_LOCAL + N_GLOBAL + N_ACTIONS + 1   # + one-hot(a_{t-1}) + tau

PPS_NORM = 10000.0
RTT_NORM_MS = 200.0
TAU_NORM_S = 30.0


def clip01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else float(x)


def build_obs(local, glob, prev_action: int, tau_s: float):
    """Concatena [x || g || onehot(a_{t-1}) || tau] (Eq. 1 do artigo)."""
    import numpy as np
    onehot = [0.0] * N_ACTIONS
    onehot[int(prev_action)] = 1.0
    v = list(local) + list(glob) + onehot + [clip01(tau_s / TAU_NORM_S)]
    return np.asarray(v, dtype=np.float32)


def load_config(path: str | os.PathLike | None = None) -> dict:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, encoding="utf8") as f:
        return yaml.safe_load(f)


@dataclass
class RewardTerms:
    """Termos observados num intervalo, usados pela Eq. 2."""
    availability: float          # D_t
    attack_util: float           # U^atq_t


def reward(cfg: dict, action: int, prev_action: int, is_attacker: bool | None,
           terms: RewardTerms, label_free: bool = False,
           mitigated: float | None = None, harm: float | None = None) -> float:
    """Recompensa da origem i (Eq. 2 do artigo).

    r = w1*1[M]*b - w2*1[L]*h + w3*D - w4*U - w5*c(a, a_prev)
    Com o efeito medido (simulador): b = fração do tráfego do atacante mitigada
    e h = max(h(a), fração do tráfego legítimo afetada); sem ele, b(a) e h(a)
    da configuração. Na variante sem rótulos, o termo local é -w6*h.
    """
    r = cfg["reward"]
    b, h = r["benefit"][action], r["harm"][action]
    if mitigated is not None:
        b = mitigated
    if harm is not None:
        h = max(h, harm)
    if label_free:
        local = -r["w6"] * h
    else:
        local = r["w1"] * b if is_attacker else -r["w2"] * h
    change = 1.0 if action != prev_action else 0.0
    return (local + r["w3"] * terms.availability - r["w4"] * terms.attack_util
            - r["w5"] * change)
