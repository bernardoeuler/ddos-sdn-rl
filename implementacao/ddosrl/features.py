"""Extração de atributos a partir de medições brutas por intervalo.

O mesmo código é usado pelo controlador (medições reais: contadores OpenFlow e
amostras sFlow) e pelo simulador (medições sintéticas), garantindo que o agente
veja atributos com a mesma semântica nos dois ambientes.
"""
from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field  # noqa: F401

import numpy as np

from .common import (N_GLOBAL, N_LOCAL, PPS_NORM, RTT_NORM_MS, build_obs,
                     clip01)


@dataclass
class SourceSample:
    """Medições de uma origem em um intervalo de duração dt."""
    pkts_out: float = 0.0
    bytes_out: float = 0.0
    bytes_in: float = 0.0
    # contagens a partir das amostras sFlow (pacotes amostrados)
    n_samples: int = 0
    n_syn: int = 0
    n_udp: int = 0
    n_icmp: int = 0
    dst_ports: set = field(default_factory=set)
    conns: set = field(default_factory=set)


@dataclass
class GlobalSample:
    victim_bytes: float = 0.0       # bytes transmitidos ao servidor no intervalo
    analysis_bytes: float = 0.0     # bytes transmitidos ao nó de análise
    rtt_ms: float = 0.0             # RTT médio das sondas (0 = sem medição)
    packet_in: int = 0
    restricted_frac: float = 0.0


def entropy_norm(counts: dict, n_sources: int) -> float:
    """Entropia de Shannon das origens, normalizada por log2(n_sources)."""
    total = sum(counts.values())
    if total <= 0 or n_sources < 2:
        return 1.0
    h = -sum((c / total) * math.log2(c / total) for c in counts.values() if c > 0)
    return clip01(h / math.log2(n_sources))


class FeatureExtractor:
    def __init__(self, cfg: dict):
        net, ag = cfg["network"], cfg["agent"]
        self.dt = float(ag["dt_s"])
        self.window = int(ag["window"])
        self.victim_bps = net["victim_bw_mbps"] * 1e6
        self.analysis_bps = net["analysis_bw_mbps"] * 1e6
        self.hist: dict[str, deque] = defaultdict(lambda: deque(maxlen=self.window))
        self.rtt_hist: deque = deque(maxlen=self.window)

    def reset(self):
        self.hist.clear()
        self.rtt_hist.clear()

    def push(self, src: str, s: SourceSample):
        self.hist[src].append(s)

    def local(self, src: str) -> np.ndarray:
        h = self.hist.get(src)
        if not h:
            return np.zeros(N_LOCAL, dtype=np.float32)
        span = self.dt * len(h)
        pkts = sum(s.pkts_out for s in h)
        b_out = sum(s.bytes_out for s in h)
        b_in = sum(s.bytes_in for s in h)
        n = sum(s.n_samples for s in h)
        ports, conns = set(), set()
        for s in h:
            ports |= s.dst_ports
            conns |= s.conns
        pps = pkts / span
        first = h[0].pkts_out / self.dt
        last = h[-1].pkts_out / self.dt
        frac = (lambda k: k / n) if n > 0 else (lambda k: 0.0)
        x = [
            clip01(pps / PPS_NORM),
            clip01(b_out * 8 / span / self.victim_bps),
            clip01((b_out / pkts) / 1500.0) if pkts > 0 else 0.0,
            frac(sum(s.n_syn for s in h)),
            frac(sum(s.n_udp for s in h)),
            frac(sum(s.n_icmp for s in h)),
            clip01(len(ports) / 64.0),
            clip01(len(conns) / 64.0),
            b_in / (b_in + b_out) if (b_in + b_out) > 0 else 0.0,
            max(-1.0, min(1.0, (last - first) / PPS_NORM)),
        ]
        return np.asarray(x, dtype=np.float32)

    def global_(self, g: GlobalSample, n_sources: int) -> np.ndarray:
        if g.rtt_ms > 0:
            self.rtt_hist.append(g.rtt_ms)
        rtt = (sum(self.rtt_hist) / len(self.rtt_hist)) if self.rtt_hist else 0.0
        # entropia das origens a partir dos contadores de pacotes na janela
        # (todo o tráfego das origens segue para a vítima nesta topologia)
        counts = {s: sum(x.pkts_out for x in h) for s, h in self.hist.items()}
        v = [
            clip01(g.victim_bytes * 8 / self.dt / self.victim_bps),
            clip01(rtt / RTT_NORM_MS),
            entropy_norm(counts, n_sources),
            clip01(g.packet_in / self.dt / 100.0),
            clip01(g.analysis_bytes * 8 / self.dt / self.analysis_bps),
            clip01(g.restricted_frac),
        ]
        assert len(v) == N_GLOBAL
        return np.asarray(v, dtype=np.float32)

    def observation(self, src: str, glob: np.ndarray, prev_action: int,
                    tau_s: float) -> np.ndarray:
        return build_obs(self.local(src), glob, prev_action, tau_s)
