"""Simulador de fluidos da rede emulada (fase 1 do treinamento).

Cada passo representa um intervalo de decisão (dt). O modelo reproduz, de forma
agregada, o que acontece no Mininet: origens legítimas (HTTP com chegadas de
Poisson e um upload contínuo), atacantes (SYN/UDP/ICMP, constantes ou
pulsados), o gargalo do enlace da vítima, a fila de conexões semiabertas do
servidor, os medidores OpenFlow e o nó de análise com limite por origem.

As medições sintéticas passam pelo mesmo FeatureExtractor usado no
controlador. Parâmetros podem ser recalibrados com traços do Mininet
(scripts/calibrate_sim.py -> configs/sim_calibration.json).
"""
from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np

from ..common import (N_ACTIONS, OBS_DIM, ROOT, Action, RewardTerms, reward)
from ..features import FeatureExtractor, GlobalSample, SourceSample

CALIBRATION_FILE = ROOT / "configs" / "sim_calibration.json"

DEFAULT_PARAMS = {
    "http_resp_bytes": 21000,      # resposta HTTP (arquivo de 20 KB + cabeçalhos)
    "http_out_bytes": 900,         # bytes enviados pelo cliente por requisição
    "http_out_pkts": 9,
    # fila de conexões semiabertas (syncookies desligados): entradas expiram após
    # ~7 s (tcp_synack_retries=2), então a fila continua cheia depois do bloqueio
    "syn_backlog": 1024,
    "syn_hold_s": 7,
    "attack_pps_scale": 1.0,       # taxa obtida pelo hping3 / taxa pedida (calibrado)
    "pkt_syn": 54, "pkt_udp": 554, "pkt_icmp": 98,
    "udp_reply_pps": 100,          # ICMP port unreachable limitado pelo kernel
    "reply_icmp_unreach": 590,
    "syn_reply_pps": 1000,
    "pkt_synack": 58,
    "base_rtt_ms": 2.0,
    "queue_ms": 120.0,             # atraso máximo de fila (max_queue_pkts)
    "counter_noise": 0.05,
    "loss_exp": 4,                 # pacotes por requisição sujeitos à perda no gargalo
    "redirect_extra_ms": 1.0,
}


def load_params() -> dict:
    p = dict(DEFAULT_PARAMS)
    if CALIBRATION_FILE.exists():
        p.update(json.loads(CALIBRATION_FILE.read_text(encoding="utf8")))
    return p


@dataclass
class SrcState:
    role: str                    # "http", "bulk", "attacker"
    atype: str = ""              # syn | udp | icmp
    pps: float = 0.0             # taxa do atacante quando ativo
    rand_ports: bool = False
    action: int = Action.ALLOW
    lock: int = 0                # passos restantes de bloqueio
    tau: int = 0                 # passos desde a última mudança de ação
    lam: float = 2.0             # taxa de requisições (http)


class DDoSSimEnv(gym.Env):
    """Ambiente multiorigem: observação (N, OBS_DIM), ação (N,) discreta."""

    metadata = {"render_modes": []}

    def __init__(self, cfg: dict, scenarios: list[str] | None = None,
                 label_free: bool = False, two_actions: bool = False,
                 params: dict | None = None, ambiguous: bool = False):
        self.cfg = cfg
        self.params = params or load_params()
        self.n = cfg["network"]["n_hosts"]
        self.scen_names = scenarios or [k for k, v in cfg["scenarios"].items()
                                        if v.get("train")]
        self.label_free = label_free
        self.two_actions = two_actions
        # treino: atacantes furtivos e legítimos intensos em parte dos episódios
        self.ambiguous = ambiguous
        ex = cfg["experiment"]
        self.T = int(ex["duration_s"] / cfg["agent"]["dt_s"])
        self.t_on = int(ex["warmup_s"])
        self.t_off = int(ex["warmup_s"] + ex["attack_s"])
        self.C = cfg["network"]["victim_bw_mbps"] * 1e6
        self.C_an = cfg["network"]["analysis_bw_mbps"] * 1e6
        self.r_lim = cfg["agent"]["limit_frac"] * self.C
        self.block_steps = int(cfg["agent"]["block_s"] / cfg["agent"]["dt_s"])
        self.an_pps = cfg["analysis_node"]["per_source_pps"]
        self.sampling = cfg["sflow"]["sampling"]
        self.fx = FeatureExtractor(cfg)
        self.observation_space = gym.spaces.Box(0.0, 1.0, (self.n, OBS_DIM), np.float32)
        self.action_space = gym.spaces.MultiDiscrete([N_ACTIONS] * self.n)
        self.rng = np.random.default_rng()

    # ------------------------------------------------------------------ setup
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        opts = options or {}
        name = opts.get("scenario") or self._sample_scenario()
        self.scenario = name
        sc = self.cfg["scenarios"][name]
        self.sc = sc
        rng, ex = self.rng, self.cfg["experiment"]
        self.degraded = sc.get("degraded")
        self.flash = sc["type"] == "flash"
        n_att = 0
        if sc["type"] not in ("normal", "flash"):
            lo, hi = sc["attackers"]
            n_att = int(rng.integers(lo, hi + 1))
        order = rng.permutation(self.n)
        is_stealth = sc["type"] == "stealth"
        stealthy = is_stealth or (self.ambiguous and rng.random() < 0.3)
        intense_legit = is_stealth or self.ambiguous
        self.src: list[SrcState] = []
        bulk_idx = order[n_att] if rng.random() < 0.8 else -1
        for i in range(self.n):
            if i in order[:n_att]:
                k = list(order[:n_att]).index(i)
                atype = sc["type"]
                if atype in ("pulse", "multi", "stealth"):
                    atype = ["syn", "udp"][k % 2]
                pps = (self.cfg["attack_pps"][sc["intensity"]] * rng.uniform(0.6, 1.4)
                       * self.params.get(f"attack_pps_scale_{atype}", self.params["attack_pps_scale"]))
                if stealthy:                 # atacante de baixa taxa, parecido com legítimo
                    pps *= rng.uniform(0.03, 0.25)
                self.src.append(SrcState("attacker", atype, pps,
                                         rand_ports=bool(rng.random() < 0.5)))
            elif i == bulk_idx:
                self.src.append(SrcState("bulk"))
            else:
                lam = ex["legit_rate_rps"] * rng.uniform(0.5, 1.5)
                if intense_legit and rng.random() < 0.3:
                    lam *= rng.uniform(1.0, 4.0)     # cliente legítimo intenso
                self.src.append(SrcState("http", lam=lam))
        self.labels = np.array([s.role == "attacker" for s in self.src])
        self.t = 0
        self.syn_hist = deque(maxlen=64)
        self.fx.reset()
        self.mitigated_bytes = 0.0
        self.sent_bytes = 0.0
        self._measure()        # primeira observação
        return self._obs(), self._info()

    def _sample_scenario(self) -> str:
        names = self.scen_names
        if "S0" in names and self.rng.random() < 0.15:
            return "S0"
        return str(self.rng.choice([n for n in names if n != "S0"] or names))

    # ---------------------------------------------------------------- dynamics
    def _attack_active(self) -> bool:
        if not (self.t_on <= self.t < self.t_off):
            return False
        if self.sc["type"] == "pulse":
            per = self.sc["on_s"] + self.sc["off_s"]
            return (self.t - self.t_on) % per < self.sc["on_s"]
        return True

    def _measure(self):
        """Gera medições do intervalo atual e atualiza o extrator."""
        P, rng = self.params, self.rng
        active = self._attack_active()
        crowd = self.flash and self.t_on <= self.t < self.t_off
        noise = lambda x: max(0.0, x * (1 + rng.normal(0, P["counter_noise"])))

        n = self.n
        send_bits = np.zeros(n)
        send_pkts = np.zeros(n)
        syn_pps = np.zeros(n)
        req = np.zeros(n)
        for i, s in enumerate(self.src):
            if s.role == "http":
                lam = s.lam * (self.cfg["experiment"]["flash_crowd_factor"] if crowd else 1)
                req[i] = rng.poisson(lam)
                send_pkts[i] = req[i] * P["http_out_pkts"]
                send_bits[i] = req[i] * P["http_out_bytes"] * 8
                syn_pps[i] = 0.0
            elif s.role == "bulk":
                bps = self.cfg["experiment"]["bulk_mbps"] * 1e6
                send_bits[i] = bps
                send_pkts[i] = bps / 8 / 1514
            elif active:
                size = P[f"pkt_{s.atype}"]
                send_pkts[i] = s.pps
                send_bits[i] = s.pps * size * 8
                if s.atype == "syn":
                    syn_pps[i] = s.pps

        # efeito das ações (por origem) -> bits que seguem rumo à vítima
        passed = np.zeros(n)
        redirected = np.zeros(n)
        for i, s in enumerate(self.src):
            a = s.action
            if a == Action.ALLOW:
                passed[i] = send_bits[i]
            elif a == Action.LIMIT:
                passed[i] = min(send_bits[i], self.r_lim)
            elif a == Action.REDIRECT:
                pkt_bits = send_bits[i] / send_pkts[i] if send_pkts[i] > 0 else 0
                redirected[i] = min(send_pkts[i], self.an_pps) * pkt_bits
            # BLOCK -> 0
        an_off = redirected.sum()
        an_loss = max(0.0, 1 - self.C_an / an_off) if an_off > 0 else 0.0
        passed += redirected * (1 - an_loss)

        offered = passed.sum()
        p = max(0.0, 1 - self.C / offered) if offered > 0 else 0.0
        if self.degraded:
            p = 1 - (1 - p) * (1 - self.degraded["loss_pct"] / 100.0)
        frac_pass = np.divide(passed, send_bits, out=np.zeros(n), where=send_bits > 0)
        # efeito real das ações: fração mitigada (atacante) e dano (legítimo)
        self.mit_frac = np.where(send_bits > 0, 1 - frac_pass, 0.0)
        harm = np.zeros(n)
        for i, s in enumerate(self.src):
            if s.action == Action.BLOCK:
                harm[i] = 1.0
            elif s.action == Action.LIMIT:
                harm[i] = 1 - min(1.0, self.r_lim / max(send_bits[i], 1.0))
            elif s.action == Action.REDIRECT:
                harm[i] = min(1.0, 0.1 + an_loss)
        self.harm = harm
        syn_in = (syn_pps * frac_pass).sum() * (1 - p)
        self.syn_hist.append(syn_in)
        occ = sum(list(self.syn_hist)[-int(P["syn_hold_s"]):])
        bl = P["syn_backlog"]
        p_syn = min(1.0, max(0.0, (occ - 0.75 * bl) / (0.25 * bl)))

        # disponibilidade: fração de requisições legítimas atendidas
        ok = 0.0
        for i, s in enumerate(self.src):
            if s.role != "http" or req[i] == 0:
                continue
            q = (1 - p) ** P["loss_exp"] * (1 - p_syn)
            if s.action == Action.BLOCK:
                q = 0.0
            elif s.action == Action.REDIRECT:
                q *= (1 - an_loss) ** P["loss_exp"]
            elif s.action == Action.LIMIT:
                q *= min(1.0, self.r_lim / max(send_bits[i], 1.0))
            ok += rng.binomial(int(req[i]), q)
        tot = req.sum()
        self.availability = ok / tot if tot > 0 else 1.0
        att_bits = (passed * self.labels).sum() * (1 - p)
        self.attack_util = min(1.0, att_bits / self.C)
        sent_att = (send_bits * self.labels).sum() / 8
        self.sent_bytes += sent_att
        self.mitigated_bytes += sent_att - (passed * self.labels).sum() / 8

        # medições (contadores + amostras sFlow) por origem
        g = GlobalSample()
        g.victim_bytes = min(offered, self.C) / 8
        g.analysis_bytes = min(an_off, self.C_an) / 8
        util = offered / self.C
        q_delay = P["queue_ms"] * min(1.0, max(0.0, (util - 0.8) / 0.2))
        base = P["base_rtt_ms"] + (4 * self.degraded["delay_ms"] if self.degraded else 0)
        g.rtt_ms = max(0.1, base + q_delay + rng.normal(0, 0.5))
        g.packet_in = int(rng.poisson(0.1))
        g.restricted_frac = float(np.mean([s.action != Action.ALLOW for s in self.src]))
        for i, s in enumerate(self.src):
            ss = SourceSample()
            ss.pkts_out = noise(send_pkts[i])
            ss.bytes_out = noise(send_bits[i] / 8)
            fp = frac_pass[i] * (1 - p)
            if s.role == "http":
                served = rng.binomial(int(req[i]), (1 - p) ** P["loss_exp"] * (1 - p_syn) * (s.action != Action.BLOCK)) if req[i] else 0
                ss.bytes_in = noise(served * P["http_resp_bytes"])
            elif s.role == "bulk":
                ss.bytes_in = noise(send_pkts[i] / 2 * 66 * fp)
            elif active:
                pp = s.pps * fp
                rr = P.get(f"resp_ratio_{s.atype}")
                if rr is not None:          # razão calibrada a partir dos traços
                    ss.bytes_in = noise(send_bits[i] / 8 * fp * rr / max(1e-6, 1 - rr))
                elif s.atype == "syn":
                    ss.bytes_in = min(pp, P["syn_reply_pps"]) * P["pkt_synack"]
                elif s.atype == "udp":
                    ss.bytes_in = min(pp, P["udp_reply_pps"]) * P["reply_icmp_unreach"]
                else:
                    ss.bytes_in = pp * P["pkt_icmp"]
                ss.bytes_in = noise(ss.bytes_in)
            k = int(rng.binomial(int(send_pkts[i]), 1.0 / self.sampling)) if send_pkts[i] > 0 else 0
            ss.n_samples = k
            if k:
                if s.role == "http":
                    ss.n_syn = int(rng.binomial(k, 1.0 / P["http_out_pkts"]))
                    ss.dst_ports = {80}
                    ss.conns = set(rng.integers(32768, 61000, size=min(k, max(1, int(req[i]))) ).tolist())
                elif s.role == "bulk":
                    ss.dst_ports = {5201}
                    ss.conns = {1}
                elif s.atype == "syn":
                    ss.n_syn = k
                    ss.dst_ports = set(rng.integers(1, 65535, size=k).tolist()) if s.rand_ports else {80}
                    ss.conns = set(range(k))
                elif s.atype == "udp":
                    ss.n_udp = k
                    ss.dst_ports = set(rng.integers(1, 65535, size=k).tolist()) if s.rand_ports else {53}
                    ss.conns = set(range(k))
                else:
                    ss.n_icmp = k
                    ss.conns = {0}
            self.fx.push(str(i), ss)
        self.glob = self.fx.global_(g, self.n)

    def _obs(self) -> np.ndarray:
        return np.stack([self.fx.observation(str(i), self.glob, s.action, s.tau)
                         for i, s in enumerate(self.src)])

    def decide_mask(self) -> np.ndarray:
        return np.array([s.lock == 0 for s in self.src])

    def _info(self) -> dict:
        return {"decide": self.decide_mask(), "labels": self.labels.copy(),
                "availability": self.availability, "attack_util": self.attack_util,
                "scenario": self.scenario}

    def step(self, actions):
        actions = np.asarray(actions, dtype=int)
        prev = np.array([s.action for s in self.src])
        for i, s in enumerate(self.src):
            if s.lock > 0:
                continue
            a = int(actions[i])
            if self.two_actions and a in (Action.LIMIT, Action.REDIRECT):
                a = Action.ALLOW
            if a != s.action:
                s.tau = 0
            s.action = a
            if a == Action.BLOCK:
                s.lock = self.block_steps
        self.t += 1
        self._measure()
        terms = RewardTerms(self.availability, self.attack_util)
        rewards = np.array([
            reward(self.cfg, s.action, int(prev[i]), bool(self.labels[i]), terms,
                   self.label_free, mitigated=float(self.mit_frac[i]), harm=float(self.harm[i]))
            for i, s in enumerate(self.src)], dtype=np.float32)
        # avança relógios; bloqueio expira (hard_timeout) e volta a permitir
        for s in self.src:
            s.tau += 1
            if s.lock > 0:
                s.lock -= 1
                if s.lock == 0:
                    s.action = Action.ALLOW
                    s.tau = 0
        truncated = self.t >= self.T
        return self._obs(), rewards, False, truncated, self._info()
