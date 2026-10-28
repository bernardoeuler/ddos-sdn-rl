"""Treina o agente Double Dueling DQN no simulador (fase 1).

Variantes (ablações do artigo):
  full        quatro ações, recompensa da Eq. 2
  two_actions apenas {permitir, bloquear}
  label_free  recompensa sem rótulos (apenas termos globais e custo das ações)

Uso: python scripts/train_agent.py --variant full --seed 1
"""
from __future__ import annotations

import argparse
import csv
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ddosrl.agent.ddqn import DDQNAgent                    # noqa: E402
from ddosrl.common import ROOT, Action, load_config        # noqa: E402
from ddosrl.sim.env import DDoSSimEnv                      # noqa: E402


class SMDPCollector:
    """Acumula a recompensa de cada origem entre duas decisões consecutivas."""

    def __init__(self, n: int, gamma: float):
        self.gamma = gamma
        self.pending = [None] * n      # [s, a, R, k]

    def decide(self, i, s_now, a, buf):
        p = self.pending[i]
        if p is not None:
            buf.add(p[0], p[1], p[2], s_now, self.gamma ** p[3], 0.0)
        self.pending[i] = [s_now, int(a), 0.0, 0]

    def reward(self, rewards):
        for i, p in enumerate(self.pending):
            if p is not None:
                p[2] += (self.gamma ** p[3]) * float(rewards[i])
                p[3] += 1

    def flush(self, obs, buf):
        for i, p in enumerate(self.pending):
            if p is not None:          # truncamento por tempo: faz bootstrap
                buf.add(p[0], p[1], p[2], obs[i], self.gamma ** p[3], 0.0)
        self.pending = [None] * len(self.pending)


def run_episode(env, agent, eps, seed, scenario=None, collect=None, train=True,
                updates_per_step=2):
    obs, info = env.reset(seed=seed, options={"scenario": scenario} if scenario else None)
    col = SMDPCollector(env.n, agent.cfg["agent"]["gamma"]) if collect else None
    ret, avail, steps, losses = 0.0, [], 0, []
    acts = np.zeros(env.n, int)
    while True:
        mask = info["decide"]
        if mask.any():
            chosen = agent.act(obs[mask], eps)
            acts[mask] = chosen
            if col:
                for j, i in enumerate(np.flatnonzero(mask)):
                    col.decide(i, obs[i], chosen[j], agent.buf)
        obs, r, _, trunc, info = env.step(acts)
        if col:
            col.reward(r)
        ret += float(r.mean())
        avail.append(info["availability"])
        steps += 1
        if train:
            for _ in range(updates_per_step):
                l = agent.update()
                if l is not None:
                    losses.append(l)
        if trunc:
            break
    if col:
        col.flush(obs, agent.buf)
    tab = env.mitigated_bytes / env.sent_bytes if env.sent_bytes > 0 else float("nan")
    return {"return": ret, "availability": float(np.mean(avail)), "tab": tab,
            "steps": steps, "loss": float(np.mean(losses)) if losses else float("nan")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="full", choices=["full", "two_actions", "label_free"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--updates", type=int, default=None, help="passos de gradiente")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    ag = cfg["agent"]
    total_updates = args.updates or ag["train_steps"]
    two = args.variant == "two_actions"
    env = DDoSSimEnv(cfg, label_free=args.variant == "label_free", two_actions=two, ambiguous=True)
    mask = [True, False, False, True] if two else None
    agent = DDQNAgent(cfg, seed=args.seed, action_mask=mask)
    # modelo da fase 1 (simulador); o ajuste fino no Mininet gera ddqn_full_s<seed>
    out = Path(args.out) if args.out else ROOT / "models" / f"ddqn_{args.variant}_sim_s{args.seed}"
    curve_path = ROOT / "results" / "training" / f"curve_{args.variant}_s{args.seed}.csv"
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    eval_scen = [k for k, v in cfg["scenarios"].items() if v.get("train")]

    t0, ep = time.time(), 0
    rng = np.random.default_rng(args.seed)
    with open(curve_path, "w", newline="", encoding="utf8") as f:
        w = csv.writer(f)
        w.writerow(["episode", "updates", "eps", "train_return", "eval_return",
                    "eval_availability", "eval_tab", "elapsed_s"])
        while agent.updates < total_updates:
            frac = min(1.0, agent.updates / ag["eps_decay_steps"])
            eps = ag["eps_start"] + frac * (ag["eps_end"] - ag["eps_start"])
            tr = run_episode(env, agent, eps, int(rng.integers(1 << 30)), collect=True)
            ep += 1
            if ep % 25 == 0:
                evs = [run_episode(env, agent, 0.0, 10_000 + k, scenario=s, train=False)
                       for k, s in enumerate(eval_scen)]
                row = [ep, agent.updates, round(eps, 3), round(tr["return"], 2),
                       round(np.mean([e["return"] for e in evs]), 2),
                       round(np.mean([e["availability"] for e in evs]), 4),
                       round(np.nanmean([e["tab"] for e in evs]), 4),
                       round(time.time() - t0, 1)]
                w.writerow(row)
                f.flush()
                print("ep", *row, flush=True)

    meta = {"variant": args.variant, "seed": args.seed, "updates": agent.updates,
            "episodes": ep, "train_s": round(time.time() - t0, 1),
            "python": platform.python_version(), "agent_cfg": ag, "reward_cfg": cfg["reward"]}
    agent.save(out, meta)
    if not args.out:
        # versão implantável; as ablações usam diretamente o modelo do simulador
        agent.save(ROOT / "models" / f"ddqn_{args.variant}_s{args.seed}", meta)
    print("salvo em", out)


if __name__ == "__main__":
    main()
