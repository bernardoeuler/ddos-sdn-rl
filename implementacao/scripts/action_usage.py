"""Distribuição das ações por política no simulador (com atacantes furtivos).

Uso: python scripts/action_usage.py --policies DQN DQN2 B4 --episodes 40
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

IMPL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(IMPL))
from ddosrl.common import ACTION_NAMES, load_config   # noqa: E402
from ddosrl.policies import load_policy               # noqa: E402
from ddosrl.sim.env import DDoSSimEnv                 # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", nargs="+", default=["DQN", "DQN2", "B4"])
    ap.add_argument("--episodes", type=int, default=40)
    a = ap.parse_args()
    cfg = load_config()
    for p in a.policies:
        env = DDoSSimEnv(cfg, ambiguous=True, two_actions=p == "DQN2")
        pol = load_policy(p)
        cnt = {"atk": np.zeros(4), "atk_furtivo": np.zeros(4), "leg": np.zeros(4)}
        av = []
        for k in range(a.episodes):
            obs, info = env.reset(seed=80_000 + k)
            low = np.array([s.role == "attacker" and s.pps < 0.3 * cfg["attack_pps"]["low"] for s in env.src])
            acts = np.zeros(env.n, int)
            while True:
                m = info["decide"]
                if m.any():
                    acts[m] = pol.decide(obs[m])
                obs, _, _, tr, info = env.step(acts)
                if env._attack_active():
                    av.append(info["availability"])
                    for i, s in enumerate(env.src):
                        key = "leg" if not env.labels[i] else ("atk_furtivo" if low[i] else "atk")
                        cnt[key][s.action] += 1
                if tr:
                    break
        print(f"\n{p}: disponibilidade média sob ataque = {np.mean(av):.3f}")
        for key, c in cnt.items():
            if c.sum():
                print(f"  {key:12s}", "  ".join(f"{ACTION_NAMES[j]}={100 * c[j] / c.sum():5.1f}%" for j in range(4)))


if __name__ == "__main__":
    main()
