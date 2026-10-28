"""Avalia as políticas no simulador (sanidade antes do Mininet; não substitui a emulação).

Uso: python scripts/eval_sim.py --policies B0 B1 B2 B3 B4 DQN --seeds 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

IMPL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(IMPL))
from ddosrl.common import Action, load_config     # noqa: E402
from ddosrl.policies import load_policy           # noqa: E402
from ddosrl.sim.env import DDoSSimEnv             # noqa: E402


def episode(env, pol, scen, seed):
    obs, info = env.reset(seed=seed, options={"scenario": scen})
    acts = np.zeros(env.n, int)
    av, restricted = [], np.zeros(env.n, bool)
    while True:
        m = info["decide"]
        if m.any():
            acts[m] = pol.decide(obs[m])
        obs, _, _, tr, info = env.step(acts)
        if env.t_on <= env.t < env.t_off:
            av.append(info["availability"])
            cur = np.array([s.action in (Action.LIMIT, Action.BLOCK) for s in env.src])
            restricted |= cur & ~env.labels
        if tr:
            break
    legit = ~env.labels
    return {"availability": float(np.mean(av)),
            "tab": env.mitigated_bytes / env.sent_bytes if env.sent_bytes else np.nan,
            "legit_restricted_any": float(restricted[legit].mean()) if legit.any() else np.nan}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", nargs="+", default=["B0", "B1", "B2", "B3", "B4", "DQN"])
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    cfg = load_config()
    env = DDoSSimEnv(cfg)
    rows = []
    for p in a.policies:
        pol = load_policy(p)
        if p == "DQN2":
            env.two_actions = True
        for scen in cfg["scenarios"]:
            for k in range(a.seeds):
                rows.append({"policy": p, "scenario": scen, "seed": k, **episode(env, pol, scen, 70_000 + k)})
        env.two_actions = False
    df = pd.DataFrame(rows)
    out = IMPL / "results" / "sim_eval.csv"
    df.to_csv(out, index=False)
    print(df.pivot_table(index="scenario", columns="policy", values="availability").round(3).to_string())
    print()
    print(df.pivot_table(index="scenario", columns="policy", values="legit_restricted_any").round(2).to_string())


if __name__ == "__main__":
    main()
