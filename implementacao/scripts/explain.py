"""Importância dos atributos por permutação sobre os registros de decisão.

Para cada atributo, embaralha a coluna nas observações registradas pelo
controlador e mede a fração de decisões do agente que mudam.
Uso: python scripts/explain.py [--runs results/runs] [--policy DQN]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

IMPL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(IMPL))
from ddosrl.common import ACTION_NAMES, GLOBAL_FEATURES, LOCAL_FEATURES  # noqa: E402
from ddosrl.policies import load_policy                                  # noqa: E402

NAMES = LOCAL_FEATURES + GLOBAL_FEATURES + [f"prev_{a}" for a in ACTION_NAMES] + ["tau"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(IMPL / "results" / "runs"))
    ap.add_argument("--policy", default="DQN")
    ap.add_argument("--repeats", type=int, default=5)
    a = ap.parse_args()
    X = []
    for f in Path(a.runs).glob(f"*/{a.policy}/seed*/decisions.jsonl"):
        for line in open(f):
            d = json.loads(line)
            if d.get("attack_active"):
                X += [s["obs"] for s in d["sources"].values() if s["decided"]]
    if not X:
        sys.exit("sem registros")
    X = np.asarray(X, np.float32)
    pol = load_policy(a.policy)
    base = pol.decide(X)
    rng = np.random.default_rng(0)
    imp = []
    for j in range(X.shape[1]):
        ch = []
        for _ in range(a.repeats):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            ch.append(np.mean(pol.decide(Xp) != base))
        imp.append(np.mean(ch))
    order = np.argsort(imp)[::-1]
    out = IMPL / "results" / "tables" / f"importancia_{a.policy}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf8") as f:
        f.write("atributo,fracao_decisoes_alteradas\n")
        for j in order:
            f.write(f"{NAMES[j]},{imp[j]:.4f}\n")
    print(f"{len(X)} decisões; ações: {np.bincount(base, minlength=4)}")
    for j in order[:8]:
        print(f"  {NAMES[j]:15s} {imp[j]:.3f}")


if __name__ == "__main__":
    main()
