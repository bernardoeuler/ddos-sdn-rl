"""Calibra os baselines B1/B2 e treina o Random Forest (B3/B4).

Dados: cenários de treinamento (S0-S3). Por padrão usa traços do simulador; com
--traces usa os registros de decisão coletados no Mininet (decisions.jsonl),
que contêm observações e rótulos.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ddosrl.common import N_LOCAL, ROOT, load_config            # noqa: E402
from ddosrl.policies import I_ENT, I_PPS                         # noqa: E402
from ddosrl.sim.env import DDoSSimEnv                            # noqa: E402


def sim_dataset(cfg, episodes_per_scen: int, seed: int):
    """Observações com política 'permitir tudo' (o detector vê o tráfego bruto)."""
    # mesmas condições do agente: inclui atacantes furtivos e legítimos intensos
    env = DDoSSimEnv(cfg, ambiguous=True)
    X, y, ent_normal, scen = [], [], [], []
    k = 0
    for name in env.scen_names:
        for _ in range(episodes_per_scen):
            obs, info = env.reset(seed=seed + k, options={"scenario": name})
            k += 1
            while True:
                obs, _, _, trunc, info = env.step(np.zeros(env.n, int))
                active = env._attack_active()
                lab = info["labels"] & active
                X.append(obs)
                y.append(lab)
                scen += [name] * len(lab)
                if name == "S0":
                    ent_normal.append(obs[0, I_ENT])
                if trunc:
                    break
    return np.concatenate(X), np.concatenate(y).astype(int), np.array(ent_normal), np.array(scen)


def trace_dataset(paths):
    X, y, ent_normal, scen = [], [], [], []
    for p in paths:
        for line in open(p, encoding="utf8"):
            d = json.loads(line)
            if "sources" not in d:
                continue
            for s in d["sources"].values():
                X.append(s["obs"])
                y.append(int(s["label"] and d.get("attack_active", True)))
                scen.append(d.get("scenario", "?"))
                if d.get("scenario") == "S0":
                    ent_normal.append(s["obs"][I_ENT])
    return np.array(X, np.float32), np.array(y), np.array(ent_normal), np.array(scen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--seed", type=int, default=500)
    ap.add_argument("--traces", nargs="*", help="decisions.jsonl dos cenários de treino")
    args = ap.parse_args()
    cfg = load_config()
    if args.traces:
        X, y, ent, scen = trace_dataset(args.traces)
        src = "mininet"
    else:
        X, y, ent, scen = sim_dataset(cfg, args.episodes, args.seed)
        src = "sim"
    legit_pps = X[y == 0, I_PPS]
    thr_pps = float(np.percentile(legit_pps, 99) * 1.2)
    thr_ent = float(np.percentile(ent, 1) * 0.95) if len(ent) else 0.8
    rf = RandomForestClassifier(n_estimators=100, max_depth=12, class_weight="balanced",
                                n_jobs=-1, random_state=0)
    idx = np.random.default_rng(0).permutation(len(X))
    cut = int(0.8 * len(X))
    tr, te = idx[:cut], idx[cut:]
    rf.fit(X[tr, :N_LOCAL], y[tr])
    print(classification_report(y[te], rf.predict(X[te, :N_LOCAL]), digits=4))
    rf.fit(X[:, :N_LOCAL], y)
    out = ROOT / "models" / "baselines.pkl"
    out.parent.mkdir(exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump({"thr_pps": thr_pps, "thr_entropy": thr_ent, "rf": rf, "source": src,
                     "n_samples": int(len(X))}, f)
    print(f"B1 thr_pps(norm)={thr_pps:.5f}  B2 thr_entropy={thr_ent:.4f}  amostras={len(X)} ({src})")


if __name__ == "__main__":
    main()
