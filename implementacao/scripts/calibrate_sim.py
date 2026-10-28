"""Calibra o simulador com traços do Mininet sem defesa (B0, cenários S0-S3).

1. Parâmetros de tráfego estimados diretamente dos atributos observados
   (taxa efetiva do hping3, tamanho de pacote, razão de resposta, perfil HTTP).
2. Parâmetros de dinâmica (fila SYN e perda no gargalo) por busca em grade,
   minimizando o erro quadrático da disponibilidade sem defesa por cenário.
Saída: configs/sim_calibration.json e results/calibration/validation.csv.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np

IMPL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(IMPL))
from ddosrl.common import LOCAL_FEATURES, PPS_NORM, RTT_NORM_MS, N_LOCAL, load_config  # noqa: E402
from ddosrl.sim.env import CALIBRATION_FILE, DEFAULT_PARAMS, DDoSSimEnv               # noqa: E402

F = {n: i for i, n in enumerate(LOCAL_FEATURES)}


def load_runs(root: Path):
    runs = []
    for m in sorted(root.glob("*/B0/seed*/metrics.json")):
        d = m.parent
        ev = json.loads((d / "events.json").read_text())
        recs = [json.loads(l) for l in open(d / "decisions.jsonl")]
        runs.append({"metrics": json.loads(m.read_text()), "events": ev, "recs": recs,
                     "scenario": json.loads((d / "run.json").read_text())["scenario"]})
    return runs


def traffic_params(runs):
    p, acc = {}, {}
    legit, rtt = [], []
    for r in runs:
        ev = r["events"]
        t_on, t_off = ev["window"]
        for rec in r["recs"]:
            active = ev["attack_start"] and t_on + 3 <= rec["t"] < t_off   # ignora a janela de aquecimento
            if r["scenario"] == "S0":
                rtt.append(rec["global"][1] * RTT_NORM_MS)
            for s in rec["sources"].values():
                x = s["obs"][:N_LOCAL]
                if s["label"] and active:
                    t = ev["types"][s["ip"]]
                    acc.setdefault(t, []).append((x, ev["pps"][s["ip"]]))
                elif not s["label"] and s["ip"] != ev.get("bulk") and x[F["syn_ratio"]] > 0:
                    legit.append(x)
    for t, rows in acc.items():
        X = np.array([x for x, _ in rows])
        req = np.array([q for _, q in rows])
        p[f"attack_pps_scale_{t}"] = float(np.median(X[:, F["pps_out"]] * PPS_NORM / req))
        p[f"pkt_{t}"] = float(np.median(X[:, F["pkt_size"]]) * 1500)
        p[f"resp_ratio_{t}"] = float(np.median(X[:, F["resp_ratio"]]))
    if legit:
        # médias: com amostragem 1:64 há poucas amostras por intervalo e a mediana é instável
        L = np.array(legit)
        syn = np.mean(L[:, F["syn_ratio"]])
        out_pkts = 1.0 / max(syn, 1e-3)
        out_bytes = out_pkts * np.mean(L[:, F["pkt_size"]]) * 1500
        rr = np.mean(L[:, F["resp_ratio"]])
        p["http_out_pkts"] = float(out_pkts)
        p["http_out_bytes"] = float(out_bytes)
        p["http_resp_bytes"] = float(out_bytes * rr / max(1e-6, 1 - rr))
    if rtt:
        p["base_rtt_ms"] = float(np.median(rtt))
    return p


def sim_availability(cfg, params, scen, seeds):
    env = DDoSSimEnv(cfg, params=params)
    ex = cfg["experiment"]
    vals = []
    for s in seeds:
        env.reset(seed=s, options={"scenario": scen})
        av = []
        while True:
            _, _, _, tr, info = env.step(np.zeros(env.n, int))
            if ex["warmup_s"] <= env.t < ex["warmup_s"] + ex["attack_s"]:
                av.append(info["availability"])
            if tr:
                break
        vals.append(np.mean(av))
    return float(np.mean(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(IMPL / "results" / "calibration"))
    ap.add_argument("--seeds", type=int, default=4)
    a = ap.parse_args()
    cfg = load_config()
    runs = load_runs(Path(a.runs))
    if not runs:
        sys.exit("nenhuma rodada B0 encontrada")
    params = dict(DEFAULT_PARAMS)
    params.update(traffic_params(runs))
    target = {}
    for r in runs:
        target.setdefault(r["scenario"], []).append(r["metrics"]["availability"])
    target = {k: float(np.nanmean(v)) for k, v in target.items() if k != "S0"}

    grid = {"syn_backlog": [2048, 4096, 8192, 16384, 32768, 65536],
            "syn_hold_s": [1, 2, 3, 5, 7], "loss_exp": [2, 4]}
    seeds = list(range(900, 900 + a.seeds))
    best, best_err = None, np.inf
    for vals in itertools.product(*grid.values()):
        trial = dict(params, **dict(zip(grid, vals)))
        err = sum((sim_availability(cfg, trial, s, seeds) - t) ** 2 for s, t in target.items())
        if err < best_err:
            best, best_err = dict(zip(grid, vals)), err
    params.update(best)
    out = {k: v for k, v in params.items() if DEFAULT_PARAMS.get(k) != v}
    CALIBRATION_FILE.write_text(json.dumps(out, indent=2), encoding="utf8")

    val = IMPL / "results" / "calibration" / "validation.csv"
    val.parent.mkdir(parents=True, exist_ok=True)
    before = dict(DEFAULT_PARAMS)
    with open(val, "w", newline="", encoding="utf8") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "mininet_availability", "sim_default", "sim_calibrated"])
        for s, t in sorted(target.items()):
            w.writerow([s, round(t, 4), round(sim_availability(cfg, before, s, seeds), 4),
                        round(sim_availability(cfg, params, s, seeds), 4)])
    print(json.dumps(out, indent=2))
    print(f"erro quadrático da disponibilidade: {best_err:.4f}")
    print(val.read_text())


if __name__ == "__main__":
    main()
