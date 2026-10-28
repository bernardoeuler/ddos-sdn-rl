"""Rodadas curtas no Mininet para resultados preliminares rápidos.

Durações reduzidas (60 s: 15 s de aquecimento, 35 s de ataque); não substituem
a campanha completa de 180 s. Saída em results/quick_runs/ e quick_summary.csv.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

IMPL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(IMPL))
sys.path.insert(0, str(IMPL / "scripts"))
from ddosrl.common import load_config    # noqa: E402
import run_experiment as R               # noqa: E402

KEYS = ["scenario", "policy", "seed", "n_attackers", "tab", "legit_restricted_any",
        "legit_redirected_any", "availability", "latency_p95_ms", "reaction_s_mean",
        "decision_ms_mean", "ctrl_cpu_pct"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", nargs="+", default=["B1", "B3", "DQN"])
    ap.add_argument("--scenarios", nargs="+", default=["S1H", "S2H", "S5"])
    ap.add_argument("--seed", type=int, default=1001)
    a = ap.parse_args()
    cfg = load_config()
    out = IMPL / "results" / "quick_runs"
    summary = IMPL / "results" / "quick_summary.csv"
    new = not summary.exists()
    with open(summary, "a", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, KEYS, extrasaction="ignore")
        if new:
            w.writeheader()
        for s in a.scenarios:
            for p in a.policies:
                t0 = time.time()
                m = R.run(p, s, a.seed, cfg, str(out), quick=(60, 15, 35))
                w.writerow(m)
                f.flush()
                print(f"{s} {p}: TAB={m['tab']:.3f} disp={m['availability']:.3f} "
                      f"legit_restr={m['legit_restricted_any']:.2f} "
                      f"reacao={m['reaction_s_mean']:.1f}s acoes={m['actions']} ({time.time() - t0:.0f}s)",
                      flush=True)


if __name__ == "__main__":
    main()
