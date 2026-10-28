"""Executa uma campanha de experimentos no Mininet (retomável).

Pula rodadas que já têm metrics.json. Exemplo:
  python scripts/run_campaign.py --policies B0 B1 B2 B3 B4 DQN --scenarios all --seeds 1 2 3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

IMPL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(IMPL))
sys.path.insert(0, str(IMPL / "scripts"))
from ddosrl.common import load_config    # noqa: E402
import run_experiment as R               # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", nargs="+", required=True)
    ap.add_argument("--scenarios", nargs="+", default=["all"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1])
    ap.add_argument("--out", default=str(IMPL / "results" / "runs"))
    ap.add_argument("--epsilon", type=float, default=0.0)
    ap.add_argument("--model-dir", default=None)
    a = ap.parse_args()
    cfg = load_config()
    scen = list(cfg["scenarios"]) if a.scenarios == ["all"] else \
        [k for k, v in cfg["scenarios"].items() if v.get("train")] if a.scenarios == ["train"] else a.scenarios
    base = cfg["experiment"]["base_seed"]
    jobs = [(s, p, k) for k in a.seeds for s in scen for p in a.policies]
    log = Path(a.out) / "campaign.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    for i, (s, p, k) in enumerate(jobs, 1):
        seed = base + k                      # mesma semente para todos os métodos
        dest = Path(a.out) / s / p / f"seed{seed}" / "metrics.json"
        if dest.exists():
            continue
        t0 = time.time()
        try:
            m = R.run(p, s, seed, cfg, a.out, a.epsilon, a.model_dir)
            msg = (f"[{i}/{len(jobs)}] {s} {p} seed{seed}: TAB={m['tab']:.3f} "
                   f"disp={m['availability']:.3f} fp={m['legit_restricted_any']:.2f} "
                   f"({time.time() - t0:.0f}s)")
        except Exception:                    # noqa: BLE001 — registra e segue para a próxima
            msg = f"[{i}/{len(jobs)}] {s} {p} seed{seed}: FALHOU\n{traceback.format_exc()}"
        print(msg, flush=True)
        with open(log, "a", encoding="utf8") as f:
            f.write(time.strftime("%H:%M:%S ") + msg + "\n")


if __name__ == "__main__":
    main()
