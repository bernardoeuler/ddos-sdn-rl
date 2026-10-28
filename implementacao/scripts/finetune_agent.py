"""Ajuste fino do agente com episódios no Mininet (fase 2 do treinamento).

Cada episódio roda a política atual com exploração epsilon em um cenário de
treinamento; as transições (decisão -> próxima decisão da mesma origem) são
reconstruídas de decisions.jsonl, com a recompensa da Eq. 2 calculada a partir
da disponibilidade medida nos clientes e da estimativa de ataque no gargalo.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

IMPL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(IMPL))
sys.path.insert(0, str(IMPL / "scripts"))
from ddosrl.agent.ddqn import DDQNAgent                       # noqa: E402
from ddosrl.common import RewardTerms, load_config, reward    # noqa: E402
import run_experiment as R                                    # noqa: E402


def transitions(run_dir: Path, cfg, gamma: float):
    recs = [json.loads(l) for l in open(run_dir / "decisions.jsonl")]
    out = []
    pending = {}
    for rec in recs:
        terms = RewardTerms(rec["availability"], rec["attack_util"])
        for n, s in rec["sources"].items():
            if s["decided"]:
                p = pending.get(n)
                if p is not None:
                    out.append((p[0], p[1], p[2], s["obs"], gamma ** p[3]))
                pending[n] = [s["obs"], s["action"], 0.0, 0]
            p = pending.get(n)
            if p is not None:
                # recompensa por efeito medido, como no simulador
                r = reward(cfg, s["action"], s["prev"], s["label"], terms,
                           mitigated=s.get("mit"), harm=s.get("harm"))
                p[2] += (gamma ** p[3]) * r
                p[3] += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default=str(IMPL / "models" / "ddqn_full_sim_s1"))
    ap.add_argument("--out", default=str(IMPL / "models" / "ddqn_full_s1"))
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--epsilon", type=float, default=0.1)
    ap.add_argument("--updates", type=int, default=300, help="passos de gradiente por episódio")
    ap.add_argument("--lr", type=float, default=3e-5)
    a = ap.parse_args()
    cfg = load_config()
    cfg["agent"]["lr"] = a.lr
    agent = DDQNAgent(cfg, seed=7)
    agent.load(a.init)
    work = IMPL / "models" / "_finetune"
    work.mkdir(parents=True, exist_ok=True)
    train_scen = [k for k, v in cfg["scenarios"].items() if v.get("train")]
    rng = np.random.default_rng(7)
    log = IMPL / "results" / "training" / "finetune_log.jsonl"
    for ep in range(a.episodes):
        agent.save(work / "ddqn_full_s1", {"finetune_episode": ep})
        scen = str(rng.choice(train_scen))
        seed = 50_000 + ep
        t0 = time.time()
        m = R.run("DQN", scen, seed, cfg, str(IMPL / "results" / "finetune_runs"),
                  epsilon=a.epsilon, model_dir=work)
        run_dir = IMPL / "results" / "finetune_runs" / scen / "DQN" / f"seed{seed}"
        tr = transitions(run_dir, cfg, cfg["agent"]["gamma"])
        for s, act, r, s2, g in tr:
            agent.buf.add(np.asarray(s, np.float32), act, r, np.asarray(s2, np.float32), g, 0.0)
        losses = [agent.update() for _ in range(a.updates)] if len(agent.buf) >= 1000 else []
        rec = {"episode": ep, "scenario": scen, "transitions": len(tr), "buffer": len(agent.buf),
               "loss": float(np.nanmean([l for l in losses if l is not None])) if losses else None,
               "tab": m["tab"], "availability": m["availability"],
               "legit_restricted_any": m["legit_restricted_any"], "elapsed_s": round(time.time() - t0)}
        print(json.dumps(rec), flush=True)
        with open(log, "a", encoding="utf8") as f:
            f.write(json.dumps(rec) + "\n")
    meta = json.loads(Path(a.init).with_suffix(".json").read_text())
    meta.update({"finetune_episodes": a.episodes, "finetune_epsilon": a.epsilon,
                 "finetune_updates_per_episode": a.updates, "finetune_lr": a.lr})
    agent.save(a.out, meta)
    shutil.rmtree(work, ignore_errors=True)
    print("salvo em", a.out)


if __name__ == "__main__":
    main()
