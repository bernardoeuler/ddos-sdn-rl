"""Agrega as rodadas, calcula IC 95% (bootstrap), testes de Wilcoxon com Holm,
gera tabelas (CSV e LaTeX) e figuras para as Seções 7 e 8 do artigo.

Uso: python scripts/analyze.py [--runs results/runs]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

IMPL = Path(__file__).resolve().parents[1]
OUT = IMPL / "results"
QP1 = ["S1L", "S1M", "S1H", "S2L", "S2M", "S2H", "S3L", "S3M", "S3H"]
QP2 = ["S4", "S5", "S6", "S7", "S8"]
METHODS = ["B0", "B1", "B2", "B3", "B4", "DQN"]
NAMES = {"B0": "B0 -- Sem defesa", "B1": "B1 -- Limiar fixo", "B2": "B2 -- Entropia",
         "B3": "B3 -- RF", "B4": "B4 -- RF graduado", "DQN": "Agente (proposto)",
         "DQN2": "Agente \\{permitir, bloquear\\}", "DQNLF": "Agente sem rótulos"}


def load(runs: Path) -> pd.DataFrame:
    rows = []
    for f in runs.glob("*/*/seed*/metrics.json"):
        m = json.loads(f.read_text())
        m.pop("actions", None)
        m.pop("reaction_s", None)
        rows.append(m)
    df = pd.DataFrame(rows)
    if df.empty:
        sys.exit("nenhuma rodada encontrada em " + str(runs))
    for c in ("reaction_s_mean", "recovery_s"):
        df[c] = df[c].replace([np.inf], np.nan)
    return df


def boot_ci(x, n=2000, seed=0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan
    if len(x) == 1:
        return x[0], np.nan
    rng = np.random.default_rng(seed)
    means = rng.choice(x, (n, len(x))).mean(1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return x.mean(), (hi - lo) / 2


def fmt(mean, ci, scale=1.0, dec=1):
    if not np.isfinite(mean):
        return "--"
    s = f"{mean * scale:.{dec}f}"
    return s + (f" $\\pm$ {ci * scale:.{dec}f}" if np.isfinite(ci) else "")


def table(df, scenarios, methods, cols, path_stem, caption=None):
    """cols: lista de (coluna, título, escala, casas)."""
    sub = df[df.scenario.isin(scenarios) & df.policy.isin(methods)]
    recs, tex = [], []
    for p in methods:
        d = sub[sub.policy == p]
        if d.empty:
            continue
        row = {"method": p, "n_runs": len(d)}
        cells = []
        for c, _, sc, dec in cols:
            m, ci = boot_ci(d[c])
            row[c], row[c + "_ci"] = m, ci
            cells.append(fmt(m, ci, sc, dec))
        recs.append(row)
        tex.append(f"{NAMES.get(p, p)} & " + " & ".join(cells) + " \\\\")
    pd.DataFrame(recs).to_csv(f"{path_stem}.csv", index=False)
    Path(f"{path_stem}.tex").write_text("\n".join(tex) + "\n", encoding="utf8")
    return pd.DataFrame(recs)


def wilcoxon_holm(df, scenarios, metric, ref="DQN", others=("B1", "B2", "B3", "B4")):
    from scipy.stats import wilcoxon
    sub = df[df.scenario.isin(scenarios)]
    piv = sub.pivot_table(index=["scenario", "seed"], columns="policy", values=metric)
    res = []
    for o in others:
        if ref not in piv or o not in piv:
            continue
        pair = piv[[ref, o]].dropna()
        d = pair[ref] - pair[o]
        if len(pair) < 2 or np.allclose(d, 0):
            p = np.nan
        else:
            p = float(wilcoxon(pair[ref], pair[o], zero_method="zsplit").pvalue)
        # tamanho de efeito: correlação rank-biserial pareada
        pos, neg = (d > 0).sum(), (d < 0).sum()
        rbc = (pos - neg) / max(1, pos + neg)
        res.append({"metric": metric, "vs": o, "n_pairs": len(pair),
                    "mean_diff": float(d.mean()) if len(d) else np.nan, "p": p, "rank_biserial": rbc})
    res = pd.DataFrame(res)
    if not res.empty:                       # correção de Holm
        valid = res.p.notna()
        order = res[valid].sort_values("p").index
        m = valid.sum()
        adj, running = {}, 0.0
        for k, i in enumerate(order):
            running = max(running, min(1.0, (m - k) * res.loc[i, "p"]))
            adj[i] = running
        res["p_holm"] = res.index.map(adj)
    return res


def figures(df, runs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir = OUT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    curves = sorted((OUT / "training").glob("curve_full_s*.csv"))
    if curves:
        plt.figure(figsize=(6, 3))
        for c in curves:
            d = pd.read_csv(c)
            plt.plot(d.updates, d.eval_return, label=c.stem.replace("curve_", ""))
        plt.xlabel("passos de gradiente")
        plt.ylabel("retorno médio (avaliação)")
        plt.grid(alpha=.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / "curva_aprendizado.pdf")
        plt.savefig(fig_dir / "curva_aprendizado.png", dpi=150)
        plt.close()
    # série temporal do cenário pulsado
    for scen in ("S4", "S1H"):
        plt.figure(figsize=(6, 3))
        plotted = False
        for p in ("B1", "B3", "DQN"):
            dirs = sorted((runs / scen / p).glob("seed*"))
            if not dirs:
                continue
            d = dirs[0]
            ev = json.loads((d / "events.json").read_text())
            rows = []
            for f in (d / "clients").glob("*.csv"):
                rows += [tuple(map(float, l.split(","))) for l in open(f) if l.count(",") == 2]
            if not rows:
                continue
            r = np.array(rows)
            t = r[:, 0] - ev["t0"]
            bins = np.arange(0, t.max() + 1, 2)
            av = [r[(t >= b) & (t < b + 2), 1].mean() if ((t >= b) & (t < b + 2)).any() else np.nan for b in bins]
            plt.plot(bins, av, label=NAMES[p])
            plotted = True
        if plotted:
            w = json.loads((dirs[0] / "events.json").read_text())
            plt.axvspan(w["window"][0] - w["t0"], w["window"][1] - w["t0"], color="0.9", zorder=0)
            plt.xlabel("tempo (s)")
            plt.ylabel("disponibilidade")
            plt.ylim(-0.05, 1.05)
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(fig_dir / f"serie_{scen}.pdf")
            plt.savefig(fig_dir / f"serie_{scen}.png", dpi=150)
        plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(OUT / "runs"))
    a = ap.parse_args()
    runs = Path(a.runs)
    df = load(runs)
    tdir = OUT / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(tdir / "all_runs.csv", index=False)

    t1 = table(df, QP1, METHODS, [("tab", "TAB", 100, 1), ("legit_restricted_any", "Legít. bloq.", 100, 1),
                                  ("availability", "Disp.", 100, 1), ("latency_p95_ms", "Lat. p95", 1, 0),
                                  ("reaction_s_mean", "Reação", 1, 1)], tdir / "qp1_eficacia")
    rows = []
    for s in QP2:
        d = df[df.scenario == s]
        cells = []
        for p in ("B1", "B3", "B4", "DQN"):
            e = d[d.policy == p]
            a_m, a_ci = boot_ci(e.availability)
            f_m, _ = boot_ci(e.legit_restricted_any)
            cells.append(f"{fmt(a_m, a_ci, 100, 1)} / {fmt(f_m, np.nan, 100, 0)}")
        rows.append(f"{s} & " + " & ".join(cells) + " \\\\")
    (tdir / "qp2_adaptacao.tex").write_text("\n".join(rows) + "\n", encoding="utf8")
    table(df, QP2, ["B1", "B3", "B4", "DQN"], [("availability", "Disp.", 100, 1),
                                                ("legit_restricted_any", "Bloq.", 100, 1)], tdir / "qp2_adaptacao_agregado")
    table(df, ["S1H", "S4", "S5", "S8"], ["DQN", "DQN2", "DQNLF"],
          [("tab", "TAB", 100, 1), ("legit_restricted_any", "Legít.", 100, 1), ("availability", "Disp.", 100, 1),
           ("reaction_s_mean", "Reação", 1, 1)], tdir / "qp3_ablacao")
    table(df, ["S1H"], ["B1", "B3", "DQN"], [("ctrl_cpu_pct", "CPU", 1, 1), ("ctrl_rss_mb", "Mem", 1, 0),
                                             ("of_msgs_per_s", "Msgs", 1, 1), ("decision_ms_mean", "Inf", 1, 2)],
          tdir / "qp4_custo")
    tests = pd.concat([wilcoxon_holm(df, QP1, m) for m in ("availability", "legit_restricted_any", "tab")]
                      + [wilcoxon_holm(df, QP2, "availability")], keys=["QP1"] * 3 + ["QP2"])
    tests.to_csv(tdir / "wilcoxon_holm.csv")
    figures(df, runs)
    print(df.groupby(["scenario", "policy"]).size().unstack(fill_value=0).to_string())
    print("\nQP1:\n", t1.round(3).to_string(index=False))
    print("\nTestes:\n", tests.round(4).to_string())


if __name__ == "__main__":
    main()
