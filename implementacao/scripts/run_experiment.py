"""Executa uma rodada (método x cenário x semente) no Mininet e calcula métricas.

Precisa de root (Mininet/OVS). Exemplo, no Ubuntu do WSL:
  /opt/ddosrl-venv/bin/python scripts/run_experiment.py --policy DQN --scenario S1H --seed 1

Saída: results/runs/<cenário>/<política>/seed<k>/ com decisions.jsonl,
metrics.json, logs dos clientes e do controlador.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

IMPL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(IMPL))
from ddosrl.common import ACTION_NAMES, Action, load_config  # noqa: E402
from ddosrl.sdn.layout import layout                          # noqa: E402

PY = sys.executable
WORK = Path("/root/ddosrl_runs")


# --------------------------------------------------------------------- papéis
def assign_roles(cfg, scen_name, seed):
    """Sorteia atacantes e o cliente de upload contínuo (mesma lógica do simulador)."""
    rng = np.random.default_rng(seed)
    sc = cfg["scenarios"][scen_name]
    names = list(layout(cfg)["hosts"])
    n_att = 0
    if sc["type"] not in ("normal", "flash"):
        lo, hi = sc["attackers"]
        n_att = int(rng.integers(lo, hi + 1))
    order = [names[i] for i in rng.permutation(len(names))]
    attackers = order[:n_att]
    bulk = order[n_att] if rng.random() < 0.8 else None
    types = {}
    for k, h in enumerate(attackers):
        t = sc["type"]
        types[h] = ["syn", "udp"][k % 2] if t in ("pulse", "multi") else t
    rand_ports = {h: bool(rng.random() < 0.5) for h in attackers}
    pps = {h: cfg["attack_pps"][sc["intensity"]] * float(rng.uniform(0.6, 1.4)) for h in attackers}
    return attackers, bulk, types, rand_ports, pps


def hping_cmd(cfg, atype, pps, rand_ports, victim_ip):
    iv = max(1, int(1e6 / pps))
    if atype == "syn":
        return ["hping3", "-S", "-p", "++80" if rand_ports else "80", "-i", f"u{iv}", "-q", victim_ip]
    if atype == "udp":
        return ["hping3", "--udp", "-p", "++1" if rand_ports else "53", "-d",
                str(cfg["attack_payload"]["udp"]), "-i", f"u{iv}", "-q", victim_ip]
    return ["hping3", "-1", "-d", str(cfg["attack_payload"]["icmp"]), "-i", f"u{iv}", "-q", victim_ip]


# ------------------------------------------------------------------ topologia
def build_net(cfg, degraded):
    from mininet.link import TCLink
    from mininet.net import Mininet
    from mininet.node import OVSSwitch, RemoteController

    L, n = layout(cfg), cfg["network"]
    q = n["max_queue_pkts"]
    d = f"{n['link_delay_ms']}ms"
    net = Mininet(controller=None, switch=OVSSwitch, link=TCLink, autoSetMacs=False, build=False)
    net.addController("c0", controller=RemoteController, ip="127.0.0.1", port=6653)
    core = net.addSwitch("s1", dpid="%016x" % L["core_dpid"], protocols="OpenFlow13", failMode="secure")
    edges = {}
    for dpid in L["edge_dpids"]:
        edges[dpid] = net.addSwitch(f"s{dpid}", dpid="%016x" % dpid, protocols="OpenFlow13", failMode="secure")
        net.addLink(edges[dpid], core, port1=L["uplink_port"], port2=L["core_port_of_edge"][dpid],
                    bw=n["uplink_bw_mbps"], delay=d, max_queue_size=q)
    for name, h in L["hosts"].items():
        host = net.addHost(name, ip=h["ip"] + "/24", mac=h["mac"])
        kw = dict(bw=n["host_bw_mbps"], delay=d, max_queue_size=q)
        if degraded:
            kw.update(delay=f"{degraded['delay_ms']}ms", loss=degraded["loss_pct"])
        net.addLink(host, edges[h["dpid"]], port2=h["port"], **kw)
    for key, bw in (("victim", n["victim_bw_mbps"]), ("analysis", n["analysis_bw_mbps"]), ("monitor", 100)):
        s = L[key]
        host = net.addHost(s["name"], ip=s["ip"] + "/24", mac=s["mac"])
        net.addLink(host, core, port2=s["port"], bw=bw, delay=d, max_queue_size=q)
    return net


def sh(*cmd):
    return subprocess.run(list(cmd), capture_output=True, text=True)


# ------------------------------------------------------------------ execução
def run(policy, scen_name, seed, cfg, out_root, epsilon=0.0, model_dir=None, quick=None):
    sc = cfg["scenarios"][scen_name]
    ex = cfg["experiment"]
    if quick:
        ex = dict(ex, duration_s=quick[0], warmup_s=quick[1], attack_s=quick[2])
    L = layout(cfg)
    attackers, bulk, types, rand_ports, pps = assign_roles(cfg, scen_name, seed)
    att_ips = [L["hosts"][h]["ip"] for h in attackers]

    run_dir = WORK / scen_name / policy / f"seed{seed}"
    shutil.rmtree(run_dir, ignore_errors=True)
    (run_dir / "clients").mkdir(parents=True)
    run_cfg = {"run_dir": str(run_dir), "policy": policy, "scenario": scen_name, "seed": seed,
               "attackers": att_ips, "epsilon": epsilon,
               "model_dir": str(model_dir) if model_dir else None,
               "config": str(IMPL / "configs" / "default.yaml")}
    (run_dir / "run.json").write_text(json.dumps(run_cfg, indent=2))

    sh("mn", "-c")
    env = dict(os.environ, DDOSRL_RUN=str(run_dir / "run.json"), PYTHONPATH=str(IMPL))
    ctl_log = open(run_dir / "controller.log", "w")
    ctl = subprocess.Popen([PY, "-m", "ddosrl.sdn.launch_controller", "--ofp-tcp-listen-port", "6653"],
                           cwd=str(IMPL), env=env, stdout=ctl_log, stderr=subprocess.STDOUT)
    procs = []
    net = None
    try:
        net = build_net(cfg, sc.get("degraded"))
        net.build()
        net.start()
        net.staticArp()
        for dpid in L["edge_dpids"]:
            sh("ovs-vsctl", "--", "--id=@s", "create", "sflow", "agent=lo",
               f'target="127.0.0.1:{cfg["sflow"]["port"]}"', f"sampling={cfg['sflow']['sampling']}",
               f"header={cfg['sflow']['header_bytes']}", "polling=0", "--", "set", "bridge", f"s{dpid}", "sflow=@s")
        for _ in range(100):
            if (run_dir / "controller_ready").exists():
                break
            time.sleep(0.2)
        else:
            raise RuntimeError("controlador não ficou pronto; veja controller.log")

        srv, anl, mon = (net.get(L[k]["name"]) for k in ("victim", "analysis", "monitor"))
        www = run_dir / "www"
        www.mkdir()
        (www / "file").write_bytes(os.urandom(cfg["network"]["http_file_kb"] * 1024))
        srv.cmd("sysctl -qw net.ipv4.tcp_syncookies=0 net.ipv4.tcp_max_syn_backlog=1024 "
                "net.ipv4.tcp_synack_retries=2 net.core.somaxconn=1024")
        procs.append(srv.popen([PY, "-m", "ddosrl.sdn.traffic", "server", "--dir", str(www), "--port", "80"],
                               cwd=str(IMPL), env=env))
        srv.cmd("iperf3 -s -D")
        anl.cmd("sysctl -qw net.ipv4.ip_forward=1 net.ipv4.conf.all.send_redirects=0 "
                f"net.ipv4.conf.{anl.defaultIntf()}.send_redirects=0 net.ipv4.conf.all.rp_filter=0")
        lim = cfg["analysis_node"]["per_source_pps"]
        for name, h in L["hosts"].items():
            anl.cmd(f"iptables -A FORWARD -s {h['ip']} -m hashlimit --hashlimit-above {lim}/sec "
                    f"--hashlimit-burst {lim} --hashlimit-mode srcip --hashlimit-name {name} -j DROP")
        time.sleep(1.5)

        t0 = time.time() + 1.0
        t_on, t_off, t_end = t0 + ex["warmup_s"], t0 + ex["warmup_s"] + ex["attack_s"], t0 + ex["duration_s"]
        events = {"t0": t0, "attack_start": t_on if attackers else None,
                  "attack_end": t_off if attackers else None, "window": [t_on, t_off],
                  "pulse": [sc["on_s"], sc["off_s"]] if sc["type"] == "pulse" else None,
                  "attackers": att_ips, "bulk": L["hosts"][bulk]["ip"] if bulk else None,
                  "types": {L["hosts"][h]["ip"]: types[h] for h in attackers},
                  "pps": {L["hosts"][h]["ip"]: pps[h] for h in attackers}}
        (run_dir / "events.json").write_text(json.dumps(events, indent=2))

        url = f"http://{L['victim']['ip']}/file"
        sched = json.dumps([[t_on, t_off, ex["flash_crowd_factor"]]]) if sc["type"] == "flash" else ""
        rng = np.random.default_rng(seed + 7)
        for name in L["hosts"]:
            h = net.get(name)
            if name in attackers:
                continue
            if name == bulk:
                procs.append(h.popen(["iperf3", "-c", L["victim"]["ip"], "-b", f"{ex['bulk_mbps']}M",
                                      "-t", str(int(ex["duration_s"]) + 2)], stdout=subprocess.DEVNULL))
                continue
            rate = ex["legit_rate_rps"] * float(rng.uniform(0.5, 1.5))
            procs.append(h.popen([PY, "-m", "ddosrl.sdn.traffic", "http", "--url", url, "--rate", f"{rate:.3f}",
                                  "--timeout", str(ex["http_timeout_s"]), "--until", str(t_end),
                                  "--seed", str(seed * 100 + L["hosts"][name]["index"]),
                                  "--out", str(run_dir / "clients" / f"{name}.csv")]
                                 + (["--schedule", sched] if sched else []), cwd=str(IMPL), env=env))
        procs.append(mon.popen([PY, "-m", "ddosrl.sdn.traffic", "probe", "--target", L["victim"]["ip"],
                                "--out", str(run_dir / "probe_rtt.txt"), "--until", str(t_end)],
                               cwd=str(IMPL), env=env))

        def tx(hname):
            h = net.get(hname)
            out = h.cmd(f"cat /sys/class/net/{h.defaultIntf()}/statistics/tx_bytes "
                        f"/sys/class/net/{h.defaultIntf()}/statistics/tx_packets").split()
            return int(out[0]), int(out[1])

        res_samples = []
        tx0 = {}
        started = False
        while time.time() < t_end:
            now = time.time()
            if attackers and not started and now >= t_on:
                tx0 = {h: tx(h) for h in attackers}
                for h in attackers:
                    cmd = hping_cmd(cfg, types[h], pps[h], rand_ports[h], L["victim"]["ip"])
                    if sc["type"] == "pulse":
                        cmd = [PY, "-m", "ddosrl.sdn.traffic", "pulse", "--on", str(sc["on_s"]),
                               "--off", str(sc["off_s"]), "--until", str(t_off), "--"] + cmd
                        procs.append(net.get(h).popen(cmd, cwd=str(IMPL), env=env,
                                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
                    else:
                        procs.append(net.get(h).popen(["timeout", "-s", "INT", str(ex["attack_s"])] + cmd,
                                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
                started = True
            res_samples.append(proc_usage(ctl.pid, now))
            time.sleep(1.0)
        tx1 = {h: tx(h) for h in attackers}
        an_drop = analysis_drops(anl.cmd("iptables -L FORWARD -v -x -n"))
    finally:
        for p in procs:
            try:
                p.kill()
            except Exception:                           # noqa: BLE001
                pass
        if net is not None:
            net.stop()
        ctl.terminate()
        try:
            ctl.wait(5)
        except subprocess.TimeoutExpired:
            ctl.kill()
        ctl_log.close()
        sh("mn", "-c")

    (run_dir / "usage.json").write_text(json.dumps(res_samples))
    m = compute_metrics(cfg, run_dir, events, attackers, L, tx0, tx1, an_drop, res_samples)
    (run_dir / "metrics.json").write_text(json.dumps(m, indent=2))
    dest = Path(out_root) / scen_name / policy / f"seed{seed}"
    shutil.rmtree(dest, ignore_errors=True)
    shutil.copytree(run_dir, dest, ignore=shutil.ignore_patterns("www"))
    return m


def proc_usage(pid, now):
    try:
        st = open(f"/proc/{pid}/stat").read().split()
        cpu = (int(st[13]) + int(st[14])) / os.sysconf("SC_CLK_TCK")
        rss = int(re.search(r"VmRSS:\s+(\d+)", open(f"/proc/{pid}/status").read()).group(1)) / 1024
        return {"t": now, "cpu_s": cpu, "rss_mb": rss}
    except (OSError, AttributeError, IndexError):
        return {"t": now, "cpu_s": None, "rss_mb": None}


def analysis_drops(text):
    """Bytes (com cabeçalho Ethernet) descartados no nó de análise, por IP de origem."""
    out = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) > 8 and parts[2] == "DROP":
            pk, by, src = int(parts[0]), int(parts[1]), parts[7]
            out[src] = out.get(src, 0) + by + 14 * pk
    return out


# ------------------------------------------------------------------- métricas
def compute_metrics(cfg, run_dir, ev, attackers, L, tx0, tx1, an_drop, usage):
    t_on, t_off = ev["window"]
    att_ips = set(ev["attackers"])
    # disponibilidade e latência (clientes HTTP legítimos)
    rows = []
    for f in (run_dir / "clients").glob("*.csv"):
        for line in open(f):
            p = line.strip().split(",")
            if len(p) == 3:
                rows.append((float(p[0]), int(p[1]), float(p[2])))
    rows = np.array(rows) if rows else np.zeros((0, 3))
    win = rows[(rows[:, 0] >= t_on) & (rows[:, 0] < t_off)] if len(rows) else rows
    pre = rows[rows[:, 0] < t_on] if len(rows) else rows
    lat_ok = win[win[:, 1] == 1, 2] if len(win) else np.array([])
    avail = float(win[:, 1].mean()) if len(win) else float("nan")
    # recuperação: primeiro segundo após o início em que a disponibilidade (janela de 3 s) >= 95%
    recovery = float("nan")
    if len(win):
        bins = np.arange(t_on, t_off, 1.0)
        a = np.array([win[(win[:, 0] >= b) & (win[:, 0] < b + 3), 1].mean()
                      if ((win[:, 0] >= b) & (win[:, 0] < b + 3)).any() else np.nan for b in bins])
        drops = np.flatnonzero(a < 0.95)
        if len(drops) == 0:
            recovery = 0.0
        else:
            ok_after = np.flatnonzero((a >= 0.95) & (np.arange(len(a)) > drops[0]))
            recovery = float(ok_after[0]) if len(ok_after) else float("inf")

    # decisões do controlador: falsos positivos e tempo de reação
    recs = [json.loads(l) for l in open(run_dir / "decisions.jsonl")]
    recs_w = [r for r in recs if t_on <= r["t"] < t_off]
    legit = [n for n, h in L["hosts"].items() if h["ip"] not in att_ips]
    def frac_restricted(names, acts):
        if not recs_w or not names:
            return float("nan"), float("nan")
        any_ = np.mean([any(r["sources"][n]["action"] in acts for r in recs_w) for n in names])
        time_ = np.mean([np.mean([r["sources"][n]["action"] in acts for r in recs_w]) for n in names])
        return float(any_), float(time_)
    fp_any, fp_time = frac_restricted(legit, (Action.LIMIT, Action.BLOCK))
    rd_any, rd_time = frac_restricted(legit, (Action.REDIRECT,))
    react = []
    for n, h in L["hosts"].items():
        if h["ip"] not in att_ips:
            continue
        ts = [r["t"] for r in recs_w if r["sources"][n]["action"] != Action.ALLOW]
        react.append(ts[0] - t_on if ts else float("inf"))
    action_hist = {}
    for r in recs_w:
        for n, s in r["sources"].items():
            k = ("atk" if s["label"] else "leg") + "_" + ACTION_NAMES[s["action"]]
            action_hist[k] = action_hist.get(k, 0) + 1

    # taxa de ataque bloqueado (somente descartes das regras de mitigação)
    summ = json.loads((run_dir / "controller_summary.json").read_text())
    sent = sum(tx1[h][0] - tx0[h][0] for h in attackers) if attackers else 0
    mitig = 0
    for h in attackers:
        ip = L["hosts"][h]["ip"]
        m = summ["mitigated"].get(ip, {})
        mitig += m.get("block_bytes", 0) + m.get("meter_bytes", 0) + an_drop.get(ip, 0)
    tab = min(1.0, mitig / sent) if sent > 0 else float("nan")

    # sonda (RTT e perda)
    pr = []
    plog = run_dir / "probe_rtt.txt.log"
    if plog.exists():
        pr = [tuple(map(float, l.split(","))) for l in open(plog) if "," in l]
    pr_w = np.array([r for t, r in pr if t_on <= t < t_off]) if pr else np.array([])
    # custo no controlador
    u = [x for x in usage if x["cpu_s"] is not None and t_on <= x["t"] < t_off]
    cpu = (100 * (u[-1]["cpu_s"] - u[0]["cpu_s"]) / (u[-1]["t"] - u[0]["t"])) if len(u) > 1 else float("nan")
    rss = max((x["rss_mb"] for x in u), default=float("nan"))
    msgs = ((recs_w[-1]["msgs_out"] + recs_w[-1]["msgs_in"] - recs_w[0]["msgs_out"] - recs_w[0]["msgs_in"])
            / max(1e-9, recs_w[-1]["t"] - recs_w[0]["t"])) if len(recs_w) > 1 else float("nan")
    dec = np.array([r["decide_ms"] + r["apply_ms"] for r in recs_w]) if recs_w else np.array([np.nan])
    return {
        "scenario": ev and json.loads((run_dir / "run.json").read_text())["scenario"],
        "policy": json.loads((run_dir / "run.json").read_text())["policy"],
        "seed": json.loads((run_dir / "run.json").read_text())["seed"],
        "n_attackers": len(attackers),
        "tab": tab, "attack_sent_mb": sent / 1e6, "attack_mitigated_mb": mitig / 1e6,
        "legit_restricted_any": fp_any, "legit_restricted_time": fp_time,
        "legit_redirected_any": rd_any, "legit_redirected_time": rd_time,
        "availability": avail, "availability_pre": float(pre[:, 1].mean()) if len(pre) else float("nan"),
        "requests": int(len(win)),
        "latency_p50_ms": float(np.median(lat_ok)) if len(lat_ok) else float("nan"),
        "latency_p95_ms": float(np.percentile(lat_ok, 95)) if len(lat_ok) else float("nan"),
        "rtt_p50_ms": float(np.median(pr_w[pr_w < 1000])) if (pr_w < 1000).any() else float("nan"),
        "probe_loss": float(np.mean(pr_w >= 1000)) if len(pr_w) else float("nan"),
        "reaction_s_mean": float(np.mean(react)) if react else float("nan"),
        "reaction_s": react,
        "recovery_s": recovery,
        "ctrl_cpu_pct": cpu, "ctrl_rss_mb": rss, "of_msgs_per_s": msgs,
        "decision_ms_mean": float(np.nanmean(dec)), "decision_ms_p95": float(np.nanpercentile(dec, 95)),
        "actions": action_hist,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, help="B0 B1 B2 B3 B4 DQN DQN2 DQNLF")
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epsilon", type=float, default=0.0)
    ap.add_argument("--out", default=str(IMPL / "results" / "runs"))
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--quick", nargs=3, type=int, metavar=("DUR", "WARM", "ATK"),
                    help="durações reduzidas para testes de integração")
    a = ap.parse_args()
    cfg = load_config()
    m = run(a.policy, a.scenario, a.seed, cfg, a.out, a.epsilon, a.model_dir, a.quick)
    print(json.dumps({k: v for k, v in m.items() if k != "actions"}, indent=1))
    print("ações:", m["actions"])


if __name__ == "__main__":
    main()
