"""Aplicação os-ken: monitor, extrator de atributos, agente e atuador.

Tabelas OpenFlow nos switches de borda:
  tabela 0  prio 200  sobreposição de mitigação por origem (meter / redirecionar / descarte)
            prio 100  contagem por origem (in_port, ipv4_src) -> tabela 1
            prio  90  contagem por destino (ipv4_dst)          -> tabela 1
            prio   0  -> tabela 1
  tabela 1  encaminhamento L2 proativo por MAC; ausência -> controlador (packet-in)
No núcleo, a tabela 0 apenas envia à tabela 1.

A execução é configurada pelo JSON indicado em DDOSRL_RUN (gerado pelo runner).
"""
from __future__ import annotations

import json
import os
import socket
import time
from pathlib import Path

import numpy as np
from os_ken.base import app_manager
from os_ken.controller import ofp_event
from os_ken.controller.handler import (CONFIG_DISPATCHER, MAIN_DISPATCHER,
                                       set_ev_cls)
from os_ken.lib import hub
from os_ken.ofproto import ofproto_v1_3

from ..common import ACTION_NAMES, Action, load_config
from ..features import FeatureExtractor, GlobalSample, SourceSample
from ..policies import load_policy
from .layout import layout
from .sflow import parse_datagram

P_OVERLAY, P_SRC, P_DST = 200, 100, 90


class DDoSController(app_manager.OSKenApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        run = json.loads(Path(os.environ["DDOSRL_RUN"]).read_text(encoding="utf8"))
        self.runcfg = run
        self.cfg = load_config(run.get("config"))
        self.lay = layout(self.cfg)
        self.run_dir = Path(run["run_dir"])
        self.dt = float(self.cfg["agent"]["dt_s"])
        self.block_s = int(self.cfg["agent"]["block_s"])
        self.limit_kbps = int(self.cfg["agent"]["limit_frac"] * self.cfg["network"]["victim_bw_mbps"] * 1000)
        self.an_pps = self.cfg["analysis_node"]["per_source_pps"]
        self.policy = load_policy(run["policy"], run.get("model_dir"))
        self.eps = float(run.get("epsilon", 0.0))
        self.rng = np.random.default_rng(run.get("seed", 0))
        self.attackers = set(run.get("attackers", []))
        self.fx = FeatureExtractor(self.cfg)

        self.hosts = self.lay["hosts"]                    # nome -> dict
        self.by_ip = {h["ip"]: n for n, h in self.hosts.items()}
        self.dps = {}
        self.action = {n: Action.ALLOW for n in self.hosts}
        self.lock_until = {n: 0.0 for n in self.hosts}
        self.changed_at = {n: time.time() for n in self.hosts}
        # contadores acumulados
        self.flow = {}                                    # (dpid, prio, ip) -> (pkts, bytes)
        self.removed = {n: [0, 0] for n in self.hosts}    # sobreposições removidas
        self.removed_block = {n: [0, 0] for n in self.hosts}
        self.pending_del = {}
        self.req_time = {}
        self.prev_out = {n: (0, 0) for n in self.hosts}
        self.prev_in = {n: 0 for n in self.hosts}
        self.meter_drop = {n: [0, 0] for n in self.hosts}  # pacotes/bytes descartados pelo meter
        self.prev_meter = {n: 0 for n in self.hosts}
        self.port_tx = {}
        self.prev_port = {}
        self.packet_in = 0
        self.msgs_out = 0
        self.msgs_in = 0
        self.samples = {}
        self._mp = {}
        self.avail_pos = {}
        self.tick = 0
        self.decide_ms = []
        self.dlog = open(self.run_dir / "decisions.jsonl", "a", encoding="utf8")
        self.attack_events = None
        hub.spawn(self._sflow_loop)
        hub.spawn(self._main_loop)
        self.logger.info("DDoSController: política=%s eps=%.2f", self.policy.name, self.eps)

    # ------------------------------------------------------------ utilitários
    def _send(self, dp, msg):
        self.msgs_out += 1
        dp.send_msg(msg)

    def _flow(self, dp, prio, match, inst, table=0, hard=0, flags=0, cmd=None):
        ofp, p = dp.ofproto, dp.ofproto_parser
        self._send(dp, p.OFPFlowMod(dp, table_id=table, priority=prio, match=match,
                                    instructions=inst, hard_timeout=hard, flags=flags,
                                    command=ofp.OFPFC_ADD if cmd is None else cmd))

    def _goto1(self, p):
        return [p.OFPInstructionGotoTable(1)]

    # ------------------------------------------------------- conexão/proativo
    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def _features(self, ev):
        try:
            self._setup_switch(ev)
        except Exception:                                  # noqa: BLE001
            self.logger.exception("falha ao configurar o switch")
            raise

    def _setup_switch(self, ev):
        dp = ev.msg.datapath
        ofp, p = dp.ofproto, dp.ofproto_parser
        self.msgs_in += 1
        self.dps[dp.id] = dp
        self._send(dp, p.OFPFlowMod(dp, command=ofp.OFPFC_DELETE, table_id=ofp.OFPTT_ALL,
                                    out_port=ofp.OFPP_ANY, out_group=ofp.OFPG_ANY,
                                    match=p.OFPMatch()))
        self._send(dp, p.OFPMeterMod(dp, command=ofp.OFPMC_DELETE, flags=0, meter_id=ofp.OFPM_ALL,
                                     bands=[]))
        L = self.lay
        # tabela 1: encaminhamento L2
        if dp.id == L["core_dpid"]:
            for h in self.hosts.values():
                self._l2(dp, h["mac"], L["core_port_of_edge"][h["dpid"]])
            for k in ("victim", "analysis", "monitor"):
                self._l2(dp, L[k]["mac"], L[k]["port"])
        else:
            for h in self.hosts.values():
                port = h["port"] if h["dpid"] == dp.id else L["uplink_port"]
                self._l2(dp, h["mac"], port)
            for k in ("victim", "analysis", "monitor"):
                self._l2(dp, L[k]["mac"], L["uplink_port"])
            for n, h in self.hosts.items():
                if h["dpid"] != dp.id:
                    continue
                self._flow(dp, P_SRC, self._src_match(p, h), self._goto1(p))
                self._flow(dp, P_DST, p.OFPMatch(eth_type=0x0800, ipv4_dst=h["ip"]), self._goto1(p))
                band = p.OFPMeterBandDrop(rate=self.limit_kbps, burst_size=max(1, self.limit_kbps // 10))
                self._send(dp, p.OFPMeterMod(dp, command=ofp.OFPMC_ADD,
                                             flags=ofp.OFPMF_KBPS | ofp.OFPMF_BURST | ofp.OFPMF_STATS,
                                             meter_id=h["index"], bands=[band]))
        self._flow(dp, 0, p.OFPMatch(), self._goto1(p))
        self._flow(dp, 0, p.OFPMatch(), [p.OFPInstructionActions(
            ofp.OFPIT_APPLY_ACTIONS, [p.OFPActionOutput(ofp.OFPP_CONTROLLER, ofp.OFPCML_NO_BUFFER)])], table=1)
        if len(self.dps) == 1 + len(L["edge_dpids"]):
            (self.run_dir / "controller_ready").write_text(str(time.time()))

    def _l2(self, dp, mac, port):
        p = dp.ofproto_parser
        self._flow(dp, 10, p.OFPMatch(eth_dst=mac),
                   [p.OFPInstructionActions(dp.ofproto.OFPIT_APPLY_ACTIONS, [p.OFPActionOutput(port)])],
                   table=1)

    def _src_match(self, p, h):
        return p.OFPMatch(in_port=h["port"], eth_type=0x0800, ipv4_src=h["ip"])

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in(self, ev):
        self.msgs_in += 1
        self.packet_in += 1

    # ---------------------------------------------------------------- atuação
    def apply(self, name: str, a: int):
        h = self.hosts[name]
        dp = self.dps.get(h["dpid"])
        if dp is None:
            return
        ofp, p = dp.ofproto, dp.ofproto_parser
        prev = self.action[name]
        if a == prev:
            return
        match = self._src_match(p, h)
        if prev != Action.ALLOW:
            # remove a sobreposição anterior; o que já foi contado passa para
            # "removed" agora e o FlowRemoved acrescenta só a diferença final
            known = self.flow.pop((h["dpid"], P_OVERLAY, h["ip"]), (0, 0, False))
            self.removed[name][0] += known[0]
            self.removed[name][1] += known[1]
            self.pending_del[name] = known[:2]
            self._send(dp, p.OFPFlowMod(dp, command=ofp.OFPFC_DELETE_STRICT, table_id=0,
                                        priority=P_OVERLAY, match=match,
                                        out_port=ofp.OFPP_ANY, out_group=ofp.OFPG_ANY))
        flags = ofp.OFPFF_SEND_FLOW_REM
        if a == Action.LIMIT:
            self._flow(dp, P_OVERLAY, match, [p.OFPInstructionMeter(h["index"])] + self._goto1(p), flags=flags)
        elif a == Action.REDIRECT:
            acts = [p.OFPActionSetField(eth_dst=self.lay["analysis"]["mac"])]
            self._flow(dp, P_OVERLAY, match,
                       [p.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, acts)] + self._goto1(p), flags=flags)
        elif a == Action.BLOCK:
            self._flow(dp, P_OVERLAY, match, [], hard=self.block_s, flags=flags)
            self.lock_until[name] = time.time() + self.block_s
        self.action[name] = a
        self.changed_at[name] = time.time()

    @set_ev_cls(ofp_event.EventOFPFlowRemoved, MAIN_DISPATCHER)
    def _flow_removed(self, ev):
        self.msgs_in += 1
        m = ev.msg
        if m.priority != P_OVERLAY or "ipv4_src" not in m.match:
            return
        name = self.by_ip.get(m.match["ipv4_src"])
        if not name:
            return
        known = self.pending_del.pop(name, None)
        if known is None:            # removida pelo switch (hard_timeout)
            known = self.flow.pop((m.datapath.id, P_OVERLAY, m.match["ipv4_src"]), (0, 0, False))[:2]
            self.removed[name][0] += known[0]
            self.removed[name][1] += known[1]
        self.removed[name][0] += max(0, m.packet_count - known[0])
        self.removed[name][1] += max(0, m.byte_count - known[1])
        if m.reason == m.datapath.ofproto.OFPRR_HARD_TIMEOUT:
            self.removed_block[name][0] += m.packet_count
            self.removed_block[name][1] += m.byte_count
            if self.action[name] == Action.BLOCK:
                self.action[name] = Action.ALLOW
                self.changed_at[name] = time.time()
                self.lock_until[name] = 0.0

    # -------------------------------------------------------------- estatística
    def _request_stats(self):
        for dpid, dp in self.dps.items():
            ofp, p = dp.ofproto, dp.ofproto_parser
            if dpid == self.lay["core_dpid"]:
                self._send(dp, p.OFPPortStatsRequest(dp, 0, ofp.OFPP_ANY))
            else:
                self.req_time[dpid] = time.time()
                self._send(dp, p.OFPFlowStatsRequest(dp, 0, 0, ofp.OFPP_ANY, ofp.OFPG_ANY, 0, 0, p.OFPMatch()))
                self._send(dp, p.OFPMeterStatsRequest(dp, 0, ofp.OFPM_ALL))

    def _multipart(self, ev, key):
        """Acumula respostas multipart; devolve o corpo completo ou None."""
        m = ev.msg
        k = (m.datapath.id, key)
        self._mp.setdefault(k, []).extend(m.body)
        if m.flags & m.datapath.ofproto.OFPMPF_REPLY_MORE:
            return None
        return self._mp.pop(k)

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def _flow_stats(self, ev):
        self.msgs_in += 1
        body = self._multipart(ev, "flow")
        if body is None:
            return
        dpid = ev.msg.datapath.id
        req_t = self.req_time.get(dpid, 0.0)
        for st in body:
            if st.table_id != 0 or st.priority not in (P_OVERLAY, P_SRC, P_DST):
                continue
            ip = st.match.get("ipv4_src") or st.match.get("ipv4_dst")
            if st.priority == P_OVERLAY:
                name = self.by_ip.get(ip)
                # resposta anterior à última troca de ação: contadores obsoletos
                if name is None or self.changed_at[name] > req_t or self.action[name] == Action.ALLOW:
                    continue
            self.flow[(dpid, st.priority, ip)] = (st.packet_count, st.byte_count,
                                                  len(st.instructions) == 0)

    @set_ev_cls(ofp_event.EventOFPMeterStatsReply, MAIN_DISPATCHER)
    def _meter_stats(self, ev):
        self.msgs_in += 1
        body = self._multipart(ev, "meter")
        if body is None:
            return
        by_idx = {h["index"]: n for n, h in self.hosts.items()}
        for st in body:
            n = by_idx.get(st.meter_id)
            if n and st.band_stats:
                self.meter_drop[n] = [st.band_stats[0].packet_band_count, st.band_stats[0].byte_band_count]

    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def _port_stats(self, ev):
        self.msgs_in += 1
        body = self._multipart(ev, "port")
        if body is None:
            return
        for st in body:
            self.port_tx[st.port_no] = st.tx_bytes

    # ------------------------------------------------------------------ sFlow
    def _sflow_loop(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", int(self.cfg["sflow"]["port"])))
        while True:
            data, _ = sock.recvfrom(65535)
            for s in parse_datagram(data):
                name = self.by_ip.get(s.src)
                if not name:
                    continue
                d = self.samples.setdefault(name, {"n": 0, "syn": 0, "udp": 0, "icmp": 0,
                                                   "ports": set(), "conns": set()})
                d["n"] += 1
                d["syn"] += int(bool(s.is_syn))
                d["udp"] += int(s.proto == 17)
                d["icmp"] += int(s.proto == 1)
                if s.proto in (6, 17):
                    d["ports"].add(s.dport)
                d["conns"].add((s.proto, s.sport, s.dport))

    # ------------------------------------------------------------ disponibilidade
    def _availability(self):
        """Lê as novas linhas dos clientes HTTP (t_fim, ok, latência)."""
        ok = tot = 0
        for f in (self.run_dir / "clients").glob("*.csv"):
            pos = self.avail_pos.get(f, 0)
            with open(f, encoding="utf8") as fh:
                fh.seek(pos)
                for line in fh:
                    parts = line.strip().split(",")
                    if len(parts) >= 2 and parts[1] in ("0", "1"):
                        tot += 1
                        ok += int(parts[1])
                self.avail_pos[f] = fh.tell()
        return (ok / tot) if tot else 1.0, tot

    def _attack_active(self, now):
        if self.attack_events is None:
            f = self.run_dir / "events.json"
            if f.exists():
                try:
                    self.attack_events = json.loads(f.read_text(encoding="utf8"))
                except json.JSONDecodeError:
                    return False
            else:
                return False
        e = self.attack_events
        if not e.get("attack_start") or not (e["attack_start"] <= now < e["attack_end"]):
            return False
        if e.get("pulse"):
            on, off = e["pulse"]
            return (now - e["attack_start"]) % (on + off) < on
        return True

    # --------------------------------------------------------------- laço principal
    def _main_loop(self):
        while len(self.dps) < 1 + len(self.lay["edge_dpids"]):
            hub.sleep(0.2)
        next_t = time.time()
        while True:
            next_t += self.dt
            self._request_stats()
            hub.sleep(min(0.3, self.dt / 3))
            try:
                self._step()
            except Exception:                      # noqa: BLE001 — registra e segue
                self.logger.exception("erro no passo de decisão")
            hub.sleep(max(0.0, next_t - time.time()))

    def _out_counters(self, name):
        h = self.hosts[name]
        base = self.flow.get((h["dpid"], P_SRC, h["ip"]), (0, 0, False))
        ov = self.flow.get((h["dpid"], P_OVERLAY, h["ip"]), (0, 0, False))
        rem = self.removed[name]
        return base[0] + ov[0] + rem[0], base[1] + ov[1] + rem[1]

    def _step(self):
        now = time.time()
        self.tick += 1
        samples, self.samples = self.samples, {}
        attack_util_bytes = 0.0
        for n, h in self.hosts.items():
            pk, by = self._out_counters(n)
            ppk, pby = self.prev_out[n]
            bin_ = self.flow.get((h["dpid"], P_DST, h["ip"]), (0, 0, False))[1]
            ss = SourceSample(pkts_out=max(0, pk - ppk), bytes_out=max(0, by - pby),
                              bytes_in=max(0, bin_ - self.prev_in[n]))
            self.prev_out[n], self.prev_in[n] = (pk, by), bin_
            d = samples.get(n)
            if d:
                ss.n_samples, ss.n_syn, ss.n_udp, ss.n_icmp = d["n"], d["syn"], d["udp"], d["icmp"]
                ss.dst_ports, ss.conns = d["ports"], d["conns"]
            self.fx.push(n, ss)
            if h["ip"] in self.attackers:          # estimativa de ataque que segue rumo à vítima
                a = self.action[n]
                mdrop = self.meter_drop[n][1] - self.prev_meter[n]
                if a == Action.ALLOW:
                    attack_util_bytes += ss.bytes_out
                elif a == Action.LIMIT:
                    attack_util_bytes += max(0, ss.bytes_out - mdrop)
                elif a == Action.REDIRECT and ss.pkts_out:
                    attack_util_bytes += min(ss.pkts_out, self.an_pps * self.dt) * ss.bytes_out / ss.pkts_out
            self.prev_meter[n] = self.meter_drop[n][1]

        L = self.lay
        g = GlobalSample()
        for key, attr in (("victim", "victim_bytes"), ("analysis", "analysis_bytes")):
            port = L[key]["port"]
            cur = self.port_tx.get(port, 0)
            setattr(g, attr, max(0, cur - self.prev_port.get(port, cur)))
            self.prev_port[port] = cur
        try:
            g.rtt_ms = float((self.run_dir / "probe_rtt.txt").read_text().strip() or 0)
        except (OSError, ValueError):
            g.rtt_ms = 0.0
        g.packet_in, self.packet_in = self.packet_in, 0
        names = list(self.hosts)
        g.restricted_frac = float(np.mean([self.action[n] != Action.ALLOW for n in names]))
        glob = self.fx.global_(g, len(names))

        # libera bloqueios vencidos (caso o FlowRemoved se atrase)
        for n in names:
            if self.action[n] == Action.BLOCK and now >= self.lock_until[n] + 1.0:
                self.action[n] = Action.ALLOW
                self.changed_at[n] = now
        deciders = [n for n in names if not (self.action[n] == Action.BLOCK and now < self.lock_until[n])]
        obs = {n: self.fx.observation(n, glob, self.action[n], now - self.changed_at[n]) for n in names}
        t0 = time.perf_counter()
        X = np.stack([obs[n] for n in deciders]) if deciders else np.zeros((0, len(glob)))
        acts = self.policy.decide(X) if deciders else []
        q = self.policy.q_values(X) if deciders else None
        if self.eps > 0 and deciders:
            rnd = self.rng.random(len(acts)) < self.eps
            acts = np.where(rnd, self.rng.integers(0, len(Action), len(acts)), acts)
        decide_ms = (time.perf_counter() - t0) * 1000
        prev = dict(self.action)
        for n, a in zip(deciders, acts):
            self.apply(n, int(a))
        apply_ms = (time.perf_counter() - t0) * 1000 - decide_ms

        avail, n_req = self._availability()
        vbps = self.cfg["network"]["victim_bw_mbps"] * 1e6
        rec = {
            "t": now, "tick": self.tick, "scenario": self.runcfg.get("scenario"),
            "policy": self.policy.name, "attack_active": self._attack_active(now),
            "global": [round(float(x), 5) for x in glob],
            "availability": avail, "requests": n_req,
            "attack_util": min(1.0, attack_util_bytes * 8 / self.dt / vbps),
            "decide_ms": decide_ms, "apply_ms": apply_ms,
            "msgs_out": self.msgs_out, "msgs_in": self.msgs_in, "n_rules": len(self.flow),
            "sources": {},
        }
        qi = {n: i for i, n in enumerate(deciders)}
        for n in names:
            h = self.hosts[n]
            rec["sources"][n] = {
                "ip": h["ip"], "obs": [round(float(x), 5) for x in obs[n]],
                "prev": int(prev[n]), "action": int(self.action[n]),
                "decided": n in qi, "label": h["ip"] in self.attackers,
                "q": [round(float(x), 4) for x in q[qi[n]]] if (q is not None and n in qi) else None,
            }
        self.dlog.write(json.dumps(rec) + "\n")
        self.dlog.flush()
        self._write_summary()

    def _write_summary(self):
        mitig = {}
        for n, h in self.hosts.items():
            ov = self.flow.get((h["dpid"], P_OVERLAY, h["ip"]), (0, 0, False))
            blk_now = ov[1] if (self.action[n] == Action.BLOCK and ov[2]) else 0
            mitig[h["ip"]] = {"block_bytes": self.removed_block[n][1] + blk_now,
                              "meter_bytes": self.meter_drop[n][1],
                              "action": ACTION_NAMES[self.action[n]]}
        s = {"ticks": self.tick, "msgs_out": self.msgs_out, "msgs_in": self.msgs_in,
             "mitigated": mitig, "policy": self.policy.name}
        tmp = self.run_dir / "controller_summary.json.tmp"
        tmp.write_text(json.dumps(s), encoding="utf8")
        tmp.replace(self.run_dir / "controller_summary.json")
