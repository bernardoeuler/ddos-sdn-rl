"""Numeração determinística da topologia, compartilhada por Mininet e controlador.

Núcleo s1 (dpid 1): portas 1..E para as bordas, depois vítima, análise e monitor.
Borda s(k+2) (dpid k+2): portas 1..H para os hosts e porta H+1 para o núcleo.
"""
from __future__ import annotations


def mac(n: int) -> str:
    return "00:00:00:00:00:%02x" % n


def layout(cfg: dict) -> dict:
    net = cfg["network"]
    E, H = net["edge_switches"], net["hosts_per_edge"]
    hosts = {}
    for k in range(E):
        for j in range(H):
            idx = k * H + j + 1
            hosts[f"h{idx}"] = {"ip": f"10.0.0.{idx}", "mac": mac(idx), "dpid": k + 2,
                                "port": j + 1, "index": idx}
    specials = {}
    for off, (name, ip) in enumerate([("victim", net["victim_ip"]),
                                      ("analysis", net["analysis_ip"]),
                                      ("monitor", net["monitor_ip"])]):
        last = int(ip.split(".")[-1])
        specials[name] = {"ip": ip, "mac": mac(last), "dpid": 1, "port": E + 1 + off,
                          "name": {"victim": "srv", "analysis": "anl", "monitor": "mon"}[name]}
    return {
        "core_dpid": 1,
        "edge_dpids": [k + 2 for k in range(E)],
        "uplink_port": H + 1,
        "core_port_of_edge": {k + 2: k + 1 for k in range(E)},
        "hosts": hosts,
        **specials,
    }
