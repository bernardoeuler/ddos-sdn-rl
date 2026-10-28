"""Geradores de tráfego executados dentro dos hosts do Mininet.

  python -m ddosrl.sdn.traffic http  --url URL --rate R --out F --until T [--schedule JSON]
  python -m ddosrl.sdn.traffic probe --target IP --out F --until T
  python -m ddosrl.sdn.traffic pulse --on S --off S --until T -- hping3 args...

O cliente HTTP grava uma linha por requisição: t_fim,ok,latencia_ms.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.request


def http_client(a):
    sched = json.loads(a.schedule) if a.schedule else []   # [[t0, t1, fator], ...]
    rng = random.Random(a.seed)
    lock = threading.Lock()
    sem = threading.Semaphore(16)
    out = open(a.out, "a", buffering=1, encoding="utf8")

    def one():
        t0 = time.time()
        ok = 0
        try:
            with urllib.request.urlopen(a.url, timeout=a.timeout) as r:
                r.read()
                ok = int(r.status == 200)
        except Exception:                                  # noqa: BLE001
            ok = 0
        t1 = time.time()
        if t1 - t0 > a.timeout:
            ok = 0
        with lock:
            out.write(f"{t1:.3f},{ok},{(t1 - t0) * 1000:.1f}\n")
        sem.release()

    while time.time() < a.until:
        now = time.time()
        f = next((x[2] for x in sched if x[0] <= now < x[1]), 1.0)
        time.sleep(rng.expovariate(a.rate * f))
        if not sem.acquire(blocking=False):                # muitas pendentes: conta falha
            with lock:
                out.write(f"{time.time():.3f},0,{a.timeout * 1000:.1f}\n")
            continue
        threading.Thread(target=one, daemon=True).start()


def probe(a):
    """Sonda ICMP: grava o RTT médio da última janela em um arquivo."""
    while time.time() < a.until:
        r = subprocess.run(["ping", "-c", "1", "-W", "1", a.target], capture_output=True, text=True)
        m = re.search(r"time=([\d.]+)", r.stdout)
        rtt = float(m.group(1)) if m else 1000.0
        with open(a.out + ".tmp", "w") as f:
            f.write(f"{rtt}\n")
        import os
        os.replace(a.out + ".tmp", a.out)
        with open(a.out + ".log", "a") as f:
            f.write(f"{time.time():.3f},{rtt}\n")
        time.sleep(max(0.0, 0.5 - (rtt / 1000 if m else 1.0)))


def server(a):
    """Servidor HTTP multithread com fila de aceitação de 128 conexões."""
    import functools
    import http.server

    class Srv(http.server.ThreadingHTTPServer):
        request_queue_size = 128
        daemon_threads = True

    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

    h = functools.partial(Quiet, directory=a.dir)
    Srv(("0.0.0.0", a.port), h).serve_forever()


def pulse(a, cmd):
    """Liga e desliga um comando (hping3) em ciclos on/off."""
    while time.time() < a.until:
        p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(min(a.on, max(0, a.until - time.time())))
        p.send_signal(signal.SIGINT)
        p.wait()
        time.sleep(min(a.off, max(0, a.until - time.time())))


def main():
    argv = sys.argv[1:]
    cmd = []
    if "--" in argv:
        i = argv.index("--")
        argv, cmd = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["http", "probe", "pulse", "server"])
    ap.add_argument("--dir", default=".")
    ap.add_argument("--port", type=int, default=80)
    ap.add_argument("--url")
    ap.add_argument("--rate", type=float, default=2.0)
    ap.add_argument("--timeout", type=float, default=2.0)
    ap.add_argument("--schedule", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target")
    ap.add_argument("--out")
    ap.add_argument("--until", type=float, default=float("inf"))
    ap.add_argument("--on", type=float, default=5)
    ap.add_argument("--off", type=float, default=10)
    a = ap.parse_args(argv)
    {"http": lambda: http_client(a), "probe": lambda: probe(a),
     "pulse": lambda: pulse(a, cmd), "server": lambda: server(a)}[a.mode]()


if __name__ == "__main__":
    main()
