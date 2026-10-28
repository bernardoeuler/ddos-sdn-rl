#!/usr/bin/env bash
# Teste de fumaça: os-ken + OVS (OpenFlow 1.3 e meters) + Mininet com 2 hosts.
set -u
V=/opt/ddosrl-venv/bin
mn -c >/dev/null 2>&1
cd "$(dirname "$0")/.."
$V/python -m ddosrl.sdn.launch_controller --ofp-tcp-listen-port 6653 os_ken.app.simple_switch_13 >/tmp/smoke_osken.log 2>&1 &
CP=$!
sleep 3
$V/python - <<'EOF'
from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch
from mininet.topo import SingleSwitchTopo
from mininet.log import setLogLevel
import subprocess
setLogLevel("warning")
net = Mininet(topo=SingleSwitchTopo(2), controller=None,
              switch=lambda n, **k: OVSSwitch(n, protocols="OpenFlow13", **k))
net.addController("c0", controller=RemoteController, ip="127.0.0.1", port=6653)
net.start()
print("ping loss %:", net.pingAll(timeout=2))
r = subprocess.run(["ovs-ofctl", "-O", "OpenFlow13", "add-meter", "s1",
                    "meter=1,kbps,bands=type=drop,rate=1000"], capture_output=True, text=True)
print("add-meter rc", r.returncode, r.stderr.strip())
print(subprocess.run(["ovs-ofctl", "-O", "OpenFlow13", "dump-meters", "s1"],
                     capture_output=True, text=True).stdout.strip())
net.stop()
EOF
kill $CP
echo "--- log controlador"; tail -5 /tmp/smoke_osken.log
