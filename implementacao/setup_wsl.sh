#!/usr/bin/env bash
# Prepara o Ubuntu (WSL2) para os experimentos. Executar como root:
#   wsl -d Ubuntu -u root -- bash setup_wsl.sh
set -euo pipefail

VENV=/opt/ddosrl-venv
export DEBIAN_FRONTEND=noninteractive

if ! command -v mn >/dev/null || ! command -v ovs-vsctl >/dev/null; then
  apt-get update -q
  apt-get install -y -q mininet openvswitch-switch hping3 iperf3 python3-venv \
    python3-pip python3-dev tcpdump net-tools curl iptables
fi

# O kernel 6.x do WSL2 traz openvswitch e as qdiscs como módulos.
modprobe -a openvswitch sch_htb sch_netem sch_tbf
service openvswitch-switch start || /usr/share/openvswitch/scripts/ovs-ctl start

# venv com acesso ao pacote mininet do sistema (python3-mininet)
if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv --system-site-packages "$VENV"
fi
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q --index-url https://download.pytorch.org/whl/cpu torch
"$VENV/bin/pip" install -q -r "$(dirname "$0")/requirements.txt"

echo "== versões =="
mn --version
ovs-vsctl --version | head -1
hping3 --version 2>&1 | head -1 || true
"$VENV/bin/python" - <<'EOF'
import sys, numpy, torch, sklearn, gymnasium, os_ken, mininet
print("python", sys.version.split()[0])
for m in (numpy, torch, sklearn, gymnasium):
    print(m.__name__, m.__version__)
print("os_ken", getattr(os_ken, "__version__", "?"), "| mininet", mininet.__file__)
EOF
echo "SETUP_OK"
