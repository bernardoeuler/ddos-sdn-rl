"""Lançador do controlador os-ken (o os-ken 4.x não inclui o osken-manager).

Uso: python -m ddosrl.sdn.launch_controller [--ofp-tcp-listen-port 6653] [app ...]
Sem apps, carrega ddosrl.sdn.controller.
"""
import sys

from os_ken.lib import hub

hub.patch(thread=False)

from os_ken import cfg                                # noqa: E402
from os_ken.base.app_manager import AppManager       # noqa: E402


def main():
    import logging
    import os
    level = logging.DEBUG if os.environ.get("DDOSRL_DEBUG") else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    argv = sys.argv[1:]
    apps = [a for a in argv if not a.startswith("-") and "." in a and not a[0].isdigit()]
    opts = [a for a in argv if a not in apps]
    cfg.CONF(args=opts, project="os_ken")
    AppManager.run_apps(apps or ["ddosrl.sdn.controller"])


if __name__ == "__main__":
    main()
