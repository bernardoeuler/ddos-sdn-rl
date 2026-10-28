# Implementação: mitigação adaptativa de DDoS em SDN com aprendizado por reforço

Código do artigo em `../main.tex`. Um agente Double Dueling DQN decide, para cada IP de origem e a cada 1 s, entre **permitir, limitar (meter OpenFlow), redirecionar ao nó de análise ou bloquear temporariamente (hard_timeout)**. Ele é comparado aos baselines B0–B4 no Mininet.

## Estrutura

```
configs/default.yaml        todos os parâmetros (rede, agente, recompensa, cenários S0–S7)
configs/sim_calibration.json  parâmetros do simulador ajustados com traços do Mininet
ddosrl/common.py            ações, layout da observação, recompensa (Eq. 2)
ddosrl/features.py          extração de atributos (a mesma no simulador e no controlador)
ddosrl/sim/env.py           simulador de fluidos (Gymnasium) para a fase 1 do treino
ddosrl/agent/ddqn.py        Double DQN dueling (PyTorch) com transições semi-Markovianas
ddosrl/agent/numpy_policy.py  inferência em numpy usada dentro do controlador
ddosrl/policies.py          agente e baselines B0–B4 com a mesma interface
ddosrl/sdn/controller.py    app os-ken: tabelas, contadores, sFlow, decisão, atuação, auditoria
ddosrl/sdn/sflow.py         parser sFlow v5
ddosrl/sdn/traffic.py       servidor HTTP, clientes Poisson, sonda RTT, ataque pulsado
ddosrl/sdn/layout.py        numeração determinística de portas/dpids
scripts/                    treino, calibração, experimentos, campanha, análise
tests/                      testes unitários (pytest) e teste de fumaça do OVS
```

## Ambiente (Linux)

O ambiente precisa de um kernel com os módulos `openvswitch`, `sch_htb`, `sch_netem` e `sch_tbf`; os kernels padrão do Ubuntu e do Debian os incluem. Preparação, como root:

```bash
sudo bash setup.sh
```

O script instala Mininet 2.3, OVS 3.x, hping3 e iperf3, e cria o venv `/opt/ddosrl-venv` com torch (CPU), os-ken, scikit-learn e gymnasium. Os experimentos exigem root.

O os-ken 4.x não traz mais o `osken-manager` nem os apps de exemplo. Por isso há o lançador próprio `python -m ddosrl.sdn.launch_controller`.

## Pipeline

Todos os comandos rodam a partir desta pasta, com `PY=/opt/ddosrl-venv/bin/python`, como root.

| Etapa | Comando |
|---|---|
| Testes | `$PY -m pytest -q tests/` |
| 1. Traços sem defesa (calibração) | `$PY scripts/run_campaign.py --policies B0 --scenarios train --seeds 1 --out results/calibration` |
| 2. Calibrar o simulador | `$PY scripts/calibrate_sim.py` |
| 3. Treinar no simulador | `$PY scripts/train_agent.py --variant full` (e `two_actions`, `label_free`) |
| 4. Baselines (limiares + RF) | `$PY scripts/train_baselines.py` |
| 5. Ajuste fino no Mininet | `$PY scripts/finetune_agent.py --episodes 12` |
| 6. Campanha | `$PY scripts/run_campaign.py --policies B0 B1 B2 B3 B4 DQN --scenarios all --seeds 1 2 ... 10` |
| 7. Ablações | `$PY scripts/run_campaign.py --policies DQN2 DQNLF --scenarios S1H S4 S5 --seeds 1 2 3` |
| 8. Análise | `$PY scripts/analyze.py` → `results/tables/*.tex`, `results/figures/` |

Uma rodada isolada (cerca de 3 min): `$PY scripts/run_experiment.py --policy DQN --scenario S4 --seed 1001`.
A campanha é retomável: ela pula as rodadas que já têm `metrics.json`.

## Decisões de implementação

- **Contadores por origem.** O OpenFlow só conta pacotes e bytes por regra. A tabela 0 das bordas tem uma entrada por IP como origem e outra como destino. A mitigação é uma **sobreposição de prioridade 200** com o mesmo match, e os contadores das sobreposições removidas chegam via `FlowRemoved`.
- **Atributos de cabeçalho via sFlow.** Proporção de SYN, portas distintas e conexões vêm do sFlow do OVS, com amostragem de 1:64 e parser próprio.
- **Entropia das origens pelos contadores de pacotes.** Com sFlow a 1:64, a entropia ficava ruidosa demais. Por isso ela é calculada pelos contadores exatos de pacotes na janela.
- **B2 adaptado.** Mousavi e St-Hilaire usam a entropia dos endereços de destino. Como aqui há um único servidor, o B2 usa a entropia das **origens** rumo à vítima.
- **Nó de análise.** É um roteador Linux com `iptables hashlimit` por origem (100 pps). O SYNPROXY foi descartado porque exige caminho simétrico, e a resposta do servidor volta direto ao cliente.
- **Bloqueio travado.** Durante T_b = 10 s a origem não decide. Na expiração ela volta a "permitir", e a transição do agente usa γ^k (k = T_b).
- **Recompensa com b(a) e h(a) separados.** Com severidade linear, o ótimo local seria sempre uma ação extrema (permitir ou bloquear). Com benefício b(a) e dano h(a) distintos, ações intermediárias passam a ser ótimas quando há incerteza (ver `tests/test_core.py`).
- **Servidor.** Syncookies desligados, `tcp_max_syn_backlog=1024`, `tcp_synack_retries=2`. Assim o SYN flood é efetivo, e a fila semiaberta persiste cerca de 7 s, efeito modelado no simulador.
- **Ablações.** `DQN2` (apenas {permitir, bloquear}) e `DQNLF` (recompensa sem rótulos) usam o modelo do simulador, sem ajuste fino.
