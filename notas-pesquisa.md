# Notas de pesquisa — Projeto 3: Mitigação de DDoS em SDN com RL

Fichamento, quadro comparativo e decisões técnicas que sustentam o `main.tex` e servem de guia para a fase de implementação. Pesquisa feita em 28/10/2026.

## 1. Solução escolhida (resumo)

- **Agente:** Double DQN com arquitetura dueling. É uma única política compartilhada, aplicada **a cada origem** (IP de origem) a cada Δt = 1 s.
- **Ações graduadas (4):** permitir · limitar (OpenFlow meter com banda drop) · redirecionar para um nó de análise · bloquear temporariamente (drop com `hard_timeout`).
- **Recompensa:** bônus por restringir atacante; penalidade **maior** por restringir legítimo (w2 > w1); termo global de disponibilidade; penalidade pelo tráfego de ataque no enlace da vítima; custo por troca de regra (evita oscilação). Os rótulos entram só no treino. Uma ablação sem rótulos usa apenas os termos globais.
- **Treino:** em duas fases. Primeiro um simulador Gymnasium (modelo de fluidos calibrado no Mininet, com domain randomization); depois ajuste fino no Mininet. **A avaliação é feita só no Mininet.**
- **Baselines:** B0 sem defesa · B1 limiar fixo de pps · B2 entropia (Mousavi 2015) · B3 Random Forest + bloqueio · B4 RF com faixas de probabilidade mapeadas para as 4 ações (isola o efeito do RL).
- **Lacuna explorada:** a literatura de RL para DDoS em SDN aprende quase sempre **uma única dimensão** de resposta (grau de throttling) ou uma resposta binária. Poucos trabalhos combinam ações de naturezas diferentes, cenários feitos para gerar falso positivo (flash crowd) e medição da sobrecarga no controlador.

## 2. Fichamento dos trabalhos relacionados

| Trabalho | O que faz | Relevância para nós |
|---|---|---|
| Malialis & Kudenko, AAAI 2013; EAAI 2015 | RL multiagente: roteadores a montante aprendem a limitar o tráfego à vítima. Difference rewards para escalar (>100 agentes). | Origem do "router throttling" com RL. Simulação, não é SDN. |
| Liu et al., CAMAD 2018 (DOI 10.1109/CAMAD.2018.8514971) | DRL em SDN; aprende a limitar inundações SYN/UDP/ICMP e supera o router throttling em 5 cenários. | Referência base de DRL+SDN. Ação = throttling. |
| Feng, Li, Nguyen, IWQoS 2020 | RL contra DDoS de camada de aplicação com recompensa multiobjetivo. É agressivo quando o servidor está sobrecarregado (mitiga 98,73% das mensagens maliciosas) e conservador nos demais casos, com dano colateral mínimo. | Ideia de resposta proporcional mais próxima da nossa, mas fora de SDN. |
| Yungaicela-Naula et al., JNCA 2022 (vol. 205, 103444) | IDS com DL + IPS com DRL contra slow-rate; Mininet + ONOS + Apache; detecção de 98% com amostragem de 30% dos fluxos. | Mesmo tipo de ambiente (Mininet). Foco em baixa taxa. |
| Yungaicela-Naula et al., FGCS 2023 (vol. 149, 637–649) | Extensão com SDN/NFV e moving target defense. | Ação "redirecionar" com NFV. |
| Chen et al., DeepThrottle, IPCCC 2022 | DRL para router throttling em SDN; generaliza para cargas não vistas. | Generalização (QP2). |
| Shen et al., ICC 2023 | DRL em SDN-IoT; punição adaptativa na recompensa; alocação de banda; +22% de tráfego normal e convergência ~200 episódios mais rápida. | Desenho de recompensa. |
| Khozam et al., NetSoft 2024 | DRL contra ICMP/SYN/UDP preservando QoS em SDN para 5G. | Trabalho mais próximo (1). |
| Khozam et al., QoSentry, JNSM 2025 (33:97) | DDQN; contramedidas de throttling e redirecionamento; recompensa baseada em latência/jitter. | **Trabalho mais próximo.** Diferenciação: 4 ações com bloqueio reversível, penalidade explícita de falso positivo, flash crowd, overhead do controlador, stack Ryu/OVS aberta. Ler o texto completo (Springer, acesso pago/institucional) para confirmar o action space exato. |
| Finistrella et al., MininetGym, SoftwareX 2025 | Ambiente Gym sobre Mininet (DQN, PPO, A2C); usa OpenDaylight. Repo: github.com/dipi-unimore/mininet-gym | Pode ser reutilizado ou servir de referência para a integração Gym↔Mininet. |
| Bahashwan et al., Sensors 2023; Tebbaa et al., 2025 | Revisões sistemáticas (ML/DL para DDoS em SDN). | Justificam que a maioria dos trabalhos faz só detecção. |
| Silva et al., SBSeg 2024 | 5 algoritmos na CIC-DDoS2019; RF com 99,8% (binário). | Baseline RF; referência nacional. |
| Dias et al., SBSeg Estendido 2023 | Detecção em SDN com microsserviços; RF com F1 de 98,65%; não sobrecarrega o controlador. | Overhead do controlador; referência nacional. |
| Albuquerque Jr. & Fonseca, SBSeg 2023 | Slow-TCAM: consolidação de regras contra esgotamento da TCAM. | Custo de regras: justifica o termo c(a) na recompensa. |
| Braga et al., LCN 2010; Mousavi & St-Hilaire, ICNC 2015 | SOM sobre estatísticas de fluxo; entropia de IP de destino no controlador. | Atributos clássicos; baseline B2. |

Itens marcados como "n.i." no quadro do artigo não foram confirmados porque só tivemos acesso aos resumos. **Antes da submissão**, confirmar no texto completo (acesso pela biblioteca/CAPES): as ações exatas de Liu 2018, Yungaicela 2022 e QoSentry, e se esses trabalhos medem a sobrecarga do controlador.

## 3. Fundamentação: pontos-chave

- **OpenFlow 1.3:** flow stats e port stats; `idle_timeout`/`hard_timeout`; **meters** com banda `drop` (rate limit), anexados ao fluxo com a instrução `meter:<id>`; set-output para redirecionar.
- **Tipos de DDoS:** SYN, UDP e ICMP flood (volumétricos); low-rate/pulsado (shrew, Kuzmanovic 2003); ataques ao plano de controle (flood de packet-in, esgotamento da TCAM).
- **RL:** MDP, Q-learning, DQN (replay + target), Double DQN (reduz a superestimação), Dueling, PPO.
- **Bases:** CIC-DDoS2019 (ICCST 2019) e InSDN (IEEE Access 2020, CC BY-NC 4.0). São usadas só para validar atributos e o baseline RF; o tráfego dos experimentos é gerado no Mininet.

## 4. Decisões técnicas para a implementação

| Item | Decisão / achado | Fonte |
|---|---|---|
| Controlador | **Ryu 4.34** (PyPI, de 2020; repo faucetsdn/ryu sem commits desde abr/2024) tem problemas conhecidos com eventlet novo e Python ≥ 3.10. **Preferir `os-ken` 4.x** (fork mantido pela OpenStack, `requires_python >= 3.10`, API quase idêntica: `from os_ken...` no lugar de `from ryu...`). Plano B: Ryu em Python 3.9 via pyenv. | PyPI / GitHub API (consultados em 28/10/2026) |
| Agente no controlador | SB3 2.9 e Gymnasium 1.4 exigem Python ≥ 3.10, o que é compatível com os-ken. Se usar Ryu em Python 3.9, rodar o agente em **processo separado** (REST/ZeroMQ). | PyPI |
| SB3 e Double/Dueling | O DQN do SB3 não é Double nem Dueling. Implementar o Double Dueling DQN em PyTorch (~150 linhas) e usar o DQN do SB3 como referência/sanidade. O PPO do SB3 fica como variante. | — |
| WSL2 + OVS | **O kernel padrão do WSL2 não traz o módulo `openvswitch`.** Opções: (a) recompilar o WSL2-Linux-Kernel com `CONFIG_OPENVSWITCH=m` mais as qdiscs `sch_htb`, `sch_netem`, `sch_tbf` (usadas pelo TCLink) e depois `wsl --shutdown`; (b) usar o OVS em userspace (`datapath_type=netdev`; no Mininet, `OVSSwitch(datapath='user')`), que é mais lento mas dispensa o módulo; (c) VM Ubuntu como último recurso. Após cada boot: `sudo modprobe -a openvswitch sch_htb sch_netem` e `sudo service openvswitch-switch start`. | Guias da comunidade (hackmd.io/@jxzhe/rkjEdtN13) + INSTALL do Mininet |
| Meters no OVS | Suportados no datapath userspace e no kernel (versões recentes do OVS e do kernel). Validar com `ovs-ofctl -O OpenFlow13 add-meter br0 meter=1,kbps,bands=type=drop,rate=1000` antes de integrar. | Documentação do OVS / testes |
| Tráfego | Ataque: Scapy 2.8 + `hping3` (`--flood`, `-S`, `--udp`, `-1`). Legítimo: clientes HTTP com chegadas de Poisson (script Python ou `wrk`), `iperf3` de fundo e `ping` como sonda. | — |
| Topologia | Árvore de 2 níveis: 1 switch de núcleo, 3 de borda e 12 hosts, mais a vítima (HTTP), o nó de análise e um host de monitoramento (sondas); gargalo de 10 Mbit/s no enlace da vítima; encaminhamento proativo. | — |
| Coleta de atributos | Contadores OpenFlow dão só pacotes e bytes por regra. A **tabela 0** das bordas tem 1 entrada por IP como origem e 1 como destino: elas contam o tráfego e recebem a mitigação, com `goto_table:1` para o encaminhamento. Proporção de SYN, portas distintas e entropia vêm de **sFlow** do OVS (`ovs-vsctl -- --id=@s create sflow ...`, amostragem de ~1:64), com um coletor em Python no controlador. | Revisão 2 |
| Nó de análise | Host Linux com **iptables SYNPROXY**. O redirecionamento reescreve o MAC de destino (set_field eth_dst) e usa output para a porta do nó. | Revisão 2 |
| Métrica TAB | Contar só o que as **regras de mitigação** descartam (contadores de meter, drop e do nó de análise). Descarte por congestionamento no gargalo não conta. | Revisão 2 |
| Cenários | S0–S3 servem para treino e calibração (pesos, limiares B1/B3/B4). S4 pulsado, S5 flash crowd, S6 multivetor e S7 rede degradada (netem com 20 ms e 1% de perda) ficam **só para teste**. A campanha tem 6 métodos × 14 configurações × 10 repetições × 3 min ≈ 42 h. | Revisão 2 |
| Reprodutibilidade | Arquivo `config.yaml` com versões, sementes e pesos; 10 repetições por par (método, cenário); logs brutos em `results/raw/`. | — |

## 5. Próximos passos (fase de implementação)

1. Preparar o WSL2: kernel com OVS (ou datapath userspace), Mininet 2.3, OVS, os-ken, Python 3.10+.
2. Montar a topologia e o controlador com o monitor (stats a cada 1 s) e o atuador (drop, meter, redirect).
3. Escrever os scripts de tráfego (cenários S0–S6) e a coleta automática de métricas.
4. Implementar os baselines B1–B4 e o dataset rotulado.
5. Implementar o simulador Gymnasium, calibrá-lo e treinar o DDQN; depois o ajuste fino no Mininet.
6. Rodar os experimentos (10×) e preencher as tabelas das seções 7 e 8 do `main.tex` (marcadas com `[a preencher]`).

Observação: as datas do cronograma no PDF do professor (5 a 21/out) já passaram. Vale confirmar com o professor os novos prazos.
