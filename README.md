# Mitigação adaptativa de DDoS em SDN com aprendizado por reforço

Projeto 3 da disciplina de Redes. É um agente de **aprendizado por reforço profundo (Double Dueling DQN)** que observa o tráfego de uma rede SDN emulada (Mininet + Open vSwitch + controlador os-ken/Ryu). A cada segundo, ele decide para cada IP de origem uma de quatro ações:

| Ação | Implementação OpenFlow 1.3 |
|---|---|
| permitir | encaminhamento normal |
| limitar a taxa | *meter* com banda *drop* (1 Mbit/s) |
| redirecionar para análise | reescreve o MAC de destino para um nó com `iptables hashlimit` |
| bloquear temporariamente | regra de descarte com `hard_timeout` de 10 s |

O objetivo é reduzir o ataque **sem bloquear usuários legítimos**. O agente é comparado a cinco *baselines*:

- B0: sem defesa;
- B1: limiar fixo;
- B2: entropia;
- B3: Random Forest;
- B4: Random Forest com ações graduadas.

Esses métodos são avaliados em 14 cenários:
- tráfego normal;
- floods SYN, UDP e ICMP em três intensidades;
- ataque pulsado;
- *flash crowd*;
- ataque multivetor;
- rede degradada.

O artigo, no modelo SBC, está em [`main.tex`](main.tex), e o código em [`implementacao/`](implementacao/).

## Resultados preliminares

Uma semente, rodadas de 60 s no Mininet, sem testes estatísticos.

| Cenário | Método | Ataque mitigado | Disponibilidade | Legítimos restringidos | Reação |
|---|---|---|---|---|---|
| SYN flood alto | B1 limiar | 90,6% | 45,3% | 0% | 1,74 s |
| | B3 RF | 94,7% | 42,9% | 0% | 0,77 s |
| | **Agente** | **95,2%** | 45,3% | 25% | 1,21 s |
| UDP flood alto | B1 limiar | 95,0% | 100% | 0% | 0,62 s |
| | B3 RF | 91,6% | 98,5% | 0% | 1,77 s |
| | **Agente** | **95,7%** | 100% | 0% | **0,35 s** |
| Flash crowd | B1 / B3 / Agente | — | 37,6% / 36,0% / 38,0% | 0% | — |

- **Mitigação, reação e custo:** o agente mitiga um pouco mais e reage mais rápido que os baselines. O custo é de 0,3 ms por decisão, contra 35 ms do Random Forest.
- **Disponibilidade:** fica empatada com os baselines.
- **Ações graduadas:** quase não foram usadas (cerca de 6% das decisões). Com floods facilmente separáveis, bloquear rápido já é ótimo.

A discussão completa e as ameaças à validade estão nas Seções 7 e 8 do artigo.

## Estrutura

```
main.tex, referencias.bib, sbc-template.sty, sbc.bst   artigo (modelo SBC)
notas-pesquisa.md                                       fichamento e decisões técnicas
implementacao/
  setup_wsl.sh            instala Mininet, OVS, hping3, iperf3 e o venv Python
  configs/default.yaml    todos os parâmetros (rede, agente, recompensa, cenários)
  configs/sim_calibration.json   parâmetros do simulador calibrados no Mininet
  ddosrl/                 pacote: atributos, simulador, agente, baselines, controlador SDN
  scripts/                treino, calibração, experimentos, campanha, análise
  tests/                  testes unitários (pytest) e teste de fumaça do OVS
  models/                 modelos já treinados (agente e baselines)
  results/                resultados das rodadas executadas
```

Detalhes de implementação estão em [`implementacao/README.md`](implementacao/README.md).

## Requisitos

- **Linux com root.** No Windows, use o **WSL2 com Ubuntu**; o kernel 6.x do WSL já inclui os módulos `openvswitch`, `sch_htb` e `sch_netem`. Também serve uma VM ou máquina Ubuntu.
- Cerca de 4 GB livres (o venv com torch CPU ocupa a maior parte) e de preferência 4 ou mais núcleos.
- Acesso à internet para instalar os pacotes.

## 1. Instalação

No Windows, abra o terminal e entre no Ubuntu como root:

```bash
wsl -d Ubuntu -u root
```

Dentro do Ubuntu, a partir da pasta do projeto (no WSL, `C:\` fica em `/mnt/c/`):

```bash
cd implementacao
bash setup_wsl.sh
```

O script instala `mininet`, `openvswitch-switch`, `hping3` e `iperf3`, e carrega os módulos do kernel. Também cria o venv `/opt/ddosrl-venv`, com torch CPU, os-ken, scikit-learn e gymnasium. Ao final ele imprime as versões e `SETUP_OK`.

Nos comandos seguintes, considere:

```bash
PY=/opt/ddosrl-venv/bin/python
```

Depois de reiniciar o WSL, recarregue os módulos e o OVS antes de rodar experimentos:

```bash
modprobe -a openvswitch sch_htb sch_netem sch_tbf && service openvswitch-switch start
```

## 2. Testes

```bash
$PY -m pytest -q tests/
bash tests/smoke_osken.sh
```

- O `pytest` roda 12 testes, em cerca de 5 s. Eles cobrem parser sFlow, atributos, recompensa, simulador, paridade torch/numpy, acumulação semi-Markoviana e baselines.
- O `smoke_osken.sh` testa Mininet, OVS e os *meters* OpenFlow.

## 3. Reproduzir os resultados preliminares (≈ 30–40 min)

Os modelos usados no artigo já estão em `implementacao/models/`. Para apenas rodar a avaliação, vá direto ao passo 3.5.

```bash
# 3.1 Traços sem defesa para calibração (4 rodadas de 180 s, ≈ 13 min)
$PY scripts/run_campaign.py --policies B0 --scenarios S0 S1L S1M S1H --seeds 1 --out results/calibration

# 3.2 Calibrar o simulador (≈ 2 min) -> configs/sim_calibration.json
$PY scripts/calibrate_sim.py --seeds 2
#     O artigo descarta os parâmetros HTTP estimados (viés de amostragem);
#     remova as chaves http_* do JSON para obter o mesmo arquivo usado:
$PY -c "import json;p='configs/sim_calibration.json';d=json.load(open(p));[d.pop(k,None) for k in ('http_resp_bytes','http_out_bytes','http_out_pkts')];json.dump(d,open(p,'w'),indent=2)"

# 3.3 Treinar os agentes no simulador (≈ 6–9 min cada; podem rodar em paralelo)
OMP_NUM_THREADS=6 $PY scripts/train_agent.py --variant full        --seed 1 --updates 160000
OMP_NUM_THREADS=6 $PY scripts/train_agent.py --variant two_actions --seed 1 --updates 160000
OMP_NUM_THREADS=6 $PY scripts/train_agent.py --variant label_free  --seed 1 --updates 160000

# 3.4 Baselines: limiares de B1/B2 e Random Forest de B3/B4 (≈ 1 min)
$PY scripts/train_baselines.py --episodes 4

# 3.5 Avaliação no simulador e uso das ações (≈ 2 min)
$PY scripts/eval_sim.py --policies B0 B1 B2 B3 B4 DQN DQN2 --seeds 3
$PY scripts/action_usage.py --policies DQN DQN2 B4

# 3.6 Rodadas curtas no Mininet: B1, B3 e agente em S1H, S2H e S5 (≈ 11 min)
$PY scripts/quick_mininet.py --policies B1 B3 DQN
```

Saídas:

| Arquivo | Conteúdo |
|---|---|
| `results/quick_summary.csv` | tabela de resultados do Mininet |
| `results/sim_eval.csv` | avaliação no simulador |
| `results/training/curve_*.csv` | curvas de aprendizado |
| `results/quick_runs/<cenário>/<método>/seed1001/` | dados brutos de cada rodada |

Os dados brutos incluem as decisões do controlador a cada segundo, os logs dos clientes HTTP e as métricas.

Os números podem variar um pouco entre máquinas. O Mininet divide a CPU com o resto do sistema, então evite carga pesada durante as rodadas.

## 4. Experimentos completos (protocolo do artigo)

Uma rodada isolada (180 s):

```bash
$PY scripts/run_experiment.py --policy DQN --scenario S4 --seed 1001
```

Políticas disponíveis:
- `B0` a `B4`: baselines;
- `DQN`: agente;
- `DQN2`: agente só com {permitir, bloquear};
- `DQNLF`: agente com recompensa sem rótulos.

Os cenários estão em `configs/default.yaml` (`S0`, `S1L` … `S3H`, `S4` … `S7`).

Campanha completa, ablações e análise:

```bash
bash scripts/pipeline_campaign.sh 1 2 3
```

Com 3 sementes, a campanha leva cerca de 14,5 h. Ela é **retomável**: se for interrompida, basta rodar de novo, e as rodadas que já têm `metrics.json` são puladas. A análise gera:

- `results/tables/*.tex`: tabelas prontas para o artigo, com média e IC 95% por bootstrap;
- `results/tables/wilcoxon_holm.csv`: testes pareados;
- `results/figures/`: curva de aprendizado e séries temporais.

Opcional: ajuste fino do agente com episódios no Mininet (cerca de 40 min):

```bash
$PY scripts/finetune_agent.py --episodes 12
```

> Atenção: o ajuste fino ainda usa a recompensa por rótulo b(a)/h(a), e não a recompensa por efeito medido usada no simulador; não foi usado nos resultados do artigo.

## 5. Problemas comuns

| Sintoma | Solução |
|---|---|
| `Unable to contact the remote controller` no início | Normal: o controlador leva cerca de 1 s para subir e o OVS reconecta sozinho. |
| `sch_htb: quantum of class ... is big` | Aviso inofensivo do TCLink. |
| `controlador não ficou pronto` | Veja `controller.log` na pasta da rodada (`/root/ddosrl_runs/...`) e rode `mn -c`. |
| `modprobe: module openvswitch not found` | O kernel não traz o módulo: use a VM Ubuntu ou recompile o kernel do WSL. |
| Rodada interrompida deixou lixo | `mn -c; pkill -f launch_controller; pkill hping3` |

O os-ken 4.x não inclui `osken-manager`. Por isso o controlador sobe pelo lançador próprio, `python -m ddosrl.sdn.launch_controller`, e os scripts já cuidam disso.

## Compilar o artigo

Envie `main.tex`, `referencias.bib`, `sbc-template.sty` e `sbc.bst` ao Overleaf, ou compile com `pdflatex main && bibtex main && pdflatex main && pdflatex main`.
