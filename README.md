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

Primeiras 40 de 270 rodadas da campanha (180 s, semente 1001, cenários S0–S2), média de S1L–S2M. A campanha completa continua em execução.

| Método | Ataque mitigado | Legítimos restringidos | Disponibilidade | Reação |
|---|---|---|---|---|
| B0 sem defesa | 0% | 0% | 53,5% | — |
| B1 limiar | 94,1% | 0% | 86,4% | 1,7 s |
| B2 entropia | 94,0% | 100% | 23,6% | 1,7 s |
| B3 RF | 94,5% | 4,2% | 86,0% | 1,1 s |
| B4 RF graduado | 95,0% | 0% | 84,4% | 1,2 s |
| **Agente** | 55,5% | 52,5% | 54,5% | **0,3 s** |

- **Resultado negativo até aqui:** o agente reage mais rápido e custa 0,3 ms por decisão (contra 28 ms do RF), mas sob SYN flood restringe 62–88% das origens legítimas. No UDP ele é competitivo (100% de disponibilidade).
- **Causa provável:** lacuna entre simulador e emulação no efeito do SYN sobre os clientes, e um ajuste fino curto e com exploração.
- **Ações graduadas:** quase não foram usadas (cerca de 6% das decisões). Com floods facilmente separáveis, bloquear rápido já é ótimo.

A discussão completa e as ameaças à validade estão nas Seções 7 e 8 do artigo.

## Estrutura

```
main.tex, referencias.bib, sbc-template.sty, sbc.bst   artigo (modelo SBC)
notas-pesquisa.md                                       fichamento e decisões técnicas
implementacao/
  setup.sh                instala Mininet, OVS, hping3, iperf3 e o venv Python
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

- **Linux com acesso root**, de preferência Ubuntu 22.04+ ou Debian 12+. O `setup.sh` usa `apt`; em outras distribuições, instale os pacotes equivalentes manualmente.
- **Kernel com os módulos** `openvswitch`, `sch_htb`, `sch_netem` e `sch_tbf`. Os kernels padrão do Ubuntu e do Debian já os incluem. Confira com `modinfo openvswitch`.
- **Python 3.10 ou superior.**
- **Recursos:** cerca de 4 GB livres (o venv com torch CPU ocupa a maior parte) e de preferência 4 ou mais núcleos.
- **Acesso à internet** para instalar os pacotes.

Uma VM com Ubuntu também serve.

## 1. Instalação

Clone o repositório e rode o script de instalação como root:

```bash
git clone git@github.com:bernardoeuler/ddos-sdn-rl.git
cd ddos-sdn-rl/implementacao
sudo bash setup.sh
```

O script instala `mininet`, `openvswitch-switch`, `hping3` e `iperf3`, e carrega os módulos do kernel. Também cria o venv `/opt/ddosrl-venv`, com torch CPU, os-ken, scikit-learn e gymnasium. Ao final ele imprime as versões e `SETUP_OK`.

O Mininet e o OVS exigem root. Por isso, todos os comandos a seguir rodam como root, a partir da pasta `implementacao/`:

```bash
sudo -i
cd /caminho/para/ddos-sdn-rl/implementacao
PY=/opt/ddosrl-venv/bin/python
```

Depois de reiniciar a máquina, se o OVS não subir sozinho:

```bash
modprobe -a openvswitch sch_htb sch_netem sch_tbf && systemctl start openvswitch-switch
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
| `modprobe: module openvswitch not found` | O kernel não traz o módulo: instale `linux-modules-extra-$(uname -r)` (Ubuntu) ou use um kernel que inclua `CONFIG_OPENVSWITCH`. |
| Ambiente WSL2 | Funciona como foi usado no artigo: o kernel 6.x do WSL já traz os módulos; rode como root (`wsl -u root`). |
| Rodada interrompida deixou lixo | `mn -c; pkill -f launch_controller; pkill hping3` |

O os-ken 4.x não inclui `osken-manager`. Por isso o controlador sobe pelo lançador próprio, `python -m ddosrl.sdn.launch_controller`, e os scripts já cuidam disso.

## Compilar o artigo

Envie `main.tex`, `referencias.bib`, `sbc-template.sty` e `sbc.bst` ao Overleaf, ou compile com `pdflatex main && bibtex main && pdflatex main && pdflatex main`.
