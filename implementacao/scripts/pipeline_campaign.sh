#!/usr/bin/env bash
# Campanha principal + ablações + análise (executar como root no WSL).
set -u
cd "$(dirname "$0")/.."
PY=/opt/ddosrl-venv/bin/python
SEEDS=${*:-1 2 3}
LOG=results/pipeline.log
step() { echo "$(date +%H:%M:%S) == $*" | tee -a $LOG; }

if [ ! -f models/ddqn_label_free_s1.npz ]; then
  step "treinar variante sem rótulos"
  OMP_NUM_THREADS=8 $PY scripts/train_agent.py --variant label_free --seed 1 --updates 160000 \
    > results/training/train_label_free_v2.log 2>&1
fi
step "campanha principal (sementes: $SEEDS)"
$PY scripts/run_campaign.py --policies B0 B1 B2 B3 B4 DQN --scenarios all --seeds $SEEDS 2>&1 \
  | grep -v -E "quantum|Unable to contact" >> $LOG
step "ablações"
$PY scripts/run_campaign.py --policies DQN2 DQNLF --scenarios S1H S4 S5 --seeds $SEEDS 2>&1 \
  | grep -v -E "quantum|Unable to contact" >> $LOG
step "análise"
$PY scripts/analyze.py >> $LOG 2>&1
step "PIPELINE_FIM"
