#!/usr/bin/env bash
# Pipeline completo após as rodadas de calibração (executar como root).
#   bash scripts/pipeline.sh [SEMENTES...]   (padrão: 1 2 3)
set -u
cd "$(dirname "$0")/.."
PY=/opt/ddosrl-venv/bin/python
SEEDS=${*:-1 2 3}
LOG=results/pipeline.log
mkdir -p results/training
step() { echo "$(date +%H:%M:%S) == $*" | tee -a $LOG; }

while pgrep -f "run_campaign.py --policies B0" >/dev/null; do sleep 20; done

step "1. calibrar simulador"
$PY scripts/calibrate_sim.py >> $LOG 2>&1 || { step "FALHOU calibração"; exit 1; }

step "2. treinar agentes no simulador (3 variantes em paralelo)"
for v in full two_actions label_free; do
  OMP_NUM_THREADS=3 $PY scripts/train_agent.py --variant $v --seed 1 > results/training/train_$v.log 2>&1 &
done
wait
tail -n 2 results/training/train_*.log >> $LOG

step "3. baselines a partir dos traços do Mininet (B0 em S0-S3)"
$PY scripts/train_baselines.py --traces results/calibration/*/B0/seed*/decisions.jsonl >> $LOG 2>&1 \
  || { step "FALHOU baselines"; exit 1; }

step "4. avaliação no simulador (sanidade)"
$PY scripts/eval_sim.py --policies B0 B1 B2 B3 B4 DQN --seeds 5 >> $LOG 2>&1

step "5. ajuste fino no Mininet"
$PY scripts/finetune_agent.py --episodes 12 >> $LOG 2>&1 || { step "FALHOU ajuste fino"; exit 1; }

step "6. campanha principal (sementes: $SEEDS)"
$PY scripts/run_campaign.py --policies B0 B1 B2 B3 B4 DQN --scenarios all --seeds $SEEDS 2>&1 \
  | grep -v -E "quantum|Unable to contact" >> $LOG

step "7. ablações"
$PY scripts/run_campaign.py --policies DQN2 DQNLF --scenarios S1H S4 S5 --seeds $SEEDS 2>&1 \
  | grep -v -E "quantum|Unable to contact" >> $LOG

step "8. análise"
$PY scripts/analyze.py >> $LOG 2>&1
step "PIPELINE_FIM"
