#!/usr/bin/env bash
# Pipeline final: calibração completa -> treino -> baselines -> ajuste fino ->
# campanha -> ablações -> análise -> explicação. Executar como root.
#   bash scripts/pipeline_final.sh [SEMENTES...]   (padrão: 1 2 3)
set -u
cd "$(dirname "$0")/.."
PY=/opt/ddosrl-venv/bin/python
SEEDS=${*:-1 2 3}
LOG=results/pipeline_final.log
mkdir -p results/training
step() { echo "$(date +%H:%M:%S) == $*" | tee -a $LOG; }

while pgrep -f "run_campaign.py --policies B0" >/dev/null; do sleep 15; done

step "1. calibrar simulador com S0-S3"
rm -f configs/sim_calibration.json
$PY scripts/calibrate_sim.py --seeds 2 >> $LOG 2>&1 || { step "FALHOU calibração"; exit 1; }
# parâmetros HTTP estimados por amostragem são enviesados; mantém os valores físicos
$PY - <<'EOF' >> $LOG
import json
p = "configs/sim_calibration.json"
d = json.load(open(p))
for k in ("http_resp_bytes", "http_out_bytes", "http_out_pkts"):
    d.pop(k, None)
json.dump(d, open(p, "w"), indent=2)
print("calibração final:", d)
EOF

step "2. treinar agentes no simulador"
for v in full two_actions label_free; do
  OMP_NUM_THREADS=4 $PY scripts/train_agent.py --variant $v --seed 1 --updates 160000 \
    > results/training/train_$v.log 2>&1 &
done
wait
tail -n 1 results/training/train_*.log >> $LOG

step "3. baselines (mesmas condições ambíguas)"
$PY scripts/train_baselines.py --episodes 4 >> $LOG 2>&1 || { step "FALHOU baselines"; exit 1; }
$PY scripts/eval_sim.py --policies B0 B1 B2 B3 B4 DQN DQN2 --seeds 3 >> $LOG 2>&1
$PY scripts/action_usage.py --policies DQN DQN2 B4 >> $LOG 2>&1

step "4. ajuste fino no Mininet"
$PY scripts/finetune_agent.py --episodes 12 2>&1 | grep -v -E "quantum|Unable to contact" >> $LOG \
  || step "ajuste fino falhou; segue com o modelo do simulador"

step "5. campanha principal (sementes: $SEEDS)"
$PY scripts/run_campaign.py --policies B0 B1 B2 B3 B4 DQN --scenarios all --seeds $SEEDS 2>&1 \
  | grep -v -E "quantum|Unable to contact" >> $LOG

step "6. ablações"
$PY scripts/run_campaign.py --policies DQN2 DQNLF --scenarios S1H S4 S5 S8 --seeds $SEEDS 2>&1 \
  | grep -v -E "quantum|Unable to contact" >> $LOG

step "7. análise e explicação"
$PY scripts/analyze.py >> $LOG 2>&1
$PY scripts/explain.py >> $LOG 2>&1
step "PIPELINE_FIM"
