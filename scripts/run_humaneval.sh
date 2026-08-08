#!/usr/bin/env bash
# HumanEval benchmark for K3 TP3 @ 8K context
# Uses --prompts-file to run all 20 prompts in one model load

set -euo pipefail

BIN=/home/vikassridhar/sparkinfer-k3/build/kimi_k3_dist_generate
MODEL=/home/vikassridhar/models/k3-neuron-iq1s/k3-neuron-iq1s-00001-of-00009.gguf
PROMPTS=/home/vikassridhar/humaneval_prompts/all_prompts.txt
COORD=10.10.10.13
PORT=41231
NPREDICT=512
CTX=8192
LOG_DIR=/home/vikassridhar/humaneval_run

mkdir -p "$LOG_DIR"

echo "=== HumanEval K3 TP3 Benchmark ==="
echo "Date: $(date)"
echo "Prompts: $PROMPTS"
echo "n-predict: $NPREDICT, max-ctx: $CTX"
echo "Log dir: $LOG_DIR"
echo ""

# Clean up any stale processes
for IP in $COORD 10.10.10.12 10.10.10.14; do
    ssh -o ConnectTimeout=5 vikassridhar@$IP 'killall kimi_k3_dist_generate 2>/dev/null; true' || true
done

# Write worker scripts
ssh vikassridhar@10.10.10.12 "cat > /tmp/run_he_r1.sh << EOF
#!/usr/bin/env bash
exec $BIN --rank 1 --world 3 --coord $COORD:$PORT --model $MODEL --n-predict $NPREDICT --max-ctx $CTX
EOF
chmod +x /tmp/run_he_r1.sh"

ssh vikassridhar@10.10.10.14 "cat > /tmp/run_he_r2.sh << EOF
#!/usr/bin/env bash
exec $BIN --rank 2 --world 3 --coord $COORD:$PORT --model $MODEL --n-predict $NPREDICT --max-ctx $CTX
EOF
chmod +x /tmp/run_he_r2.sh"

# Launch workers
echo "Launching workers..."
ssh vikassridhar@10.10.10.12 "nohup bash /tmp/run_he_r1.sh > $LOG_DIR/r1.log 2>&1 &"
ssh vikassridhar@10.10.10.14 "nohup bash /tmp/run_he_r2.sh > $LOG_DIR/r2.log 2>&1 &"

sleep 3

# Launch coordinator (foreground — blocks until done)
echo "Launching coordinator..."
$BIN --rank 0 --world 3 --listen 0.0.0.0:$PORT --model $MODEL \
    --prompts-file $PROMPTS --n-predict $NPREDICT --max-ctx $CTX \
    > "$LOG_DIR/r0.log" 2>"$LOG_DIR/r0_stderr.log"

echo ""
echo "=== Coordinator finished ==="
echo "Results: $LOG_DIR/r0.log"
echo "Stderr:  $LOG_DIR/r0_stderr.log"
date
