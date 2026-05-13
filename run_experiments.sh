#!/usr/bin/env bash
# Sequential experiment runner.
#
# Runs each config in CONFIGS in order, logs to a timestamped per-run file,
# and continues to the next even if one fails (so a bad run doesn't kill
# the queue).  All training output goes to logs/<timestamp>/<config>.log;
# this script itself only prints START/OK/FAIL banners + timing.
#
# Usage
# -----
# Foreground (terminal must stay open):
#     ./run_experiments.sh
#
# Background, survives logout, output to file:
#     nohup ./run_experiments.sh > logs/dispatcher.log 2>&1 &
#     # follow:  tail -f logs/dispatcher.log
#
# Inside a tmux session (recommended — detach with C-b d, reattach later):
#     tmux new -s exp
#     ./run_experiments.sh
#
# GPU selection (defaults to GPU 2 = our RTX 5090):
#     CUDA_VISIBLE_DEVICES=1 ./run_experiments.sh
#
# Skip a config: comment it out in the CONFIGS array below.

set -u
cd "$(dirname "${BASH_SOURCE[0]}")"

# ---- config queue ------------------------------------------------------------

CONFIGS=(
    # ---- NTU120 ablation queue (priority order) -----------------------------
    # 1. Single-view baseline (essential — without this we can't claim
    #    multi-view fusion helps).
    configs/ntu120_xsub_1view.yaml

    # 2. Multi-view + augmentation (Issue 1 — the main candidate).
    configs/ntu120_xsub_adamw_aug.yaml

    # 3. SA-heads ablation (Issue 3): 1, 2 (heads=4 is covered by #2).
    configs/ntu120_xsub_adamw_aug_sa1.yaml
    configs/ntu120_xsub_adamw_aug_sa2.yaml

    # 4. Per-view BN ablation (Issue 2).
    configs/ntu120_xsub_adamw_aug_pvbn.yaml

    # 5. SA = weighted-sum ablation (Issue 3 alt).
    configs/ntu120_xsub_adamw_aug_wsum.yaml

    # 6. Longer schedule (Issue 4) — re-run the best config at 100 epochs
    #    once results from #2–#5 are in.  Left in by default; comment out
    #    to skip the +3 h cost.
    configs/ntu120_xsub_adamw_aug_100ep.yaml

    # ---- legacy / NTU60 runs (commented out by default; uncomment as needed) -
    # configs/ntu60_xsub_adamw.yaml
    # configs/ntu60_xsub_sgd.yaml
    # configs/ntu60_xsub_adamw_long.yaml
)

# ---- env defaults ------------------------------------------------------------

: "${CUDA_VISIBLE_DEVICES:=2}"
export CUDA_VISIBLE_DEVICES

PYTHON_BIN="${PYTHON_BIN:-conda run -n pose-ctr python}"

# ---- logging dir -------------------------------------------------------------

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs/$STAMP"
mkdir -p "$LOG_DIR"

echo "==> dispatcher started at $STAMP"
echo "==> CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "==> ${#CONFIGS[@]} configs queued · logs → $LOG_DIR"
for c in "${CONFIGS[@]}"; do echo "      - $c"; done

# ---- queue loop --------------------------------------------------------------

PASS=0
FAIL=0
START_ALL=$SECONDS

for cfg in "${CONFIGS[@]}"; do
    name=$(basename "$cfg" .yaml)
    log="$LOG_DIR/${name}.log"
    started=$(date '+%Y-%m-%d %H:%M:%S')
    t0=$SECONDS

    echo
    echo "==> $started  START  $name"
    echo "         config : $cfg"
    echo "         log    : $log"

    if $PYTHON_BIN train.py --config "$cfg" > "$log" 2>&1; then
        dur=$(( SECONDS - t0 ))
        echo "==> $(date '+%Y-%m-%d %H:%M:%S')  OK     $name  (${dur}s)"
        PASS=$(( PASS + 1 ))
    else
        rc=$?
        dur=$(( SECONDS - t0 ))
        echo "==> $(date '+%Y-%m-%d %H:%M:%S')  FAIL   $name  (rc=$rc, ${dur}s — see $log)"
        FAIL=$(( FAIL + 1 ))
    fi
done

# ---- summary -----------------------------------------------------------------

total=$(( SECONDS - START_ALL ))
echo
echo "==> done at $(date '+%Y-%m-%d %H:%M:%S')"
echo "==> total: ${total}s  ·  pass=$PASS  fail=$FAIL"
echo "==> logs:  $LOG_DIR"
