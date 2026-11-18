#!/usr/bin/env bash
# Runs the whole rerun: prepare, train base, fine-tune, generate, evaluate.
#
#   DATA_ROOT=~/data/final_dataset RUN_DIR=~/runs/rerun1 SHUTDOWN_WHEN_DONE=1 bash run_all.sh
#
# Every step skips work that is already finished, so after an interruption you
# start the same command again and it continues where it stopped.
# SHUTDOWN_WHEN_DONE=1 stops the machine when the script ends, whether it
# succeeded or failed, so an EC2 instance never sits idle and billing.
set -euo pipefail

DATA_ROOT=${DATA_ROOT:-$HOME/data/final_dataset}
RUN_DIR=${RUN_DIR:-$HOME/runs/rerun1}
EXTRA_ARGS=${EXTRA_ARGS:-}            # "--smoke" in the laptop smoke test
TRAIN_ARGS=${TRAIN_ARGS:-}            # e.g. "--gradient_checkpointing" if training runs out of GPU memory
SHUTDOWN_WHEN_DONE=${SHUTDOWN_WHEN_DONE:-0}
PY=${PY:-python}
export PYTHONUNBUFFERED=1

cd "$(dirname "$0")"
if [ "$SHUTDOWN_WHEN_DONE" = "1" ]; then
  trap 'echo "run_all.sh finished; shutting down in 60 s (Ctrl-C to cancel)"; sleep 60; sudo shutdown -h now' EXIT
fi
mkdir -p "$RUN_DIR/logs"
log() { tee -a "$RUN_DIR/logs/$1.log"; }

BASE="$RUN_DIR/checkpoints/base/final/unet"
FT="$RUN_DIR/checkpoints/finetune/final/unet"

echo "== 1/5 prepare"
$PY prepare.py --data_root "$DATA_ROOT" --run_dir "$RUN_DIR" $EXTRA_ARGS 2>&1 | log prepare

echo "== 2/5 train base model"
$PY train.py --stage base --data_root "$DATA_ROOT" --out_dir "$RUN_DIR/checkpoints/base" $EXTRA_ARGS $TRAIN_ARGS 2>&1 | log train_base

echo "== 3/5 fine-tune on leprosy"
$PY train.py --stage finetune --data_root "$DATA_ROOT" --split "$RUN_DIR/split.json" \
  --init_unet "$BASE" --out_dir "$RUN_DIR/checkpoints/finetune" $EXTRA_ARGS $TRAIN_ARGS 2>&1 | log train_finetune

echo "== 4/5 generate"
gen() { $PY generate.py "$@" $EXTRA_ARGS 2>&1 | log generate; }
gen --unet_dir "$BASE" --manifest "$RUN_DIR/manifests/heldout_gen.json" --out_dir "$RUN_DIR/gen/heldout_base"
gen --unet_dir "$FT"   --manifest "$RUN_DIR/manifests/heldout_gen.json" --out_dir "$RUN_DIR/gen/heldout_ft"
gen --unet_dir "$BASE" --prompt_from "$FT" --seeds first \
    --manifest "$RUN_DIR/manifests/heldout_gen.json" --out_dir "$RUN_DIR/gen/heldout_base_ftprompt"
gen --unet_dir "$BASE" --manifest "$RUN_DIR/manifests/kaggle_gen.json" --out_dir "$RUN_DIR/gen/kaggle_base"
gen --unet_dir "$FT"   --manifest "$RUN_DIR/manifests/kaggle_gen.json" --out_dir "$RUN_DIR/gen/kaggle_ft"

echo "== 5/5 evaluate"
$PY evaluate.py --run_dir "$RUN_DIR" $EXTRA_ARGS 2>&1 | log evaluate

# Small bundle to download first: metrics, figures, configs, logs (no weights or images).
tar -czf "$RUN_DIR/results.tar.gz" -C "$RUN_DIR" metrics manifests split.json prepare_summary.json logs \
  checkpoints/base/final/train_config.json checkpoints/base/final/history.json \
  checkpoints/finetune/final/train_config.json checkpoints/finetune/final/history.json
echo "Done. Results bundle: $RUN_DIR/results.tar.gz"
