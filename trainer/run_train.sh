#!/bin/bash
# trainer/run_train.sh — consolidated entry that replaces the legacy
# run_epoch4.sh and run_epoch4_dream.sh variants. Trains one or more
# configs (LLaDA or Dream) on the requested GPUs.
#
# Usage:
#   ./trainer/run_train.sh <gpus_csv> <num_procs> <config_name> [<config_name> ...]
#
# Examples:
#   # LLaDA 8B on wikitext (2 GPUs)
#   ./trainer/run_train.sh 6,7 2 \
#       LLaDA-8B-Base-pretrained-wikitext-wikitext-103-v1-epoch4.yaml
#
#   # Dream 7B on mimir-arxiv (2 GPUs)
#   ./trainer/run_train.sh 1,2 2 \
#       Dream-v0-Base-7B-pretrained-mimir-arxiv-epoch4.yaml

set -e

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <gpus_csv> <num_procs> <config_name> [<config_name> ...]"
    exit 1
fi

GPUS=$1
NUM_PROCS=$2
shift 2
CONFIG_NAMES="$@"

for config_name in $CONFIG_NAMES; do
    CUDA_VISIBLE_DEVICES=${GPUS} accelerate launch --config_file ./trainer/accelerate.yaml \
        --num_machines 1 \
        --num_processes ${NUM_PROCS} \
        trainer/run.py \
        --config_path "./trainer/configs/$config_name" \
        --base_path "./outputs" \
        --train_subset_size 10000 \
        --ref_subset_size 1000 2>&1 | tee ./outputs/train_${config_name%.yaml}.log
done
