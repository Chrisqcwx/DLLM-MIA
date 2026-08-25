#!/bin/bash
# attack/run_attack.sh — consolidated entry that replaces the legacy
# run_epoch4.sh / run_epoch4.2.sh variants. Iterates over one or more
# attack configs for a single (already-trained) experiment name.
#
# Usage:
#   ./attack/run_attack.sh <exp_name> <config_name> [<config_name> ...]
#
# Examples:
#   # ITS on the LLaDA arxiv experiment
#   ./attack/run_attack.sh \
#       LLaDA-8B-Base-pretrained-mimir-arxiv-4_12_1.0e-5_4_512 \
#       config_its4-p0.1
#
#   # Full baseline sweep on wikitext
#   ./attack/run_attack.sh \
#       LLaDA-8B-Base-pretrained-wikitext_document_level-wikitext-103-v1-4_12_1.0e-5_4_512 \
#       config_all config_its4-p0.1
#
# Environment overrides:
#   BASE_DIR      target model directory   (default: ./outputs/<exp_name>)
#   OUTPUT_ROOT   results root             (default: ./attack_results/<exp_name>)
#   CACHE_DIR     tokenizer/model cache    (default: $OUTPUT_ROOT/cache)

set -e

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <exp_name> <config_name> [<config_name> ...]"
    exit 1
fi

EXP_NAME=$1
shift
CONFIG_NAMES="$@"

BASE_DIR=${BASE_DIR:-./outputs/${EXP_NAME}}
OUTPUT_ROOT=${OUTPUT_ROOT:-./attack_results/${EXP_NAME}}
CACHE_DIR=${CACHE_DIR:-${OUTPUT_ROOT}/cache}

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

for config_name in $CONFIG_NAMES; do
    output_dir=${OUTPUT_ROOT}/${config_name}
    if ls ${output_dir}/*.json 1>/dev/null 2>&1; then
        echo "Skip ${output_dir} (results already exist)"
        continue
    fi
    mkdir -p ${output_dir}
    echo "Running ${output_dir}"
    SAMA_METADATA_DIR=$output_dir \
        python -m attack.run \
        -c attack/configs/${config_name}.yaml \
        --output ${output_dir} \
        --base-dir ${BASE_DIR} \
        --cache-dir ${CACHE_DIR} \
        2>&1 | tee ${output_dir}/attack.log
done
