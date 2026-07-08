#!/bin/bash

# exp_name=LLaDA-8B-Base-pretrained-mimir-github-4_12_1.0e-5_10_512

# CUDA_VISIBLE_DEVICES=6 python -m attack.run \
#     -c attack/configs/config_all.yaml \
#     --output ./attack_results/${exp_name} \
#     --base-dir /mnt/data/yuhongyao/paper_codes/difftext/SAMA/outputs/${exp_name} \
#     --cache-dir ./attack_results/${exp_name}/cache \
#     2>&1 | tee ./attack_results/${exp_name}/attack.log


bash attack/run_epoch4.sh
bash attack/run_lora.sh