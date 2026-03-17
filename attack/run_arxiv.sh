#!/bin/bash

exp_name=LLaDA-8B-Base-pretrained-mimir-github-4_12_1.0e-5_10_512

for exp_name in Dream-v0-Base-7B-pretrained-mimir-github-4_12_1.0e-5_10_512; do
for config_name in config_all config_myp1 config_myp2 config_myp3; do

output_dir=./attack_results/${exp_name}/${config_name}
mkdir -p ${output_dir}
CUDA_VISIBLE_DEVICES=6 python -m attack.run \
    -c attack/configs/${config_name}.yaml \
    --output ${output_dir} \
    --base-dir /mnt/data/yuhongyao/paper_codes/difftext/SAMA/outputs/${exp_name} \
    --cache-dir ./attack_results/${exp_name}/cache \
    2>&1 | tee ${output_dir}/attack.log

done
done