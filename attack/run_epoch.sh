#!/bin/bash

exp_name=LLaDA-8B-Base-pretrained-mimir-github-4_12_1.0e-5_10_512

# for config_name in config_samalossupdate ; do
# for config_name in config_mylossupdate ; do
# samanewloss


for exp_name in LLaDA-8B-Base-pretrained-mimir-github-4_12_1.0e-5_10_512 \
    ; do

for epoch in 10 20 31 41 52 62 72 83 93 100; do
     
# for exp_name in LLaDA-8B-Base-pretrained-mimir-github-4_12_1.0e-5_4_512 \
#      ; do

# LLaDA-8B-Base-pretrained-mimir-arxiv-4_12_1.0e-5_10_512-lora
# for exp_name in LLaDA-8B-Base-pretrained-mimir-arxiv-4_12_1.0e-5_10_512 ; do
# for exp_ratio in 1 0.5 2; do

for config_name in config_mtc5 ; do

# for div_type in r t m n; do

output_dir=./attack_results/ablations/epoch/${epoch}/${exp_name}/${config_name}
mkdir -p ${output_dir}
CUDA_VISIBLE_DEVICES=4 SAMA_METADATA_DIR=$output_dir \
    python -m attack.run \
    -c attack/configs/${config_name}.yaml \
    --output ${output_dir} \
    --base-dir /mnt/data/yuhongyao/paper_codes/difftext/SAMA/outputs/ablations/epoch_num/ablations/epoch_num/${exp_name}/ckpts/checkpoint-${epoch} \
    --cache-dir ./attack_results/${exp_name}/cache \
    2>&1 | tee ${output_dir}/attack.log

done
done
done
# for exp_name in LLaDA-8B-Base-pretrained-mimir-github-4_12_1.0e-5_10_512 ; do

# for config_name in config_samamultirun ; do

# for run_times in 2 3 4; do

# # for div_type in r t m n; do

# output_dir=./attack_results/${exp_name}/${config_name}_${run_times}
# mkdir -p ${output_dir}
# CUDA_VISIBLE_DEVICES=3 SAME_RUNTIMES=$run_times SAMA_METADATA_DIR=$output_dir \
#     python -m attack.run \
#     -c attack/configs/${config_name}.yaml \
#     --output ${output_dir} \
#     --base-dir /mnt/data/yuhongyao/paper_codes/difftext/SAMA/outputs/${exp_name} \
#     --cache-dir ./attack_results/${exp_name}/cache \
#     2>&1 | tee ${output_dir}/attack.log
# done
# done

# done
# done