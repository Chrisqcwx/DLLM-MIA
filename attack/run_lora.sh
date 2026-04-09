#!/bin/bash

# exp_name=LLaDA-8B-Base-pretrained-mimir-github-4_12_1.0e-5_10_512

# for config_name in config_samalossupdate ; do
# for config_name in config_mylossupdate ; do
# samanewloss
    # Dream-v0-Base-7B-pretrained-mimir-arxiv-1_6_1.0e-4_10_512-lora-mlp1 \
    # Dream-v0-Base-7B-pretrained-mimir-arxiv-1_6_1.0e-4_10_512-lora-mlp2 \
    # Dream-v0-Base-7B-pretrained-mimir-arxiv-1_6_1.0e-4_10_512-lora-attn1 \
    # Dream-v0-Base-7B-pretrained-mimir-arxiv-1_6_1.0e-4_10_512-lora-attn2 \

    # LLaDA-8B-Base-pretrained-mimir-arxiv-4_12_1.0e-4_10_512-lora-mlp1 \
    # LLaDA-8B-Base-pretrained-mimir-arxiv-4_12_1.0e-4_10_512-lora-mlp2 \
    # LLaDA-8B-Base-pretrained-mimir-arxiv-4_12_1.0e-4_10_512-lora-attn1 \
    # LLaDA-8B-Base-pretrained-mimir-arxiv-4_12_1.0e-4_10_512-lora-attn2 \

# for exp_name in LLaDA-8B-Base-pretrained-mimir-arxiv-4_12_1.0e-4_10_512-lora-mlpr1 \
#     LLaDA-8B-Base-pretrained-mimir-arxiv-4_12_1.0e-4_10_512-lora-mlpr2 \
#     LLaDA-8B-Base-pretrained-mimir-arxiv-4_12_1.0e-4_10_512-lora-mlpr4 \
#     LLaDA-8B-Base-pretrained-mimir-arxiv-4_12_1.0e-4_10_512-lora-mlpr2ffout \
#     LLaDA-8B-Base-pretrained-mimir-arxiv-4_12_1.0e-4_10_512-lora-mlpr2ffproj \
#     LLaDA-8B-Base-pretrained-mimir-arxiv-4_12_1.0e-4_10_512-lora-mlpr2uproj \
#     LLaDA-8B-Base-pretrained-mimir-arxiv-4_12_1.0e-4_10_512-lora-mlpr8 \
#     ; do
     
for exp_name in  \
    LLaDA-8B-Base-pretrained-mimir-arxiv-4_12_1.0e-4_10_512-lora-mlp3 \
    LLaDA-8B-Base-pretrained-mimir-arxiv-4_12_1.0e-4_10_512-lora-mlp4 \
    LLaDA-8B-Base-pretrained-mimir-arxiv-4_12_1.0e-4_10_512-lora-attn3 \
    LLaDA-8B-Base-pretrained-mimir-arxiv-4_12_1.0e-4_10_512-lora-attn4 \
     ; do

# LLaDA-8B-Base-pretrained-mimir-arxiv-4_12_1.0e-5_10_512-lora
# for exp_name in LLaDA-8B-Base-pretrained-mimir-arxiv-4_12_1.0e-5_10_512 ; do
# for exp_ratio in 1 0.5 2; do

for config_name in config_mtc5depend3 config_all config_mtc5  ; do

# for div_type in r t m n; do

output_dir=./attack_results/${exp_name}/${config_name}

# 如果output_dir下面有json文件则跳过，否则执行攻击
if ls ${output_dir}/*.json 1> /dev/null 2>&1;
then
    echo "Skip ${output_dir}"
    sleep 1s
else
    echo "Running ${output_dir}"
    
mkdir -p ${output_dir}
CUDA_VISIBLE_DEVICES=3 SAMA_METADATA_DIR=$output_dir \
    python -m attack.run \
    -c attack/configs/${config_name}.yaml \
    --output ${output_dir} \
    --base-dir /mnt/data/yuhongyao/paper_codes/difftext/SAMA/outputs/${exp_name} \
    --cache-dir ./attack_results/${exp_name}/cache \
    2>&1 | tee ${output_dir}/attack.log


fi
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