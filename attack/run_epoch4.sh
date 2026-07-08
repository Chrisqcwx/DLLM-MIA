#!/bin/bash

exp_name=LLaDA-8B-Base-pretrained-mimir-github-4_12_1.0e-5_10_512

# for config_name in config_samalossupdate ; do
# for config_name in config_mylossupdate ; do
# samanewloss

#     LLaDA-8B-Base-pretrained-mimir-github-4_12_1.0e-5_10_512 \
#     LLaDA-8B-Base-pretrained-mimir-arxiv-4_12_1.0e-5_10_512 \
#     LLaDA-8B-Base-pretrained-ag_news-4_12_1.0e-5_4_512 \
#     LLaDA-8B-Base-pretrained-ag_news-4_12_1.0e-5_10_512 \
#    LLaDA-8B-Base-pretrained-mimir-pile_cc-4_12_1.0e-5_10_512 \
    # LLaDA-8B-Base-pretrained-ag_news-4_12_1.0e-5_4_512 \
    # LLaDA-8B-Base-pretrained-mimir-github-4_12_1.0e-5_4_512 \
    # LLaDA-8B-Base-pretrained-mimir-pile_cc-4_12_1.0e-5_4_512 \
    # LLaDA-8B-Base-pretrained-mimir-pubmed_central-4_12_1.0e-5_4_512 \
    # "LLaDA-8B-Base-pretrained-mimir-wikipedia_(en)-4_12_1.0e-5_4_512" \
for config_name in config_mtc5depend4neg3-p0.1-least1  \
    config_mtc5depend4cut1-p0.1-least1 \
    config_mtc5depend4cut2-p0.1-least1 \
    config_mtc5depend4cut3-p0.1-least1 \
    config_mtc5depend4cut4-p0.1-least1 \
    config_mtc5depend4step3-p0.1-least1 \
    ; do
# for exp_name in \
#     LLaDA-8B-Base-pretrained-mimir-arxiv-4_12_1.0e-5_4_512 \
#     ; do

for exp_name in \
    LLaDA-8B-Base-pretrained-mimir-arxiv-4_12_1.0e-5_4_512 \
    ; do
# for exp_name in \
#     LLaDA-8B-Base-pretrained-mimir-github-4_12_1.0e-5_4_512 \
#     ; do
     
# for exp_name in  \
# LLaDA-8B-Base-pretrained-mimir-pubmed_central-4_12_1.0e-5_4_512 \
# "LLaDA-8B-Base-pretrained-mimir-wikipedia_(en)-4_12_1.0e-5_4_512" \
#      ; do

# LLaDA-8B-Base-pretrained-mimir-arxiv-4_12_1.0e-5_10_512-lora
# for exp_name in LLaDA-8B-Base-pretrained-mimir-arxiv-4_12_1.0e-5_10_512 ; do
# for exp_ratio in 1 0.5 2; do

# for config_name in config_mtc5depend4step2 config_samastep2  ; do
# for config_name in config_baselinecontrast2ratio config_baselinecontrast2  config_baselineorigin2dfmia  ; do
# for config_name in config_baselineorigin2 config_baselineorigin2mink config_baselineorigin2minkpp; do
# for div_type in r t m n; do

output_dir=./attack_results/${exp_name}/${config_name}

if ls ${output_dir}/*.json 1> /dev/null 2>&1;
then
    echo "Skip ${output_dir}"
    # sleep 1s
else
    echo "Running ${output_dir}"
    # sleep 1s


export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

mkdir -p ${output_dir}
# CUDA_VISIBLE_DEVICES=0  \
SAMA_METADATA_DIR=$output_dir \
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