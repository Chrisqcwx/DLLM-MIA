#!/bin/bash

# "LLaDA-8B-Base-pretrained-mimir-arxiv.yaml" \
# "LLaDA-8B-Base-pretrained-mimir-arxiv-lora.yaml" \
# "LLaDA-8B-Base-pretrained-mimir-github.yaml" \
# "LLaDA-8B-Base-pretrained-ag_news.yaml" \

for config_name in \
  "LLaDA-8B-Base-pretrained-mimir-github-lora.yaml" \
   "LLaDA-8B-Base-pretrained-mimir-arxiv-lora2.yaml" \
   ;do
# "LLaDA-8B-Base-pretrained-mimir-arxiv.yaml" \
# for config_name in \
# "LLaDA-8B-Base-pretrained-mimir-arxiv-lora.yaml" ;do


# phoenixcat launch \
#   --gpu-use-num 2 --gpu-ids 0-7 --threshold-gb 2 \
#   -- \
  CUDA_VISIBLE_DEVICES=0,2 accelerate launch --config_file ./trainer/accelerate.yaml \
  --num_machines 1 \
  --num_processes 2 \
  trainer/run.py \
  --config_path "./trainer/configs/$config_name" \
  --base_path "./outputs" \
  --train_subset_size 10000 \
  --ref_subset_size 1000 2>&1 | tee ./outputs/train_${config_name%.yaml}.log

done