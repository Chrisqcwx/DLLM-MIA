#!/bin/bash

# "LLaDA-8B-Base-pretrained-mimir-arxiv.yaml" \
# "LLaDA-8B-Base-pretrained-mimir-arxiv-lora.yaml" \
# "LLaDA-8B-Base-pretrained-mimir-github.yaml" \
# "LLaDA-8B-Base-pretrained-ag_news.yaml" \
# "LLaDA-8B-Base-pretrained-mimir-github-epoch4.yaml" \
# "LLaDA-8B-Base-pretrained-mimir-arxiv-epoch4.yaml" \
# "LLaDA-8B-Base-pretrained-ag_news-epoch4.yaml" \

# for config_name in \
# "LLaDA-8B-Base-pretrained-wikitext-wikitext-103-v1.yaml" \
# "LLaDA-8B-Base-pretrained-mimir-pile_cc.yaml" \
# "LLaDA-8B-Base-pretrained-xsum.yaml" \
# "LLaDA-8B-Base-pretrained-mimir-pubmed_central.yaml" ;do

  # Dream-v0-Base-7B-pretrained-mimir-arxiv-lora-attn2.yaml \
  # Dream-v0-Base-7B-pretrained-mimir-arxiv-lora-mlp1.yaml \
  # Dream-v0-Base-7B-pretrained-mimir-arxiv-lora-mlp2.yaml \
  # Dream-v0-Base-7B-pretrained-mimir-arxiv-lora-attn1.yaml \
for config_name in \
  LLaDA-8B-Base-pretrained-mimir-arxiv-lora-mlpr1.yaml \
  LLaDA-8B-Base-pretrained-mimir-arxiv-lora-mlpr2ffproj.yaml \
  LLaDA-8B-Base-pretrained-mimir-arxiv-lora-mlpr2uproj.yaml \
  LLaDA-8B-Base-pretrained-mimir-arxiv-lora-mlpr2ffout.yaml \
 ;do
# "LLaDA-8B-Base-pretrained-mimir-arxiv.yaml" \
# for config_name in \
# "LLaDA-8B-Base-pretrained-mimir-arxiv-lora.yaml" ;do


CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file ./trainer/accelerate.yaml \
  --num_machines 1 \
  --num_processes 1 \
  trainer/run.py \
  --config_path "./trainer/configs/$config_name" \
  --base_path "./outputs" \
  --train_subset_size 10000 \
  --ref_subset_size 1000 2>&1 | tee ./outputs/train_${config_name%.yaml}.log

done