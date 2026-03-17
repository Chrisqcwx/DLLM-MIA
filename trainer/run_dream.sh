#!/bin/bash
CUDA_VISIBLE_DEVICES=2 python trainer/run.py \
  --config_path "./trainer/configs/Dream-v0-Base-7B-pretrained-mimir-arxiv.yaml" \
  --base_path "./outputs" \
  --train_subset_size 10000 \
  --ref_subset_size 1000