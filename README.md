# Weak Ties, Strong Signals: Efficient Training Data Detection in Diffusion LLMs via Independent Token Sampling

[![arXiv](https://img.shields.io/badge/EMNLP-2026-blue.svg)](https://aclanthology.org/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

> **Hongyao Yu, Tianqu Zhuang, Ziyuan Xu, Hao Fang, Jiaxin Hong, Bin Chen, Shu-Tao Xia — EMNLP 2026**
> Implementation of **Independent Token Sampling (ITS)** for membership-inference /
> training-data detection on diffusion language models (dLLMs).

This repository contains the implementation of **ITS** and the baseline attacks
used to evaluate training-data leakage of fine-tuned **Diffusion Language
Models** such as [LLaDA](https://huggingface.co/GSAI-ML/LLaDA-8B-Base) and
[Dream](https://huggingface.co/Dream-org/Dream-v0-Base-7B).



---

## Why ITS?

Random token sampling (e.g. SAMA) treats each masked token's loss as if it
contributed independently to the joint probability. We show this introduces a
**structural estimation error** characterised by the cumulative conditional
mutual information (CMI) among the masked tokens — random subsets can hide
memorisation signals.

ITS replaces random subsets with two complementary mechanisms:

| | |
|---|---|
| **Attention-guided proxy** | Use the reference dLLM's self-attention as a pairwise dependency proxy, and let `sample_engine` pick token subsets with *low* internal dependency. |
| **Diversity penalty** (`lambda_penalty`) | Across the `num_subsets` subsets drawn at each step, discourage resampling the same tokens so coverage stays broad under a tight query budget. |

See [Method: ITS in one paragraph](#method-its-in-one-paragraph) below for the
end-to-end algorithm.

---

## Repository Structure

```
.
├── trainer/                    # DLM fine-tuning module
│   ├── model/
│   │   ├── llada/              # LLaDA-8B architecture files
│   │   └── dream/              # Dream-v0-Base-7B architecture files
│   ├── misc/                   # data.py, env_setup.py, models.py, trainer.py, utils.py
│   ├── configs/                # Core training YAMLs (4 shipped)
│   ├── run.py                  # accelerate launch entrypoint
│   ├── run_train.sh            # Consolidated training launcher
│   └── requirements.txt
│
├── attack/                     # Membership-inference attacks
│   ├── attacks/                # Attack implementations
│   │   ├── its.py              # ← proposed method (ITS)
│   │   ├── sama.py             # SAMA baseline (Chen et al., 2026)
│   │   ├── loss.py, ratio.py, zlib.py           # NLL / Loss-Calibration / Zlib
│   │   ├── dfmia.py, mink.py, minkpp.py         # DF-MIA, Min-K, Min-K++
│   │   └── utils.py
│   ├── configs/                # YAMLs for LLaDA-8B targets
│   ├── configs_dream/          # YAMLs for Dream-7B targets
│   ├── misc/                   # attack.py, config.py, dataset.py, io.py, metric.py, models.py, ...
│   ├── run.py                  # Attack entrypoint: python -m attack.run
│   └── run_attack.sh           # Consolidated attack launcher
│
└── dataset/                    # Dataset preparation utilities
    └── prep.py, prep_mimir.py
```

---

## Requirements

### Environment Setup

```bash
conda create -n its python=3.8
conda activate its

pip install -r trainer/requirements.txt
pip install -r attack/requirements.txt   # if present, otherwise see attack/misc
```

### Environment Variables

| Variable | Used in | Default |
|---|---|---|
| `SAMA_DATASET_PATH` | dataset prep helpers | `./` |
| `SAMA_METADATA_DIR` | `attack/attacks/its.py`, `attack/attacks/sama.py` | `./` |
| `SAMA_WANDB_PROJECT` | `trainer/configs/prep.py` | `Diff_LLM` |
| `SAMA_WANDB_GROUP` | `trainer/configs/prep.py` | `SAMA` |
| `HF_TOKEN` | `attack/attacks/sama.py`, `attack/attacks/ratio.py` | empty |

---

## Usage

### 1. Prepare datasets

```bash
python dataset/prep_mimir.py                # MIMIR benchmark (arxiv, github, pile_cc, ...)
python dataset/prep.py                      # wikitext / ag_news / xsum
```

### 2. Fine-tune a target DLM

Use the consolidated launcher:

```bash
# LLaDA-8B on MIMIR arxiv (4 epochs, 2 GPUs)
./trainer/run_train.sh 0,1 2 \
    LLaDA-8B-Base-pretrained-mimir-arxiv-epoch4.yaml

# Dream-7B on MIMIR arxiv (4 epochs, 2 GPUs)
./trainer/run_train.sh 2,3 2 \
    Dream-v0-Base-7B-pretrained-mimir-arxiv-epoch4.yaml
```

The four shipped training configs are:

- `trainer/configs/LLaDA-8B-Base-pretrained-mimir-arxiv.yaml`
- `trainer/configs/LLaDA-8B-Base-pretrained-mimir-arxiv-epoch4.yaml`
- `trainer/configs/Dream-v0-Base-7B-pretrained-mimir-arxiv.yaml`
- `trainer/configs/Dream-v0-Base-7B-pretrained-mimir-arxiv-epoch4.yaml`

Run `python trainer/configs/prep.py` to regenerate additional variants
(dataset, LoRA, etc.).

### 3. Run training-data-detection attacks

```bash
./attack/run_attack.sh \
    LLaDA-8B-Base-pretrained-mimir-arxiv-4_12_1.0e-5_4_512 \
    config_its4-p0.1

# Or run the full baseline sweep + ITS in one go:
./attack/run_attack.sh \
    LLaDA-8B-Base-pretrained-mimir-arxiv-4_12_1.0e-5_4_512 \
    config_all config_its4-p0.1
```

For Dream targets, swap to the matching `configs_dream/` YAMLs.

Under the hood `run_attack.sh` calls:

```bash
python -m attack.run \
    -c attack/configs/<config_name>.yaml \
    --output ./attack_results/<exp_name>/<config_name> \
    --base-dir ./outputs/<exp_name> \
    --cache-dir ./attack_results/<exp_name>/cache
```

| Argument | Description |
|---|---|
| `-c, --config` | Path to attack configuration YAML (required) |
| `--output` | Directory to save results and metadata (required) |
| `--base-dir` | Base directory for resolving relative model/dataset paths |
| `--target-model` | Path to target model (overrides config) |
| `--lora-path` | Path to a LoRA adapter (overrides config) |
| `--seed` | Random seed (default `42`) |

---

## Attack Configurations

| Config (in `attack/configs/`) | Purpose |
|---|---|
| `config_its4-p0.1.yaml` | ITS at λ=0.1, 4 progressive-mask steps (recommended headline config) |
| `config_all.yaml` | Runs Loss, Loss-Calibration, Zlib, Sama and Its in one file |
| `config_dfmia.yaml` | DF-MIA baseline only |
| `config_mink.yaml`, `config_minkpp.yaml` | Min-K and Min-K++ baselines |

Mirror configs for Dream targets live in `attack/configs_dream/`.

### ITS YAML at a glance

```yaml
Its:
  module: "its"
  steps: 4                   # progressive-mask steps
  subset_size: 8             # tokens per vote
  num_subsets: 128           # subsets per step
  l_schedule: "linear"       # or "geometric"
  min_mask_frac: 0.05        # mask fraction at step 0
  max_mask_frac: 0.50        # mask fraction at the last step
  lambda_penalty: 0.1        # diversity penalty across sampling rounds
  sample_alpha: 1.0          # internal-connectivity coefficient (positive = repel)
  least_m: 3                 # top-m tie-breaking window when picking the next token
  weight_start_layer_ratio: 0
  weight_end_layer_ratio: 1  # use all layers' attention as the dependency proxy
  reference_model_path: "GSAI-ML/LLaDA-8B-Base"   # reference dLLM
  batch_size: 8
  max_length: 512
  seed: 42
  save_metadata: false
```

The `sample_engine` function in [attack/attacks/its.py](attack/attacks/its.py)
implements the iterative graph sampler. `sample_alpha` controls how strongly
already-selected tokens *repel* the next pick (the dependency proxy), while
`lambda_penalty` *rewards diversity* across the multiple subsets by penalising
tokens that have already been chosen in earlier samples.

---

## Method: ITS in one paragraph

For each evaluation text, ITS first computes an L×L attention-derived
adjacency on the unmasked input using the reference dLLM; this adjacency is
used as a **pairwise dependency proxy** that approximates the cumulative
conditional mutual information (CMI) among masked tokens. Across `steps`
progressive masking rounds, ITS draws `num_subsets` disjoint groups of
`subset_size` tokens whose internal dependency is low; the diversity penalty
keeps successive samples complementary. Each newly-sampled token subset is
masked on both the target and the reference model, and per-token
cross-entropies on the masked positions are aggregated into a per-subset vote
(`ref_loss > target_loss`?). Step-level vote means are averaged into the final
membership score.

---

## Method: SAMA Baseline (Chen et al., 2026)

SAMA ([attack/attacks/sama.py](attack/attacks/sama.py)) is the closest prior
work — it samples random token subsets at each progressive-mask step and
averages the per-subset loss comparison. ITS keeps SAMA's framework and
replaces the uniform sampler with the attention-aware, diversity-aware
graph sampler above. SAMA serves as the strongest non-dependency-aware
baseline in our experiments.

> **Reference:**
> Chen, Y., Zhang, K., Du, Y., Stoppa, E., Fleming, C., Kundu, A., Ribeiro, B.,
> & Li, N. (2026). *Membership Inference Attacks Against Fine-tuned Diffusion
> Language Models.* arXiv:2601.20125.

---

## Other Baselines

| Baseline | File | Idea |
|---|---|---|
| Loss | `attack/attacks/loss.py` | Plain per-token NLL averaged over `mc_num` Monte-Carlo masks |
| Loss-Calibration | `attack/attacks/ratio.py` | NLL divided by the same NLL on a reference dLLM |
| Zlib | `attack/attacks/zlib.py` | NLL divided by zlib compression entropy |
| DF-MIA | `attack/attacks/dfmia.py` | Reference-free score → pseudo-non-member calibration pool |
| Min-K | `attack/attacks/mink.py` | Mean of the top-`mink` highest per-token losses |
| Min-K++ | `attack/attacks/minkpp.py` | Min-K on top of label-normalised log-probabilities |

---

## Citation

If you use ITS in your work, please cite our paper:

```bibtex

```
