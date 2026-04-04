# CaS_DETR Readme

## Overview

This repository now uses `experiments/CaS-DETR/` as the active CaS-related implementation.

The current path is not the old standalone RT-DETR-style `cas_detr` project anymore. It is a **CaS-on-DEIM** variant:

- keep the overall training structure, config system, and entrypoint style of `DEIM`
- port in the CaS ideas that are still in use
- focus on model-side changes only

The main migrated ideas are:

- encoder-side token pruning
- optional CAIP switch
- decoder FFN MoE
- CASS loss

## Current Directory

Active implementation:

- `experiments/CaS-DETR/`

Important locations:

- `experiments/CaS-DETR/engine/deim/hybrid_encoder.py`
- `experiments/CaS-DETR/engine/deim/token_level_pruning.py`
- `experiments/CaS-DETR/engine/deim/moe_components.py`
- `experiments/CaS-DETR/engine/deim/dfine_decoder.py`
- `experiments/CaS-DETR/engine/deim/deim_criterion.py`
- `experiments/CaS-DETR/configs/base/cas_deim.yml`

## What Is Kept

The project currently keeps these configurable switches:

- `HybridEncoder.enable_cas_predictor`
- `HybridEncoder.token_keep_ratio`
- `HybridEncoder.use_caip`
- `HybridEncoder.use_cass`
- `DFINETransformer.use_moe`
- `DFINETransformer.num_experts`
- `DFINETransformer.moe_top_k`
- `DEIMCriterion.cass_loss_weight`
- `DEIMCriterion.decoder_moe_balance_weight`

## Current Stage-1 Ablations

The current first-stage ablation set is limited to **DAIR-V2X** and lives in:

- `experiments/CaS-DETR/configs/dataset/ablation/`

Current five configs:

- `cas_deim_moe_only_hgnetv2_s_dairv2x.yml`
- `cas_deim_cass_only_keep07_hgnetv2_s_dairv2x.yml`
- `cas_deim_cass_only_keep05_hgnetv2_s_dairv2x.yml`
- `cas_deim_moe_cass_keep07_hgnetv2_s_dairv2x.yml`
- `cas_deim_moe_cass_keep05_hgnetv2_s_dairv2x.yml`

Their meanings are:

- `moe_only`: decoder MoE only, no pruning, no CASS
- `cass_only_keep07`: pruning + CASS, fixed `token_keep_ratio=0.7`, no decoder MoE
- `cass_only_keep05`: pruning + CASS, fixed `token_keep_ratio=0.5`, no decoder MoE
- `moe_cass_keep07`: pruning + CASS + decoder MoE, fixed `token_keep_ratio=0.7`
- `moe_cass_keep05`: pruning + CASS + decoder MoE, fixed `token_keep_ratio=0.5`

For this stage:

- `use_caip=False`
- only fixed keep ratios are compared
- `moe_only` does not prune

## Default Base Config

Base defaults are defined in:

- `experiments/CaS-DETR/configs/base/cas_deim.yml`

Current defaults include:

- `token_keep_ratio: 0.3`
- `use_caip: True`
- `use_cass: True`
- `cass_loss_type: vfl`
- `num_experts: 6`
- `moe_top_k: 3`

These are only base defaults. The stage-1 ablation files override them where needed.

## How To Run

Batch script:

```bash
./experiments/run_batch_experiments.sh --cas_detr
```

Only DAIR-V2X CaS stage-1:

```bash
./experiments/run_batch_experiments.sh --dairv2x --cas_detr
```

Only keep `0.7` branch:

```bash
./experiments/run_batch_experiments.sh --cas_detr --k0.7
```

Only keep `0.5` branch:

```bash
./experiments/run_batch_experiments.sh --cas_detr --k0.5
```

Quick test mode:

```bash
./experiments/run_batch_experiments.sh --test --cas_detr
```

## Notes

- The old `experiments/cas_detr/` tree has been removed from the repository.
- New CaS-related work should be based on `experiments/CaS-DETR/`.
- Shared helpers under `experiments/common/` should avoid depending on removed legacy paths.
