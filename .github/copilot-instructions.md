# CaS_DETR Project — Copilot Instructions

## Project Overview

This repository benchmarks object detection models on **DAIR-V2X** and related datasets. The current sparse DETR path is **CaS-on-DEIM**, implemented under `experiments/CaS-DETR/`, which ports token pruning, decoder MoE, and CASS supervision into the DEIM codebase while keeping DEIM training structure.

**8 detection classes**: Car, Truck, Van, Bus, Pedestrian, Cyclist, Motorcyclist, Trafficcone.

## Repository Layout

- `experiments/CaS-DETR/` — primary CaS-on-DEIM implementation
  - `engine/deim/` — sparse encoder, decoder MoE, and CASS wiring
  - `configs/base/` — shared defaults such as `cas_deim.yml`
  - `configs/deim_dfine/` — dataset-specific DEIM-style configs
  - `configs/dataset/ablation/` — current stage-1 ablation configs
  - `train.py` — DEIM-style training entrypoint
- `experiments/DEIM/`, `experiments/D-FINE/` — upstream-style DETR baselines
- `experiments/RT-DETR/rtdetrv2_pytorch/` — RT-DETR v2 baseline via `train_adapter`
- `experiments/deformable-detr/`, `experiments/yolo/` — comparison models
- `experiments/common/` — shared evaluation and visualization helpers

## Key Architecture Patterns

### Sparse Encoder

- `HybridEncoder` supports token pruning through `shared_token_pruner`
- pruning is currently used for semantic filtering, not aggressive full multi-level sparsification
- `use_caip` and `use_cass` remain configurable

### Decoder MoE

- `DFINETransformer` replaces decoder FFN with `MoELayer` when enabled
- load-balance loss is attached through `DEIMCriterion`

### CASS Supervision

- token-importance supervision is implemented in `token_level_pruning.py`
- the current default CASS loss type is `vfl`

## Configuration Conventions

YAML configs live in `experiments/CaS-DETR/configs/`. Important fields:

- `HybridEncoder.token_keep_ratio`
- `HybridEncoder.use_caip`
- `HybridEncoder.use_cass`
- `DFINETransformer.use_moe`
- `DFINETransformer.num_experts`
- `DFINETransformer.moe_top_k`
- `DEIMCriterion.cass_loss_weight`
- `DEIMCriterion.decoder_moe_balance_weight`

Current stage-1 ablations are under `experiments/CaS-DETR/configs/dataset/ablation/` and focus on DAIR-V2X.

## Common Workflows

```bash
./experiments/run_batch_experiments.sh
./experiments/run_batch_experiments.sh --test
./experiments/run_batch_experiments.sh --cas_detr
./experiments/run_batch_experiments.sh --cas_detr --k0.7
./experiments/run_batch_experiments.sh --dairv2x --cas_detr
```

## Coding Notes

- Prefer the `experiments/CaS-DETR/` tree for any new CaS-related work
- Do not reintroduce dependencies on the removed legacy `experiments/cas_detr/` directory
- Shared helpers in `experiments/common/` should stay framework-agnostic when possible
