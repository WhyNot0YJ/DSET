# CaS_DETR Readme

This document describes the current implementation under `experiments/cas_detr`.
It is intended to match the code as it exists now, rather than older reports or
historical experiment notes.

## 1. What CaS_DETR is in this repo

CaS_DETR is a customized RT-DETR-style detector with two main ideas:

1. **Sparse tokens in the encoder**
   - A shared token-pruning module scores encoder tokens.
   - Only a fraction of the most important tokens are sent into the Transformer
     encoder branch.

2. **MoE FFN in the decoder**
   - The decoder keeps standard self-attention and cross-attention.
   - The decoder FFN is replaced by a Mixture-of-Experts layer.

In the current codebase, the main training path is implemented directly in:

- `experiments/cas_detr/train.py`

The actual model class used for training is:

- `CaS_DETRRTDETR`

The main trainer class is:

- `CaS_DETRTrainer`

## 2. Current architecture

### 2.1 Backbone

The backbone is created by `create_backbone()` in `train.py`.

Currently supported families include:

- `presnet*`
- `hgnetv2_*`
- `cspresnet*`
- `cspdarknet`
- `mresnet`

The checked-in CaS-DETR configs use:

- `presnet18`

### 2.2 Hybrid Encoder

The encoder implementation lives in:

- `experiments/cas_detr/src/zoo/rtdetr/hybrid_encoder.py`

The current encoder flow is:

1. Project multi-scale backbone features to the hidden dimension.
2. Select the feature levels listed in `model.encoder.use_encoder_idx`.
3. Flatten and concatenate tokens from those selected levels.
4. Apply shared token pruning.
5. Run a standard Transformer encoder on the kept tokens.
6. Scatter the encoded sparse tokens back to their original feature-map layout.
7. Continue with FPN/PAN style top-down and bottom-up fusion.

Important note:

- The encoder is **not** MoE.
- The encoder sparsity comes from **token pruning**, not expert routing.

### 2.3 Token pruning

The pruning logic lives in:

- `experiments/cas_detr/src/zoo/rtdetr/token_level_pruning.py`

Key characteristics of the current implementation:

- Shared global pruning across selected encoder levels
- Controlled by `model.cas_detr.token_keep_ratio`
- Pruning is enabled in both training and evaluation
- The pruner can also receive CASS supervision

### 2.4 CASS supervision

CASS is an explicit supervision signal for token importance prediction.

Relevant config fields:

- `model.cas_detr.use_cass`
- `model.cas_detr.cass_loss_weight`
- `model.cas_detr.cass_expansion_ratio`
- `model.cas_detr.cass_min_size`
- `model.cas_detr.cass_loss_type`
- `model.cas_detr.cass_focal_alpha`
- `model.cas_detr.cass_focal_beta`

This is part of the actual training loss in `train.py`.

### 2.5 Decoder

The decoder implementation lives in:

- `experiments/cas_detr/src/zoo/rtdetr/rtdetrv2_decoder.py`

Current behavior:

- RT-DETR-style query selection and decoder input construction
- Standard decoder self-attention
- Standard deformable cross-attention
- **MoE layer replaces the FFN**

Important note:

- MoE is only used in the decoder FFN path.
- It is not a class-specific hard-coded expert assignment.
- Expert routing is learned dynamically by a router.

### 2.6 Prediction heads

Prediction heads follow RT-DETR conventions:

- score head for classification logits
- bbox head for normalized `cxcywh` box prediction

The post-processing path expects:

- `pred_logits`
- `pred_boxes`

## 3. Training losses used now

The current training objective is a combination of:

1. **RT-DETR detection losses**
   - VFL classification loss
   - bbox loss
   - GIoU loss
   - auxiliary losses
   - denoising losses

2. **Decoder MoE balance loss**
   - controlled by `training.decoder_moe_balance_weight`
   - can be delayed using `training.moe_balance_warmup_epochs`

3. **CASS loss**
   - only active when `use_cass: true`

## 4. Current config files

The current checked-in CaS-DETR configs use token keep ratio **0.5** only (all use **S5-only** into the hybrid encoder, `use_encoder_idx: [2]`):

- `experiments/cas_detr/configs/cas_detr6_r18_ratio0.5.yaml` — DAIR-V2X
- `experiments/cas_detr/configs/cas_detr6_r18_ratio0.5_uadetrac.yaml` — UA-DETRAC
- `experiments/cas_detr/configs/cas_detr6_r18_ratio0.5_640.yaml` — DAIR-V2X, multi-scale disabled (640 only)

RT-DETR and MoE-RT-DETR use the same S5-only encoder setting in `rtdetr_r18.yaml`, `moe_rtdetr6_r18.yaml`, and UA-DETRAC counterparts.

Their meanings:

- `ratio0.5`: token keep ratio 0.5
- `uadetrac`: UA-DETRAC dataset variant

## 5. Current default experiment behavior

### DAIR-V2X configs

- `num_queries: 300`
- `batch_size: 24`
- `epochs: 100`
- `stop_epoch: 71`
- `num_workers: 16`
- `prefetch_factor: 4`

### UA-DETRAC configs

- `num_queries: 300`
- `batch_size: 16`
- `epochs: 50`
- `stop_epoch: 71`
- `num_workers: 16`
- `prefetch_factor: 4`

## 6. Evaluation and test behavior

The trainer performs:

1. scheduled validation during training
2. best-model evaluation on validation after training
3. optional best-model evaluation on test after training

The test evaluation path is only used when a valid test split can be built.
If no usable test data is available, the trainer logs that test evaluation is
skipped.

Common utility used for post-training eval:

- `experiments/common/detr_eval_utils.py`

## 7. Inference and visualization

Relevant files:

- `experiments/cas_detr/batch_inference.py`
- `experiments/cas_detr/run_inference.sh`
- `experiments/cas_detr/visualize_sparsity.py`
- `experiments/cas_detr/run_visualize_sparsity.sh`

The training script also saves:

- `inference_samples/`
- token importance heatmaps under `visualizations/`

## 8. Known caveats in the current implementation

### 8.1 `use_subpixel_offset` is present in YAML but not fully wired

The configs include:

- `model.cas_detr.use_subpixel_offset`

But the current `HybridEncoder` construction path does not explicitly forward
this flag into the pruner builder. So this YAML option should not be treated as
fully configurable unless the code is updated.

### 8.2 `batch_inference.py` is not fully aligned with the current trainer naming

The main training class is:

- `CaS_DETRTrainer`

If a helper script still imports `RTDETRTrainer` from the CaS-DETR experiment
folder, that path is outdated and should be treated carefully.

### 8.3 This README describes the training path in `train.py`

There are other prototype or legacy modules in the repo, but this document is
only meant to describe the code that is actually used by:

- `experiments/cas_detr/train.py`

## 9. Summary

The current CaS-DETR implementation in this repo is best described as:

- an RT-DETR-style detector
- with token pruning in the encoder path
- with CASS supervision for token importance
- with MoE replacing the decoder FFN
- trained through a unified trainer defined directly in `experiments/cas_detr/train.py`

