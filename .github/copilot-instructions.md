# DSET Project — Copilot Instructions

## Project Overview

Research project benchmarking object detection models on the **DAIR-V2X** (车路协同) dataset. The core contribution is **DSET (Dual-Sparse Expert Transformer)** — an RT-DETR variant that combines **token pruning** (discard background patches) and **Mixture-of-Experts** (route tokens to expert subsets) for efficient detection.

**8 detection classes**: Car, Truck, Van, Bus, Pedestrian, Cyclist, Motorcyclist, Trafficcone.

## Repository Layout

- `experiments/dset/` — **Primary DSET implementation** (model, training, inference)
  - `src/zoo/rtdetr/` — Core architecture: `hybrid_encoder.py`, `moe_components.py`, `token_level_pruning.py`, `rtdetrv2_decoder.py`
  - `src/nn/backbone/` — Backbones (PResNet, HGNetv2, CSPResNet)
  - `src/data/dataset/dairv2x_detection.py` — DAIR-V2X dataset loader (COCO format)
  - `configs/` — YAML configs named `dset{experts}_r{backbone}_ratio{ratio}.yaml`
  - `train.py` — `DSETTrainer` class: full training/validation pipeline with EMA, AMP, early stopping
- `experiments/moe-rtdetr/`, `experiments/rt-detr/` — RT-DETR baselines (MoE-only and vanilla)
- `experiments/deformable-detr/`, `experiments/yolov8/`, `experiments/yolov10/` — Comparison models
- `experiments/analysis/` — Benchmarking, Pareto plots, report generation
- `dair2coco.py` — Converts DAIR-V2X raw annotations to COCO JSON format
- `logs/` — Timestamped experiment outputs: `dset6_r18_YYYYMMDD_HHMMSS/`

## Key Architecture Patterns

### Dual-Sparse Pipeline (in `hybrid_encoder.py`)
```
Backbone features [P3, P4, P5]
  → TokenLevelPruner on P5 (keep top K% by importance score)
  → Shared MoE Transformer Encoder (single MoELayer instance reused across layers)
  → CCFF cross-scale FPN fusion
  → MoE Decoder (deformable cross-attention + MoE FFN per layer)
  → Detection heads
```

### MoE Implementation (`moe_components.py`)
- **Vectorized**: loops over experts (not tokens) using `torch.where()` + `index_add_()` — no Python loops over tokens.
- Router produces top-K expert assignments; balance loss prevents expert collapse (Switch Transformer style).

### Token Pruning (`token_level_pruning.py`)
- `LinearImportancePredictor` scores each token; top-K% kept.
- **CASS (Context-Aware Soft Supervision)**: generates soft importance targets from GT bboxes with Gaussian/linear decay, trained via Varifocal Loss.

### Loss Composition
```
L_total = L_detection(VFL + 5·L1_bbox + 2·GIoU)
        + λ_decoder · L_moe_balance
        + λ_encoder · L_moe_balance
        + λ_cass · L_cass
```
MoE balance loss has configurable warmup epochs (`moe_balance_warmup_epochs`).

## Configuration Conventions

YAML configs live in `experiments/dset/configs/`. Key tunable parameters:
- `dset.token_keep_ratio`: {0.3, 0.5, 0.7, 0.9} — pruning aggressiveness
- `model.num_experts` / `encoder_moe_top_k` — MoE width and routing
- `training.early_stopping_patience`: 20 epochs default
- `data.data_root`: points to DAIR-V2X dataset (default `/root/autodl-tmp/datasets/DAIR-V2X`)

Naming convention: `dset{num_experts}_r{18|34}_ratio{keep_ratio}.yaml`

## Common Workflows

```bash
# Train DSET (default config or specify)
./experiments/dset/run_training.sh
./experiments/dset/run_training.sh configs/dset6_r18_ratio0.5.yaml

# Resume training
./experiments/dset/run_training.sh configs/dset6_r34.yaml --resume AUTO

# Batch experiments (all model variants)
./experiments/run_batch_experiments.sh
./experiments/run_batch_experiments.sh --test       # Quick 2-epoch test
./experiments/run_batch_experiments.sh --dset --r18  # Only DSET R18 variants

# Inference
./experiments/dset/run_inference.sh --config configs/dset6_r18.yaml \
  --checkpoint logs/best_model.pth --conf 0.5

# Benchmarking & analysis
python experiments/analysis/generate_benchmark_table.py --model_type dset
python experiments/analysis/plot_efficiency_pareto.py
```

## Coding Conventions

- **Checkpoints**: `logs/{experiment}/best_model.pth` and `latest_checkpoint.pth`; training history in `training_history.csv`
- **Box format**: internally normalized `(cx, cy, w, h)` ∈ [0,1], converted to `(x1, y1, x2, y2)` for processing
- **Backbone channel dims**: S3=128, S4=256, S5=512 (PResNet); configs specify `in_channels: [128, 256, 512]`
- **Evaluation**: pycocotools COCOeval (mAP@0.5, @0.75, @0.5:0.95); validation runs with pruning disabled (all tokens kept)
- **Dependencies**: PyTorch ≥2.5, torchvision ≥0.20, `faster-coco-eval`, `pycocotools`, `thop` for FLOPs
- **Reproducibility**: `experiments/seed_utils.py` for deterministic seeding across experiments
