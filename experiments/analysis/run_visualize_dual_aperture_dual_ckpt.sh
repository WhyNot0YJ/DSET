#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash experiments/analysis/run_visualize_dual_aperture_dual_ckpt.sh
# Override any variable by exporting it before running, e.g.
#   RESUME_A=... RESUME_B=... IMG_ROW_1=... bash ...

ROOT_DIR="/root/autodl-tmp/CaS_DETR"
cd "${ROOT_DIR}"

# Model A: DAIR-V2X (rows 1-2)
CONFIG_A="${CONFIG_A:-experiments/CaS-DETR/configs/dataset/ablation/cas_deim_moe4_cass_caip_base03_a10_hgnetv2_s_dairv2x.yml}"
RESUME_A="${RESUME_A:-experiments/CaS-DETR/outputs/ablation/cas_deim_moe4_cass_caip_base03_a10_hgnetv2_s_dairv2x/best_stg2.pth}"

# Model B: UA-DETRAC (rows 3-4)
CONFIG_B="${CONFIG_B:-experiments/CaS-DETR/configs/dataset/ablation/cas_deim_moe4_cass_caip_base05_a10_hgnetv2_s_uadetrac.yml}"
RESUME_B="${RESUME_B:-experiments/CaS-DETR/outputs/ablation/base05_a10/cas_deim_moe4_cass_caip_base05_a10_hgnetv2_s_uadetrac/best_stg2.pth}"

DEVICE="${DEVICE:-cuda}"
EVAL_EPOCH_A="${EVAL_EPOCH_A:-5}"
EVAL_EPOCH_B="${EVAL_EPOCH_B:-5}"
CONF_THRESHOLD="${CONF_THRESHOLD:-0.3}"
OUTPUT_PDF="${OUTPUT_PDF:-experiments/analysis/figure5_qualitative_cas_detr.pdf}"

# Image row order (editable):
# Row1/Row2 use model A; Row3/Row4 use model B.
IMG_ROW_1="${IMG_ROW_1:-/root/autodl-fs/datasets/DAIR-V2X/image/000056.jpg}"
IMG_ROW_2="${IMG_ROW_2:-/root/autodl-fs/datasets/DAIR-V2X/image/000032.jpg}"
IMG_ROW_3="${IMG_ROW_3:-/root/autodl-fs/datasets/UA-DETRAC_COCO/test/4254.jpg}"
IMG_ROW_4="${IMG_ROW_4:-/root/autodl-fs/datasets/UA-DETRAC_COCO/test/6446.jpg}"

for f in "${CONFIG_A}" "${RESUME_A}" "${CONFIG_B}" "${RESUME_B}" "${IMG_ROW_1}" "${IMG_ROW_2}" "${IMG_ROW_3}" "${IMG_ROW_4}"; do
  if [[ ! -f "${f}" ]]; then
    echo "Missing file: ${f}"
    echo "Please update variables in this script or pass them as env vars."
    exit 1
  fi
done

python experiments/analysis/visualize_dual_aperture_cas_detr.py \
  -c "${CONFIG_A}" \
  -r "${RESUME_A}" \
  --config_b "${CONFIG_B}" \
  --resume_b "${RESUME_B}" \
  --split_index 2 \
  --images "${IMG_ROW_1}" "${IMG_ROW_2}" "${IMG_ROW_3}" "${IMG_ROW_4}" \
  --output "${OUTPUT_PDF}" \
  --device "${DEVICE}" \
  --conf_threshold "${CONF_THRESHOLD}" \
  --eval_epoch "${EVAL_EPOCH_A}" \
  --eval_epoch_b "${EVAL_EPOCH_B}"

echo "Saved figure: ${OUTPUT_PDF}"
