#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash experiments/analysis/run_visualize_dual_aperture_dual_ckpt.sh
# Override any variable by exporting it before running, e.g.
#   RESUME_A=... RESUME_B=... BASELINE_RESUME_A=... BASELINE_RESUME_B=... IMG_ROW_1=... bash ...
# Smaller PDF: defaults use dpi 240 and 14×8.8 in; optional COMPACT=1 or PDF_SLIM_FONTS=1.
# Baseline FN from CaS is on by default; MARK_BASELINE_FAILURE_FROM_CAS=0 turns it off.

ROOT_DIR="/root/autodl-tmp/CaS_DETR"
cd "${ROOT_DIR}"

# Model A: DAIR-V2X (rows 1-2)
CONFIG_A="${CONFIG_A:-experiments/CaS-DETR/configs/dataset/ablation/cas_deim_moe4_cass_caip_base03_a10_hgnetv2_s_dairv2x.yml}"
RESUME_A="${RESUME_A:-experiments/CaS-DETR/outputs/ablation/cas_deim_moe4_cass_caip_base03_a10_hgnetv2_s_dairv2x/best_stg2.pth}"

# Model B: UA-DETRAC (rows 3-4)
CONFIG_B="${CONFIG_B:-experiments/CaS-DETR/configs/dataset/ablation/cas_deim_moe4_cass_caip_base05_a10_hgnetv2_s_uadetrac.yml}"
RESUME_B="${RESUME_B:-experiments/CaS-DETR/outputs/ablation/base05_a10/cas_deim_moe4_cass_caip_base05_a10_hgnetv2_s_uadetrac/best_stg2.pth}"

# Baseline models for the last column.
BASELINE_CONFIG_A="${BASELINE_CONFIG_A:-experiments/CaS-DETR/configs/dataset/ablation/cas_deim_all_off_hgnetv2_s_dairv2x.yml}"
BASELINE_RESUME_A="${BASELINE_RESUME_A:-experiments/CaS-DETR/outputs/ablation/cas_deim_all_off_hgnetv2_s_dairv2x/best_stg2.pth}"
BASELINE_CONFIG_B="${BASELINE_CONFIG_B:-experiments/CaS-DETR/configs/dataset/ablation/cas_deim_all_off_hgnetv2_s_uadetrac.yml}"
BASELINE_RESUME_B="${BASELINE_RESUME_B:-experiments/CaS-DETR/outputs/ablation/cas_deim_all_off_hgnetv2_s_uadetrac/best_stg2.pth}"

DEVICE="${DEVICE:-cuda}"
EVAL_EPOCH_A="${EVAL_EPOCH_A:-5}"
EVAL_EPOCH_B="${EVAL_EPOCH_B:-5}"
BASELINE_EVAL_EPOCH_A="${BASELINE_EVAL_EPOCH_A:-${EVAL_EPOCH_A}}"
BASELINE_EVAL_EPOCH_B="${BASELINE_EVAL_EPOCH_B:-${EVAL_EPOCH_B}}"
CONF_THRESHOLD="${CONF_THRESHOLD:-0.3}"
# Defaults match visualize_dual_aperture_cas_detr.py defaults and --compact dpi for smaller PDFs.
SAVE_DPI="${SAVE_DPI:-240}"
FIG_WIDTH="${FIG_WIDTH:-14}"
FIG_HEIGHT="${FIG_HEIGHT:-8.8}"
# Set COMPACT=1 to pass --compact so Python also forces PNG zlib 9 and the same dpi or fig overrides.
COMPACT="${COMPACT:-0}"
PDF_SLIM_FONTS="${PDF_SLIM_FONTS:-0}"
# Default 1: red FN on baseline from CaS boxes; 0 disables.
MARK_BASELINE_FAILURE_FROM_CAS="${MARK_BASELINE_FAILURE_FROM_CAS:-1}"
OUTPUT_PATH="${OUTPUT_PATH:-experiments/analysis/figure5_qualitative_cas_detr.pdf}"

# Image row order (editable):
# Row1/Row2 use model A; Row3/Row4 use model B.
IMG_ROW_1="${IMG_ROW_1:-/root/autodl-fs/datasets/DAIR-V2X/image/004258.jpg}"
IMG_ROW_2="${IMG_ROW_2:-/root/autodl-fs/datasets/DAIR-V2X/image/007135.jpg}"
IMG_ROW_3="${IMG_ROW_3:-/root/autodl-fs/datasets/UA-DETRAC_COCO/test/2604.jpg}"
IMG_ROW_4="${IMG_ROW_4:-/root/autodl-fs/datasets/UA-DETRAC_COCO/test/3963.jpg}"

for f in "${CONFIG_A}" "${RESUME_A}" "${CONFIG_B}" "${RESUME_B}" "${BASELINE_CONFIG_A}" "${BASELINE_CONFIG_B}" "${IMG_ROW_1}" "${IMG_ROW_2}" "${IMG_ROW_3}" "${IMG_ROW_4}"; do
  if [[ ! -f "${f}" ]]; then
    echo "Missing file: ${f}"
    echo "Please update variables in this script or pass them as env vars."
    exit 1
  fi
done

for f in "${BASELINE_RESUME_A}" "${BASELINE_RESUME_B}"; do
  if [[ -n "${f}" && ! -f "${f}" ]]; then
    echo "Missing baseline checkpoint: ${f}"
    echo "Please update BASELINE_RESUME_A/BASELINE_RESUME_B or unset them."
    exit 1
  fi
done

PY_ARGS=(
  experiments/analysis/visualize_dual_aperture_cas_detr.py
  -c "${CONFIG_A}"
  -r "${RESUME_A}"
  --config_b "${CONFIG_B}"
  --resume_b "${RESUME_B}"
  --split_index 2
  --images "${IMG_ROW_1}" "${IMG_ROW_2}" "${IMG_ROW_3}" "${IMG_ROW_4}"
  --output "${OUTPUT_PATH}"
  --device "${DEVICE}"
  --conf_threshold "${CONF_THRESHOLD}"
  --eval_epoch "${EVAL_EPOCH_A}"
  --eval_epoch_b "${EVAL_EPOCH_B}"
  --dpi "${SAVE_DPI}"
  --fig_width "${FIG_WIDTH}"
  --fig_height "${FIG_HEIGHT}"
)

if [[ -n "${BASELINE_RESUME_A}" ]]; then
  PY_ARGS+=(
    --baseline_config "${BASELINE_CONFIG_A}"
    --baseline_resume "${BASELINE_RESUME_A}"
    --baseline_eval_epoch "${BASELINE_EVAL_EPOCH_A}"
  )
fi

if [[ -n "${BASELINE_RESUME_B}" ]]; then
  PY_ARGS+=(
    --baseline_config_b "${BASELINE_CONFIG_B}"
    --baseline_resume_b "${BASELINE_RESUME_B}"
    --baseline_eval_epoch_b "${BASELINE_EVAL_EPOCH_B}"
  )
fi

if [[ "${COMPACT}" == "1" ]]; then
  PY_ARGS+=(--compact)
fi
if [[ "${PDF_SLIM_FONTS}" == "1" ]]; then
  PY_ARGS+=(--pdf-slim-fonts)
fi

if [[ "${MARK_BASELINE_FAILURE_FROM_CAS}" == "0" ]]; then
  PY_ARGS+=(--no-mark-baseline-failure-from-cas)
fi

python3 "${PY_ARGS[@]}"

echo "Saved figure: ${OUTPUT_PATH}"
