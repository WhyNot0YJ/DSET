#!/usr/bin/env bash
set -euo pipefail

# Build candidate gallery PDFs for Figure 5 case selection.
# Produces two multi-page PDFs (DAIR-V2X and UA-DETRAC), each showing 100
# randomly sampled test-set images with columns:
#   [Original | Importance Map S5 | Ours mask+pred | Baseline pred]
# Scan the PDFs, pick the good case ids, then plug their file paths back into
# run_visualize_dual_aperture_dual_ckpt.sh (IMG_ROW_2 / IMG_ROW_3).
#
# Usage:
#   bash experiments/analysis/run_build_candidate_gallery.sh
# Override env vars before calling to change paths.

ROOT_DIR="/root/autodl-tmp/CaS_DETR"
cd "${ROOT_DIR}"

# --- Ours (CaS-DETR full) checkpoints, same as the 4x4 script ---
CONFIG_A="${CONFIG_A:-experiments/CaS-DETR/configs/dataset/ablation/cas_deim_moe4_cass_caip_base03_a10_hgnetv2_s_dairv2x.yml}"
RESUME_A="${RESUME_A:-experiments/CaS-DETR/outputs/ablation/cas_deim_moe4_cass_caip_base03_a10_hgnetv2_s_dairv2x/best_stg2.pth}"

CONFIG_B="${CONFIG_B:-experiments/CaS-DETR/configs/dataset/ablation/cas_deim_moe4_cass_caip_base05_a10_hgnetv2_s_uadetrac.yml}"
RESUME_B="${RESUME_B:-experiments/CaS-DETR/outputs/ablation/base05_a10/cas_deim_moe4_cass_caip_base05_a10_hgnetv2_s_uadetrac/best_stg2.pth}"

# --- Baseline (all-off) checkpoints ---
BASELINE_CONFIG_A="${BASELINE_CONFIG_A:-experiments/CaS-DETR/configs/dataset/ablation/cas_deim_all_off_hgnetv2_s_dairv2x.yml}"
BASELINE_RESUME_A="${BASELINE_RESUME_A:-experiments/CaS-DETR/outputs/ablation/cas_deim_all_off_hgnetv2_s_dairv2x/best_stg2.pth}"

BASELINE_CONFIG_B="${BASELINE_CONFIG_B:-experiments/CaS-DETR/configs/dataset/ablation/cas_deim_all_off_hgnetv2_s_uadetrac.yml}"
BASELINE_RESUME_B="${BASELINE_RESUME_B:-experiments/CaS-DETR/outputs/ablation/cas_deim_all_off_hgnetv2_s_uadetrac/best_stg2.pth}"

# --- Test set annotation + image root ---
# DAIR-V2X test json: file_name like 'image/000056.jpg'; image_root should
# contain 'image/' as a subdir, so strip_prefix is empty and image_root is the
# dataset base.
DAIRV2X_ANN="${DAIRV2X_ANN:-/root/autodl-fs/datasets/DAIR-V2X/instances_test.json}"
DAIRV2X_IMG_ROOT="${DAIRV2X_IMG_ROOT:-/root/autodl-fs/datasets/DAIR-V2X}"
DAIRV2X_STRIP_PREFIX="${DAIRV2X_STRIP_PREFIX:-}"

# UA-DETRAC test json: file_name like '2546.jpg'; actual images live at
# /root/autodl-fs/datasets/UA-DETRAC_COCO/test/2546.jpg, so image_root must be
# the test/ folder.
UADETRAC_ANN="${UADETRAC_ANN:-/root/autodl-fs/datasets/UA-DETRAC_COCO/annotations/instances_test.json}"
UADETRAC_IMG_ROOT="${UADETRAC_IMG_ROOT:-/root/autodl-fs/datasets/UA-DETRAC_COCO/test}"
UADETRAC_STRIP_PREFIX="${UADETRAC_STRIP_PREFIX:-}"

# --- Sampling + inference options ---
NUM_SAMPLES="${NUM_SAMPLES:-100}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda}"
EVAL_EPOCH_A="${EVAL_EPOCH_A:-5}"
EVAL_EPOCH_B="${EVAL_EPOCH_B:-5}"
BASELINE_EVAL_EPOCH_A="${BASELINE_EVAL_EPOCH_A:-${EVAL_EPOCH_A}}"
BASELINE_EVAL_EPOCH_B="${BASELINE_EVAL_EPOCH_B:-${EVAL_EPOCH_B}}"
CONF_THRESHOLD="${CONF_THRESHOLD:-0.3}"

# --- PDF layout ---
ROWS_PER_PAGE="${ROWS_PER_PAGE:-8}"
FIG_WIDTH="${FIG_WIDTH:-14}"
ROW_HEIGHT="${ROW_HEIGHT:-1.9}"
DPI="${DPI:-120}"

OUT_DIR="${OUT_DIR:-experiments/analysis/gallery}"
mkdir -p "${OUT_DIR}"

OUTPUT_DAIRV2X="${OUTPUT_DAIRV2X:-${OUT_DIR}/gallery_dairv2x.pdf}"
INDEX_DAIRV2X="${INDEX_DAIRV2X:-${OUT_DIR}/gallery_dairv2x_index.json}"
OUTPUT_UADETRAC="${OUTPUT_UADETRAC:-${OUT_DIR}/gallery_uadetrac.pdf}"
INDEX_UADETRAC="${INDEX_UADETRAC:-${OUT_DIR}/gallery_uadetrac_index.json}"

# --- Sanity check files exist ---
for f in "${CONFIG_A}" "${RESUME_A}" "${CONFIG_B}" "${RESUME_B}" \
         "${BASELINE_CONFIG_A}" "${BASELINE_RESUME_A}" \
         "${BASELINE_CONFIG_B}" "${BASELINE_RESUME_B}" \
         "${DAIRV2X_ANN}" "${UADETRAC_ANN}"; do
  if [[ ! -f "${f}" ]]; then
    echo "Missing file: ${f}"
    exit 1
  fi
done
for d in "${DAIRV2X_IMG_ROOT}" "${UADETRAC_IMG_ROOT}"; do
  if [[ ! -d "${d}" ]]; then
    echo "Missing image root dir: ${d}"
    exit 1
  fi
done

run_gallery() {
  local tag="$1"
  local cfg="$2"
  local ckpt="$3"
  local bcfg="$4"
  local bckpt="$5"
  local eep="$6"
  local beep="$7"
  local ann="$8"
  local img_root="$9"
  local strip_prefix="${10}"
  local out_pdf="${11}"
  local out_idx="${12}"

  echo ""
  echo "=========================================================="
  echo "Building gallery: ${tag}"
  echo "  config:        ${cfg}"
  echo "  resume:        ${ckpt}"
  echo "  baseline cfg:  ${bcfg}"
  echo "  baseline ckpt: ${bckpt}"
  echo "  ann_json:      ${ann}"
  echo "  image_root:    ${img_root}"
  echo "  samples:       ${NUM_SAMPLES}  (seed=${SEED})"
  echo "  output:        ${out_pdf}"
  echo "=========================================================="

  local args=(
    experiments/analysis/build_candidate_gallery.py
    -c "${cfg}"
    -r "${ckpt}"
    --baseline_config "${bcfg}"
    --baseline_resume "${bckpt}"
    --ann_json "${ann}"
    --image_root "${img_root}"
    --num_samples "${NUM_SAMPLES}"
    --seed "${SEED}"
    --device "${DEVICE}"
    --eval_epoch "${eep}"
    --baseline_eval_epoch "${beep}"
    --conf_threshold "${CONF_THRESHOLD}"
    --output "${out_pdf}"
    --index_json "${out_idx}"
    --rows_per_page "${ROWS_PER_PAGE}"
    --fig_width "${FIG_WIDTH}"
    --row_height "${ROW_HEIGHT}"
    --dpi "${DPI}"
  )
  if [[ -n "${strip_prefix}" ]]; then
    args+=(--strip_prefix "${strip_prefix}")
  fi

  python3 "${args[@]}"
}

run_gallery \
  "DAIR-V2X" \
  "${CONFIG_A}" "${RESUME_A}" \
  "${BASELINE_CONFIG_A}" "${BASELINE_RESUME_A}" \
  "${EVAL_EPOCH_A}" "${BASELINE_EVAL_EPOCH_A}" \
  "${DAIRV2X_ANN}" "${DAIRV2X_IMG_ROOT}" "${DAIRV2X_STRIP_PREFIX}" \
  "${OUTPUT_DAIRV2X}" "${INDEX_DAIRV2X}"

run_gallery \
  "UA-DETRAC" \
  "${CONFIG_B}" "${RESUME_B}" \
  "${BASELINE_CONFIG_B}" "${BASELINE_RESUME_B}" \
  "${EVAL_EPOCH_B}" "${BASELINE_EVAL_EPOCH_B}" \
  "${UADETRAC_ANN}" "${UADETRAC_IMG_ROOT}" "${UADETRAC_STRIP_PREFIX}" \
  "${OUTPUT_UADETRAC}" "${INDEX_UADETRAC}"

echo ""
echo "Done."
echo "  DAIR-V2X gallery: ${OUTPUT_DAIRV2X}"
echo "  UA-DETRAC gallery: ${OUTPUT_UADETRAC}"
echo ""
echo "Next step: scan the PDFs, pick good cases, then update IMG_ROW_2 and"
echo "IMG_ROW_3 in experiments/analysis/run_visualize_dual_aperture_dual_ckpt.sh"
