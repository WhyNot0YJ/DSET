#!/usr/bin/env bash
set -euo pipefail

# Quick launcher for evaluate_coco_complexity_subsets.py
#
# Usage:
#   bash experiments/analysis/run_evaluate_coco_complexity_subsets.sh
#
# Override any variable by exporting it before running, e.g.
#   RESUME_DYNAMIC=/path/to.pth bash experiments/analysis/run_evaluate_coco_complexity_subsets.sh
#
# Notes:
# - 仅支持在线评估：需要四个 YAML 与四个 checkpoint。
# - Default GT is DAIR-V2X test split; image root defaults to dataset root next to annotations.
# - Fixed 1.0 defaults to moe4_only, no CAIP or CASS, token_keep_ratio 1.0. Override if your
#   Fixed 1.0 checkpoint uses another YAML.

ROOT_DIR="${ROOT_DIR:-/root/autodl-tmp/CaS_DETR}"
cd "${ROOT_DIR}"

DEVICE="${DEVICE:-cuda:0}"
PYTHON="${PYTHON:-python3}"

GT_JSON="${GT_JSON:-/root/autodl-fs/datasets/DAIR-V2X/annotations/instances_test.json}"
TEST_IMG_FOLDER="${TEST_IMG_FOLDER:-/root/autodl-fs/datasets/DAIR-V2X}"

ENCODER_EPOCH="${ENCODER_EPOCH:--1}"

# Optional: shared YAML 更新项，空格分隔，会展开为 --online-update 的多个参数
# 例：ONLINE_UPDATE='HybridEncoder.foo=1 HybridEncoder.bar=2'
ONLINE_UPDATE="${ONLINE_UPDATE:-}"

# Dynamic keep-ratio fallback JSON, optional
DYNAMIC_KEEP_FALLBACK_JSON="${DYNAMIC_KEEP_FALLBACK_JSON:-}"

# Fixed keep constants for table, should match your fixed experiments
FIXED_KEEP_03="${FIXED_KEEP_03:-0.3}"
FIXED_KEEP_07="${FIXED_KEEP_07:-0.7}"
FIXED_KEEP_10="${FIXED_KEEP_10:-1.0}"

# --- configs and checkpoints, defaults aligned with run_visualize_dual_aperture_dual_ckpt.sh for dynamic ---
CONFIG_FIXED_03="${CONFIG_FIXED_03:-experiments/CaS-DETR/configs/dataset/ablation/cas_deim_moe4_cass_caip_base03_a10_keep03_fixed_hgnetv2_s_dairv2x.yml}"
RESUME_FIXED_03="${RESUME_FIXED_03:-experiments/CaS-DETR/outputs/ablation/cas_deim_moe4_cass_caip_base03_a10_keep03_fixed_hgnetv2_s_dairv2x/best_stg2.pth}"

CONFIG_FIXED_07="${CONFIG_FIXED_07:-experiments/CaS-DETR/configs/dataset/ablation/cas_deim_moe4_cass_caip_base03_a10_keep07_fixed_hgnetv2_s_dairv2x.yml}"
RESUME_FIXED_07="${RESUME_FIXED_07:-experiments/CaS-DETR/outputs/ablation/cas_deim_moe4_cass_caip_base03_a10_keep07_fixed_hgnetv2_s_dairv2x/best_stg2.pth}"

# Fixed 1.0: repository baseline with full keep ratio, no CAIP or CASS
CONFIG_FIXED_10="${CONFIG_FIXED_10:-experiments/CaS-DETR/configs/dataset/ablation/cas_deim_moe4_only_hgnetv2_s_dairv2x.yml}"
RESUME_FIXED_10="${RESUME_FIXED_10:-experiments/CaS-DETR/outputs/ablation/cas_deim_moe4_only_hgnetv2_s_dairv2x/best_stg2.pth}"

CONFIG_DYNAMIC="${CONFIG_DYNAMIC:-experiments/CaS-DETR/configs/dataset/ablation/cas_deim_moe4_cass_caip_base03_a10_hgnetv2_s_dairv2x.yml}"
RESUME_DYNAMIC="${RESUME_DYNAMIC:-experiments/CaS-DETR/outputs/ablation/cas_deim_moe4_cass_caip_base03_a10_hgnetv2_s_dairv2x/best_stg2.pth}"

if [[ ! -f "${GT_JSON}" ]]; then
  echo "Missing GT: ${GT_JSON}"
  exit 1
fi

for f in \
  "${CONFIG_FIXED_03}" "${RESUME_FIXED_03}" \
  "${CONFIG_FIXED_07}" "${RESUME_FIXED_07}" \
  "${CONFIG_FIXED_10}" "${RESUME_FIXED_10}" \
  "${CONFIG_DYNAMIC}" "${RESUME_DYNAMIC}"; do
  if [[ ! -f "${f}" ]]; then
    echo "Missing file: ${f}"
    echo "Update variables in this script or pass them as env vars."
    exit 1
  fi
done

CMD=(
  "${PYTHON}" experiments/analysis/evaluate_coco_complexity_subsets.py
  --gt "${GT_JSON}"
  --test-img-folder "${TEST_IMG_FOLDER}"
  --device "${DEVICE}"
  --encoder-epoch "${ENCODER_EPOCH}"
  --online-fixed-03-config "${CONFIG_FIXED_03}"
  --online-fixed-03-resume "${RESUME_FIXED_03}"
  --online-fixed-07-config "${CONFIG_FIXED_07}"
  --online-fixed-07-resume "${RESUME_FIXED_07}"
  --online-fixed-10-config "${CONFIG_FIXED_10}"
  --online-fixed-10-resume "${RESUME_FIXED_10}"
  --online-dynamic-config "${CONFIG_DYNAMIC}"
  --online-dynamic-resume "${RESUME_DYNAMIC}"
  --fixed-keep-03 "${FIXED_KEEP_03}"
  --fixed-keep-07 "${FIXED_KEEP_07}"
  --fixed-keep-10 "${FIXED_KEEP_10}"
)

if [[ -n "${ONLINE_UPDATE}" ]]; then
  # shellcheck disable=2206
  CMD+=(--online-update ${ONLINE_UPDATE})
fi

if [[ -n "${DYNAMIC_KEEP_FALLBACK_JSON}" ]]; then
  if [[ ! -f "${DYNAMIC_KEEP_FALLBACK_JSON}" ]]; then
    echo "DYNAMIC_KEEP_FALLBACK_JSON is set but file missing: ${DYNAMIC_KEEP_FALLBACK_JSON}"
    exit 1
  fi
  CMD+=(--dynamic-keep-fallback-json "${DYNAMIC_KEEP_FALLBACK_JSON}")
fi

"${CMD[@]}"
