#!/bin/bash

# 解决 32G 环境下 Dataloader 多进程与 OpenMP 线程池冲突导致的 libgomp 报错
export OMP_NUM_THREADS=1

################################################################################
# 批量实验运行脚本 - 方案2：鲁棒批量运行
# 
# 功能：
#   - 顺序运行多个配置
#   - 失败自动继续下一个
#   - 详细日志记录
#   - 实验进度追踪
#   - 完成后生成汇总报告
#
# 使用方法：
#   ./run_batch_experiments.sh                                 # 运行所有实验（完整epochs）
#   ./run_batch_experiments.sh --test                          # 测试模式：运行所有配置，每个只跑2个epoch
#   ./run_batch_experiments.sh --rtdetrv2                      # 官方 RT-DETR v2（rtdetrv2_pytorch + train_adapter.py，默认 --cas-eval）
#   ./run_batch_experiments.sh --rt-detr                       # 与 --rtdetrv2 相同（v1 已移除，仅保留 v2）
#   DAIRV2X_DATA_ROOT=/path UADETRAC_DATA_ROOT=/path ./run_batch_experiments.sh --yes --rtdetrv2
#   RTDETR_TUNING_CKPT=/path/to.pth ./run_batch_experiments.sh --yes --rtdetrv2
#   ./run_batch_experiments.sh --dairv2x --rtdetrv2          # 仅 DAIR-V2X 的 RT-DETR v2
#   ./run_batch_experiments.sh --cas_detr                      # 只运行新的 CaS-DETR 第一阶段消融实验（5个配置）
#   ./run_batch_experiments.sh --yolov5                        # 只运行YOLOv5实验
#   ./run_batch_experiments.sh --yolov8                        # 只运行YOLOv8实验
#   ./run_batch_experiments.sh --yolov12                       # 只运行YOLOv12实验
#   ./run_batch_experiments.sh --yolox                         # 只运行 YOLOX（Megvii）实验
#   ./run_batch_experiments.sh --yolo                          # 一键：YOLOv5+v8+v12+YOLOX（常与 --n/--s/--m 组合）
#   ./run_batch_experiments.sh --yolo --s                      # 仅 s 规模（两数据集全跑）
#   ./run_batch_experiments.sh --yes --test --yolo --m --dairv2x  # 测试模式仅 DAIR 的 m 规模 YOLO 全家桶
#   ./run_batch_experiments.sh --fasterrcnn                    # 只运行 torchvision Faster R-CNN（DAIR + UA-DETRAC）
#   ./run_batch_experiments.sh --deim                           # 只运行 DEIM-S（DAIR + UA-DETRAC）
#   ./run_batch_experiments.sh --dfine                          # 只运行 D-FINE-S（DAIR + UA-DETRAC）
#   ./run_batch_experiments.sh --deformable-detr               # 只运行Deformable-DETR实验
#   ./run_batch_experiments.sh --test --rt-detr                # 测试模式只跑 RT-DETR v2，等价 --rtdetrv2
#   ./run_batch_experiments.sh --test --rtdetrv2               # 测试模式只跑官方 RT-DETRv2（2 epoch + cas-eval）
#   ./run_batch_experiments.sh --test --cas_detr               # 测试模式只运行新的 CaS-DETR 第一阶段消融实验
#   ./run_batch_experiments.sh --test --yolov5                 # 测试模式只运行YOLOv5
#   ./run_batch_experiments.sh --test --yolov8                 # 测试模式只运行YOLOv8
#   ./run_batch_experiments.sh --test --yolov12                # 测试模式只运行YOLOv12
#   ./run_batch_experiments.sh --test --yolox                  # 测试模式只运行 YOLOX
#   ./run_batch_experiments.sh --test --fasterrcnn             # 测试模式只运行 Faster R-CNN
#   ./run_batch_experiments.sh --test --deformable-detr        # 测试模式只运行Deformable-DETR
#   ./run_batch_experiments.sh --r18                           # 只运行ResNet-18实验
#   ./run_batch_experiments.sh --n                             # 只运行所有 n 规模 YOLO（v5/v8/v12）
#   ./run_batch_experiments.sh --s                             # 只运行所有 s 规模 YOLO / YOLOX
#   ./run_batch_experiments.sh --m                             # 只运行所有 m 规模 YOLO / YOLOX
#   ./run_batch_experiments.sh --custom cfg1.yaml cfg2.yaml    # 自定义配置列表
#   ./run_batch_experiments.sh --keys rtdetrv2-r18-dairv2x moe6-r18-dairv2x   # 使用内置键名选择
#   ./run_batch_experiments.sh --dairv2x                       # 只保留 DAIR-V2X 维度（可叠 --rtdetrv2 / --cas_detr 等）
#   ./run_batch_experiments.sh --uadetrac                     # 只保留 UA-DETRAC 维度
#   ./run_batch_experiments.sh --dataset dairv2x --rtdetrv2    # 同上：--dataset 与 --dairv2x 等价；可与 --rtdetrv2 任意顺序组合
#   ./run_batch_experiments.sh --rtdetrv2 --dataset dairv2x     # 同上
#   ./run_batch_experiments.sh --dataset dairv2x --rt-detr      # --rt-detr 与 --rtdetrv2 相同
#   ./run_batch_experiments.sh --dataset dairv2x,uadetrac     # 同传 --dairv2x --uadetrac（两者都选则不按数据集筛）
#   ./run_batch_experiments.sh --select                        # 交互式选择待运行配置
#   ./run_batch_experiments.sh --rerun-failed [LOG_DIR]        # 自动选择上次失败的实验
#
# CaS_DETR / RT-DETR / MOE：输入为 letterbox 到 target_size（见各 YAML 中 augmentation.resize.letterbox_fill，一般 114）。
# 一键非交互（跳过确认）示例：
#   ./run_batch_experiments.sh --yes --cas_detr
#   ./run_batch_experiments.sh --yes --test --rtdetrv2 --cas_detr
################################################################################

set -e  # 遇到错误时不退出（我们会手动处理）

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 创建日志目录（使用绝对路径）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BATCH_LOG_DIR=""  # 禁用日志目录
# mkdir -p "$BATCH_LOG_DIR"

# 全局变量
TOTAL_EXPERIMENTS=0
SUCCESSFUL_EXPERIMENTS=0
FAILED_EXPERIMENTS=0
SKIPPED_EXPERIMENTS=0
TEST_MODE=false  # 测试模式标志
SKIP_CONFIRM=false  # --yes：跳过「是否开始」确认，便于一键 / CI
# --dairv2x / --uadetrac（或 --dataset）解析后写入；二者同时为 true 时不筛选
SCOPE_DAIRV2X=false
SCOPE_UADETRAC=false
SCOPE_SIZE_N=false
SCOPE_SIZE_S=false
SCOPE_SIZE_M=false
TOTAL_PLANNED_RUNS=0

# 选择可用的 Python 解释器
PYTHON_BIN=""
if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
else
    echo -e "${RED}[ERROR]${NC} 未找到可用的 Python 解释器（python/python3）"
    exit 1
fi

# 日志函数（仅输出到控制台，不写入文件）
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# 根据路径与 NAME_TO_PATH 键判断是否属于 dairv2x / uadetrac；want_* 为 true 时表示当前筛选需要该侧
_path_allowed_by_dataset_scope() {
    local path="$1"
    local want_dair="$2"
    local want_ua="$3"
    local is_ua=false
    local is_dair=false
    [[ "$path" == *uadetrac* ]] && is_ua=true
    [[ "$path" == *dairv2x* ]] && is_dair=true
    local cand
    for cand in "${!NAME_TO_PATH[@]}"; do
        [ "${NAME_TO_PATH[$cand]}" = "$path" ] || continue
        [[ "$cand" == *uadetrac* ]] && is_ua=true
        [[ "$cand" == *dairv2x* ]] && is_dair=true
    done
    if [ "$is_ua" = true ]; then
        if [ "$want_ua" = true ]; then return 0; else return 1; fi
    fi
    if [ "$is_dair" = true ]; then
        if [ "$want_dair" = true ]; then return 0; else return 1; fi
    fi
    if [ "$want_dair" = true ]; then return 0; else return 1; fi
}

apply_dataset_scope_filter_to_configs() {
    if [ "$SCOPE_DAIRV2X" != true ] && [ "$SCOPE_UADETRAC" != true ]; then
        return 0
    fi
    if [ "$SCOPE_DAIRV2X" = true ] && [ "$SCOPE_UADETRAC" = true ]; then
        log_info "数据集: 已同时指定 DAIR-V2X 与 UA-DETRAC，不按数据集筛配置"
        return 0
    fi
    local want_dair=false
    local want_ua=false
    [ "$SCOPE_DAIRV2X" = true ] && want_dair=true
    [ "$SCOPE_UADETRAC" = true ] && want_ua=true
    local before=${#CONFIGS_TO_RUN[@]}
    local filtered=()
    local p
    for p in "${CONFIGS_TO_RUN[@]}"; do
        if _path_allowed_by_dataset_scope "$p" "$want_dair" "$want_ua"; then
            filtered+=("$p")
        fi
    done
    CONFIGS_TO_RUN=("${filtered[@]}")
    if [ "$SCOPE_DAIRV2X" = true ] && [ "$SCOPE_UADETRAC" != true ]; then
        log_info "数据集筛选: 仅 DAIR-V2X（与 --rt-detr 等组合使用）"
    elif [ "$SCOPE_UADETRAC" = true ] && [ "$SCOPE_DAIRV2X" != true ]; then
        log_info "数据集筛选: 仅 UA-DETRAC"
    fi
    if [ "$before" -gt 0 ] && [ ${#CONFIGS_TO_RUN[@]} -eq 0 ]; then
        log_warning "按数据集筛选后队列为空，请检查与实验类型开关（如 --rt-detr）是否匹配"
    fi
}

_path_allowed_by_model_size_scope() {
    local path="$1"
    local want_n="$2"
    local want_s="$3"
    local want_m="$4"
    local base
    base=$(basename "$path")
    # 仅 YOLO / YOLOX 的 yaml 受 --n/--s/--m 筛选；其它（如 RT-DETRv2、CaS）不受规模开关影响
    if [[ "$base" != yolov* ]] && [[ "$base" != yolox* ]]; then
        return 0
    fi

    if [[ "$base" =~ ^yolov(5|8|12)n_.*\.yaml$ ]]; then
        [ "$want_n" = true ] && return 0 || return 1
    fi
    if [[ "$base" =~ ^(yolov(5|8|12)s_|yoloxs_).*.yaml$ ]]; then
        [ "$want_s" = true ] && return 0 || return 1
    fi
    if [[ "$base" =~ ^(yolov(5|8|12)m_|yoloxm_).*.yaml$ ]]; then
        [ "$want_m" = true ] && return 0 || return 1
    fi

    return 1
}

apply_model_size_filter_to_configs() {
    if [ "$SCOPE_SIZE_N" != true ] && [ "$SCOPE_SIZE_S" != true ] && [ "$SCOPE_SIZE_M" != true ]; then
        return 0
    fi
    local want_n=false
    local want_s=false
    local want_m=false
    [ "$SCOPE_SIZE_N" = true ] && want_n=true
    [ "$SCOPE_SIZE_S" = true ] && want_s=true
    [ "$SCOPE_SIZE_M" = true ] && want_m=true
    local before=${#CONFIGS_TO_RUN[@]}
    local filtered=()
    local p
    for p in "${CONFIGS_TO_RUN[@]}"; do
        if _path_allowed_by_model_size_scope "$p" "$want_n" "$want_s" "$want_m"; then
            filtered+=("$p")
        fi
    done
    CONFIGS_TO_RUN=("${filtered[@]}")

    local selected_sizes=()
    [ "$want_n" = true ] && selected_sizes+=("n")
    [ "$want_s" = true ] && selected_sizes+=("s")
    [ "$want_m" = true ] && selected_sizes+=("m")
    local size_str
    size_str=$(IFS='/'; echo "${selected_sizes[*]}")
    log_info "模型规模筛选: ${size_str}"

    if [ "$before" -gt 0 ] && [ ${#CONFIGS_TO_RUN[@]} -eq 0 ]; then
        log_warning "按模型规模筛选后队列为空，请使用 --yolo 或同时开启 --yolov5/--yolov8/--yolov12/--yolox，并检查 --n/--s/--m"
    fi
}

# 官方 RT-DETR v2（RT-DETR/rtdetrv2_pytorch/tools/train_adapter.py），训练后 CaS 风格评估见 --cas-eval。
# 路径格式：<yml 相对 experiments 的路径>@<dairv2x|uadetrac>；数据根可通过环境变量覆盖：
#   DAIRV2X_DATA_ROOT / UADETRAC_DATA_ROOT
# 整网 COCO 等预训练权重：RTDETR_TUNING_CKPT=/path/to.pth（或 train_adapter 的 -t）
# 关闭训练后 CaS 评估：RTDETRV2_CAS_EVAL=0
declare -A RTDETRV2_ADAPTER_CONFIGS=(
    ["rtdetrv2-r18-dairv2x"]="RT-DETR/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r18vd_100e_dairv2x.yml@dairv2x"
    ["rtdetrv2-r18-uadetrac"]="RT-DETR/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r18vd_100e_uadetrac.yml@uadetrac"
)

declare -a CORE_EXPERIMENTS=(
    "CaS-DETR/configs/dataset/ablation/cas_deim_cass_only_keep07_hgnetv2_s_dairv2x.yml"
)

declare -A CaS_DETR_CONFIGS=(
    ["casdeim-moe-only-dairv2x"]="CaS-DETR/configs/dataset/ablation/cas_deim_moe_only_hgnetv2_s_dairv2x.yml"
    ["casdeim-cass-only-keep07-dairv2x"]="CaS-DETR/configs/dataset/ablation/cas_deim_cass_only_keep07_hgnetv2_s_dairv2x.yml"
    ["casdeim-cass-only-keep05-dairv2x"]="CaS-DETR/configs/dataset/ablation/cas_deim_cass_only_keep05_hgnetv2_s_dairv2x.yml"
    ["casdeim-moe-cass-keep07-dairv2x"]="CaS-DETR/configs/dataset/ablation/cas_deim_moe_cass_keep07_hgnetv2_s_dairv2x.yml"
    ["casdeim-moe-cass-keep05-dairv2x"]="CaS-DETR/configs/dataset/ablation/cas_deim_moe_cass_keep05_hgnetv2_s_dairv2x.yml"
)

declare -A YOLOV5_CONFIGS=(
    ["yolov5n-dairv2x"]="yolo/configs/yolov5n_dairv2x.yaml"
    ["yolov5n-uadetrac"]="yolo/configs/yolov5n_uadetrac.yaml"
    ["yolov5s-dairv2x"]="yolo/configs/yolov5s_dairv2x.yaml"
    ["yolov5s-uadetrac"]="yolo/configs/yolov5s_uadetrac.yaml"
    ["yolov5m-dairv2x"]="yolo/configs/yolov5m_dairv2x.yaml"
    ["yolov5m-uadetrac"]="yolo/configs/yolov5m_uadetrac.yaml"
)

declare -A YOLOV8_CONFIGS=(
    ["yolov8n-dairv2x"]="yolo/configs/yolov8n_dairv2x.yaml"
    ["yolov8n-uadetrac"]="yolo/configs/yolov8n_uadetrac.yaml"
    ["yolov8s-dairv2x"]="yolo/configs/yolov8s_dairv2x.yaml"
    ["yolov8s-uadetrac"]="yolo/configs/yolov8s_uadetrac.yaml"
    ["yolov8m-dairv2x"]="yolo/configs/yolov8m_dairv2x.yaml"
    ["yolov8m-uadetrac"]="yolo/configs/yolov8m_uadetrac.yaml"
)

declare -A YOLOV12_CONFIGS=(
    ["yolov12n-dairv2x"]="yolo/configs/yolov12n_dairv2x.yaml"
    ["yolov12n-uadetrac"]="yolo/configs/yolov12n_uadetrac.yaml"
    ["yolov12s-dairv2x"]="yolo/configs/yolov12s_dairv2x.yaml"
    ["yolov12s-uadetrac"]="yolo/configs/yolov12s_uadetrac.yaml"
    ["yolov12m-dairv2x"]="yolo/configs/yolov12m_dairv2x.yaml"
    ["yolov12m-uadetrac"]="yolo/configs/yolov12m_uadetrac.yaml"
)

declare -A YOLOX_CONFIGS=(
    ["yoloxs-dairv2x"]="yolo/configs/yoloxs_dairv2x.yaml"
    ["yoloxs-uadetrac"]="yolo/configs/yoloxs_uadetrac.yaml"
    ["yoloxm-dairv2x"]="yolo/configs/yoloxm_dairv2x.yaml"
    ["yoloxm-uadetrac"]="yolo/configs/yoloxm_uadetrac.yaml"
)

declare -A FASTER_RCNN_CONFIGS=(
    ["fasterrcnn-resnet50-dairv2x"]="yolo/configs/fasterrcnn_resnet50_dairv2x.yaml"
    ["fasterrcnn-resnet50-uadetrac"]="yolo/configs/fasterrcnn_resnet50_uadetrac.yaml"
)

declare -A DEFORMABLE_DETR_CONFIGS=(
    ["deformable-detr-r18"]="deformable-detr/train_deformable_r18.py"
)

declare -A DEIM_CONFIGS=(
    ["deim-s-dairv2x"]="DEIM/configs/deim_dfine/deim_hgnetv2_s_dairv2x.yml"
    ["deim-s-uadetrac"]="DEIM/configs/deim_dfine/deim_hgnetv2_s_uadetrac.yml"
)

declare -A DFINE_CONFIGS=(
    ["dfine-s-dairv2x"]="D-FINE/configs/dfine/dfine_hgnetv2_s_dairv2x.yml"
    ["dfine-s-uadetrac"]="D-FINE/configs/dfine/dfine_hgnetv2_s_uadetrac.yml"
)

# 构建全部配置列表与名称映射
all_configs_paths=()
declare -A NAME_TO_PATH

build_all_configs() {
    all_configs_paths=()
    NAME_TO_PATH=()
    local _config_stem
    for key in "${!RTDETRV2_ADAPTER_CONFIGS[@]}"; do
        local p="${RTDETRV2_ADAPTER_CONFIGS[$key]}"
        all_configs_paths+=("$p")
        NAME_TO_PATH["$key"]="$p"
        local b
        _config_stem=$(basename "${p%%@*}")
        b="${_config_stem%.yaml}"
        b="${b%.yml}"
        NAME_TO_PATH["$b"]="$p"
    done
    for key in "${!CaS_DETR_CONFIGS[@]}"; do
        local p="${CaS_DETR_CONFIGS[$key]}"
        all_configs_paths+=("$p")
        local b
        _config_stem=$(basename "$p")
        b="${_config_stem%.yaml}"
        b="${b%.yml}"
        NAME_TO_PATH["$key"]="$p"
        NAME_TO_PATH["$b"]="$p"
    done
    for key in "${!YOLOV5_CONFIGS[@]}"; do
        local p="${YOLOV5_CONFIGS[$key]}"
        all_configs_paths+=("$p")
        local b
        _config_stem=$(basename "$p")
        b="${_config_stem%.yaml}"
        b="${b%.yml}"
        NAME_TO_PATH["$key"]="$p"
        NAME_TO_PATH["$b"]="$p"
    done
    for key in "${!YOLOV8_CONFIGS[@]}"; do
        local p="${YOLOV8_CONFIGS[$key]}"
        all_configs_paths+=("$p")
        local b
        _config_stem=$(basename "$p")
        b="${_config_stem%.yaml}"
        b="${b%.yml}"
        NAME_TO_PATH["$key"]="$p"
        NAME_TO_PATH["$b"]="$p"
    done
    for key in "${!YOLOV12_CONFIGS[@]}"; do
        local p="${YOLOV12_CONFIGS[$key]}"
        all_configs_paths+=("$p")
        local b
        _config_stem=$(basename "$p")
        b="${_config_stem%.yaml}"
        b="${b%.yml}"
        NAME_TO_PATH["$key"]="$p"
        NAME_TO_PATH["$b"]="$p"
    done
    for key in "${!YOLOX_CONFIGS[@]}"; do
        local p="${YOLOX_CONFIGS[$key]}"
        all_configs_paths+=("$p")
        local b
        _config_stem=$(basename "$p")
        b="${_config_stem%.yaml}"
        b="${b%.yml}"
        NAME_TO_PATH["$key"]="$p"
        NAME_TO_PATH["$b"]="$p"
    done
    for key in "${!FASTER_RCNN_CONFIGS[@]}"; do
        local p="${FASTER_RCNN_CONFIGS[$key]}"
        all_configs_paths+=("$p")
        local b
        _config_stem=$(basename "$p")
        b="${_config_stem%.yaml}"
        b="${b%.yml}"
        NAME_TO_PATH["$key"]="$p"
        NAME_TO_PATH["$b"]="$p"
    done
    for key in "${!DEFORMABLE_DETR_CONFIGS[@]}"; do
        local p="${DEFORMABLE_DETR_CONFIGS[$key]}"
        all_configs_paths+=("$p")
        local b
        b=$(basename "$p" .py)
        NAME_TO_PATH["$key"]="$p"
        NAME_TO_PATH["$b"]="$p"
    done
    for key in "${!DEIM_CONFIGS[@]}"; do
        local p="${DEIM_CONFIGS[$key]}"
        all_configs_paths+=("$p")
        local b
        _config_stem=$(basename "$p")
        b="${_config_stem%.yaml}"
        b="${b%.yml}"
        NAME_TO_PATH["$key"]="$p"
        NAME_TO_PATH["$b"]="$p"
    done
    for key in "${!DFINE_CONFIGS[@]}"; do
        local p="${DFINE_CONFIGS[$key]}"
        all_configs_paths+=("$p")
        local b
        _config_stem=$(basename "$p")
        b="${_config_stem%.yaml}"
        b="${b%.yml}"
        NAME_TO_PATH["$key"]="$p"
        NAME_TO_PATH["$b"]="$p"
    done
}

# 从批量日志中收集失败实验并映射到配置
collect_failed_from_logs() {
    local log_dir="$1"
    local csv="$log_dir/results.csv"
    local failed=()
    if [ ! -f "$csv" ]; then
        log_error "未找到结果文件: $csv"
        return 1
    fi
    while IFS=',' read -r name status duration timestamp; do
        if [ "$status" = "FAILED" ]; then
            failed+=("$name")
        fi
    done < <(tail -n +2 "$csv")

    if [ ${#failed[@]} -eq 0 ]; then
        log_info "上次没有失败的实验"
        return 2
    fi

    CONFIGS_TO_RUN=()
    for n in "${failed[@]}"; do
        if [[ -n "${NAME_TO_PATH[$n]}" ]]; then
            CONFIGS_TO_RUN+=("${NAME_TO_PATH[$n]}")
            continue
        fi
        local n2=${n%.yaml}
        if [[ -n "${NAME_TO_PATH[$n2]}" ]]; then
            CONFIGS_TO_RUN+=("${NAME_TO_PATH[$n2]}")
        else
            log_warning "无法映射失败实验名称到配置: $n"
        fi
    done
}

# 交互式序号选择，支持 1,3-5
select_by_indices() {
    local -n list_ref=$1
    local input="$2"
    CONFIGS_TO_RUN=()
    IFS=',' read -ra parts <<< "$input"
    for part in "${parts[@]}"; do
        if [[ "$part" =~ ^([0-9]+)-([0-9]+)$ ]]; then
            local start=${BASH_REMATCH[1]}
            local end=${BASH_REMATCH[2]}
            for ((i=start; i<=end; i++)); do
                local idx=$((i-1))
                if [ $idx -ge 0 ] && [ $idx -lt ${#list_ref[@]} ]; then
                    CONFIGS_TO_RUN+=("${list_ref[$idx]}")
                fi
            done
        elif [[ "$part" =~ ^[0-9]+$ ]]; then
            local idx=$((part-1))
            if [ $idx -ge 0 ] && [ $idx -lt ${#list_ref[@]} ]; then
                CONFIGS_TO_RUN+=("${list_ref[$idx]}")
            fi
        fi
    done
}

# 解析命令行参数
CONFIGS_TO_RUN=()

parse_arguments() {
    build_all_configs
    SCOPE_DAIRV2X=false
    SCOPE_UADETRAC=false
    
    # 首先检查特殊标志（--test 和 backbone选择）
    local args=("$@")
    local has_test=false
    local has_r18=false
    local has_n=false
    local has_s=false
    local has_m=false
    local has_k05=false
    local has_k07=false
    local filtered_args=()
    
    local idx=0
    while [ $idx -lt ${#args[@]} ]; do
        local arg="${args[$idx]}"
        if [ "$arg" == "--dairv2x" ]; then
            SCOPE_DAIRV2X=true
            idx=$((idx + 1))
            continue
        fi
        if [ "$arg" == "--uadetrac" ]; then
            SCOPE_UADETRAC=true
            idx=$((idx + 1))
            continue
        fi
        if [ "$arg" == "--test" ]; then
            has_test=true
            TEST_MODE=true
            idx=$((idx + 1))
            continue
        fi
        if [ "$arg" == "--yes" ] || [ "$arg" == "-y" ]; then
            SKIP_CONFIRM=true
            idx=$((idx + 1))
            continue
        fi
        if [ "$arg" == "--r18" ]; then
            has_r18=true
            idx=$((idx + 1))
            continue
        fi
        if [ "$arg" == "--n" ]; then
            has_n=true
            SCOPE_SIZE_N=true
            idx=$((idx + 1))
            continue
        fi
        if [ "$arg" == "--s" ]; then
            has_s=true
            SCOPE_SIZE_S=true
            idx=$((idx + 1))
            continue
        fi
        if [ "$arg" == "--m" ]; then
            has_m=true
            SCOPE_SIZE_M=true
            idx=$((idx + 1))
            continue
        fi
        if [ "$arg" == "--k0.5" ]; then
            has_k05=true
            idx=$((idx + 1))
            continue
        fi
        if [ "$arg" == "--k0.7" ]; then
            has_k07=true
            idx=$((idx + 1))
            continue
        fi
        if [ "$arg" == "--dataset" ]; then
            local dataset_arg="${args[$((idx + 1))]}"
            if [ -z "$dataset_arg" ]; then
                log_error "--dataset 需要提供值（可逗号分隔，如 dairv2x 或 dairv2x,uadetrac）"
                exit 1
            fi
            IFS=',' read -ra ds_items <<< "$dataset_arg"
            for ds in "${ds_items[@]}"; do
                local cleaned="${ds//[[:space:]]/}"
                if [ "$cleaned" = "dairv2x" ]; then
                    SCOPE_DAIRV2X=true
                elif [ "$cleaned" = "uadetrac" ]; then
                    SCOPE_UADETRAC=true
                elif [ -n "$cleaned" ]; then
                    log_warning "忽略未知的数据集名（仅支持 dairv2x、uadetrac）: $cleaned"
                fi
            done
            idx=$((idx + 2))
            continue
        fi
        filtered_args+=("$arg")
        idx=$((idx + 1))
    done

    if [ "$SCOPE_DAIRV2X" = true ] || [ "$SCOPE_UADETRAC" = true ]; then
        log_info "数据集作用域已启用（--dataset / --dairv2x / --uadetrac），可与 --rt-detr、--cas_detr 等任意顺序组合"
    fi
    if [ "$has_n" = true ] || [ "$has_s" = true ] || [ "$has_m" = true ]; then
        log_info "模型规模作用域已启用（--n / --s / --m）；推荐与 --yolo 或 --yolov5/--yolov8/--yolov12/--yolox 组合"
    fi

    # 如果设置了测试模式，显示提示
    if [ "$has_test" = true ]; then
        log_info "测试模式：每个实验只跑2个epoch进行快速验证"
    fi
    
    # 如果指定了backbone过滤，应用过滤逻辑
    local backbone_filter=""
    if [ "$has_r18" = true ]; then
        local selected_backbones=()
        [ "$has_r18" = true ] && selected_backbones+=("R18")
        backbone_filter=$(IFS='+'; echo "${selected_backbones[*]}")
        log_info "Backbone过滤: $backbone_filter"
    fi

    if [ "$has_k05" = true ] || [ "$has_k07" = true ]; then
        local selected_ratios=()
        [ "$has_k05" = true ] && selected_ratios+=("Ratio 0.5")
        [ "$has_k07" = true ] && selected_ratios+=("Ratio 0.7")
        local ratio_filter=$(IFS='+'; echo "${selected_ratios[*]}")
        log_info "Ratio过滤: $ratio_filter"
    fi
    
    # 根据过滤后的参数决定运行哪些配置
    set -- "${filtered_args[@]}"
    
    # 过滤配置函数
    filter_config() {
        local config_path="$1"
        
        # 1. Backbone Filter
        if [ -n "$backbone_filter" ]; then
            local match_backbone=false
            if [ "$has_r18" = true ] && ([[ "$config_path" == *"r18"* ]] || [[ "$config_path" == *"presnet18"* ]]); then
                match_backbone=true
            fi
            if [ "$match_backbone" = false ]; then return 1; fi
        fi
        
        # 2. Ratio Filter (k0.5 / k0.7)
        if [ "$has_k05" = true ] || [ "$has_k07" = true ]; then
            local match_ratio=false
            if [ "$has_k05" = true ] && ([[ "$config_path" == *"ratio0.5"* ]] || [[ "$config_path" == *"keep05"* ]]); then
                match_ratio=true
            fi
            if [ "$has_k07" = true ] && ([[ "$config_path" == *"ratio0.7"* ]] || [[ "$config_path" == *"keep07"* ]]); then
                match_ratio=true
            fi
            if [ "$match_ratio" = false ]; then return 1; fi
        fi
        
        return 0
    }
    
    # 检查是否有--core选项（优先处理）
    for arg in "$@"; do
        if [ "$arg" == "--core" ]; then
            CONFIGS_TO_RUN=()
            local _core_p
            for _core_p in "${CORE_EXPERIMENTS[@]}"; do
                if filter_config "$_core_p"; then
                    CONFIGS_TO_RUN+=("$_core_p")
                fi
            done
            apply_dataset_scope_filter_to_configs
            apply_model_size_filter_to_configs
            log_info "运行核心实验配置（CaS_DETR R18 DAIR）"
            return 0
        fi
    done
    
    # 收集所有指定的实验类型（支持多个参数叠加）
    local has_rtdetrv2=false
    local has_cas_detr=false
    local has_yolov5=false
    local has_yolov8=false
    local has_yolov12=false
    local has_yolox=false
    local has_fasterrcnn=false
    local has_deformable_detr=false
    local has_deim=false
    local has_dfine=false
    
    for arg in "$@"; do
        case "$arg" in
            --rt-detr|--rtdetr|--rtdetrv2|--rt-detr-v2)
                has_rtdetrv2=true
                ;;
            --rt-detr-finetune)
                log_warning "--rt-detr-finetune 已移除，请使用 --rtdetrv2"
                has_rtdetrv2=true
                ;;
            --cas_detr)
                has_cas_detr=true
                ;;
            --yolov5)
                has_yolov5=true
                ;;
            --yolov8)
                has_yolov8=true
                ;;
            --yolov12)
                has_yolov12=true
                ;;
            --yolox)
                has_yolox=true
                ;;
            --yolo|--yolo-all)
                has_yolov5=true
                has_yolov8=true
                has_yolov12=true
                has_yolox=true
                ;;
            --fasterrcnn)
                has_fasterrcnn=true
                ;;
            --deformable-detr)
                has_deformable_detr=true
                ;;
            --deim)
                has_deim=true
                ;;
            --dfine|--d-fine)
                has_dfine=true
                ;;
        esac
    done
    
    # 如果指定了实验类型，只运行指定的类型（支持多个）
    if [ "$has_rtdetrv2" = true ] || [ "$has_cas_detr" = true ] || [ "$has_yolov5" = true ] || [ "$has_yolov8" = true ] || [ "$has_yolov12" = true ] || [ "$has_yolox" = true ] || [ "$has_fasterrcnn" = true ] || [ "$has_deformable_detr" = true ] || [ "$has_deim" = true ] || [ "$has_dfine" = true ]; then
        # 显示将要运行的类型
        local selected_types=()
        [ "$has_rtdetrv2" = true ] && selected_types+=("RT-DETRv2+train_adapter")
        [ "$has_cas_detr" = true ] && selected_types+=("CaS_DETR")
        [ "$has_yolov5" = true ] && selected_types+=("YOLOv5")
        [ "$has_yolov8" = true ] && selected_types+=("YOLOv8")
        [ "$has_yolov12" = true ] && selected_types+=("YOLOv12")
        [ "$has_yolox" = true ] && selected_types+=("YOLOX")
        [ "$has_fasterrcnn" = true ] && selected_types+=("FasterRCNN")
        [ "$has_deformable_detr" = true ] && selected_types+=("Deformable-DETR")
        [ "$has_deim" = true ] && selected_types+=("DEIM")
        [ "$has_dfine" = true ] && selected_types+=("D-FINE")
        local types_str=$(IFS='+'; echo "${selected_types[*]}")
        if [ "$has_test" = true ]; then
            log_info "测试模式：运行指定实验类型（按字典序排序）: $types_str"
        else
            log_info "运行指定实验类型（按字典序排序）: $types_str"
        fi
        
        # 根据指定的类型添加配置
        if [ "$has_rtdetrv2" = true ]; then
            for key in $(printf '%s\n' "${!RTDETRV2_ADAPTER_CONFIGS[@]}" | sort); do
                local p="${RTDETRV2_ADAPTER_CONFIGS[$key]}"
                if filter_config "$p"; then
                    CONFIGS_TO_RUN+=("$p")
                fi
            done
        fi
        
        if [ "$has_cas_detr" = true ]; then
            for key in $(printf '%s\n' "${!CaS_DETR_CONFIGS[@]}" | sort); do
                local p="${CaS_DETR_CONFIGS[$key]}"
                if filter_config "$p"; then
                    CONFIGS_TO_RUN+=("$p")
                fi
            done
        fi

        if [ "$has_yolov5" = true ]; then
            for key in $(printf '%s\n' "${!YOLOV5_CONFIGS[@]}" | sort); do
                local p="${YOLOV5_CONFIGS[$key]}"
                CONFIGS_TO_RUN+=("$p")
            done
        fi

        if [ "$has_yolov8" = true ]; then
            for key in $(printf '%s\n' "${!YOLOV8_CONFIGS[@]}" | sort); do
                local p="${YOLOV8_CONFIGS[$key]}"
                # YOLOv8不使用backbone过滤（它有自己的模型大小）
                CONFIGS_TO_RUN+=("$p")
            done
        fi

        if [ "$has_yolov12" = true ]; then
            for key in $(printf '%s\n' "${!YOLOV12_CONFIGS[@]}" | sort); do
                local p="${YOLOV12_CONFIGS[$key]}"
                # YOLOv12不使用backbone过滤（它有自己的模型大小）
                CONFIGS_TO_RUN+=("$p")
            done
        fi

        if [ "$has_yolox" = true ]; then
            for key in $(printf '%s\n' "${!YOLOX_CONFIGS[@]}" | sort); do
                local p="${YOLOX_CONFIGS[$key]}"
                CONFIGS_TO_RUN+=("$p")
            done
        fi

        if [ "$has_fasterrcnn" = true ]; then
            for key in $(printf '%s\n' "${!FASTER_RCNN_CONFIGS[@]}" | sort); do
                local p="${FASTER_RCNN_CONFIGS[$key]}"
                CONFIGS_TO_RUN+=("$p")
            done
        fi

        if [ "$has_deformable_detr" = true ]; then
            for key in $(printf '%s\n' "${!DEFORMABLE_DETR_CONFIGS[@]}" | sort); do
                local p="${DEFORMABLE_DETR_CONFIGS[$key]}"
                if filter_config "$p"; then
                    CONFIGS_TO_RUN+=("$p")
                fi
            done
        fi

        if [ "$has_deim" = true ]; then
            for key in $(printf '%s\n' "${!DEIM_CONFIGS[@]}" | sort); do
                local p="${DEIM_CONFIGS[$key]}"
                CONFIGS_TO_RUN+=("$p")
            done
        fi

        if [ "$has_dfine" = true ]; then
            for key in $(printf '%s\n' "${!DFINE_CONFIGS[@]}" | sort); do
                local p="${DFINE_CONFIGS[$key]}"
                CONFIGS_TO_RUN+=("$p")
            done
        fi
    elif [ $# -eq 0 ]; then
        # 默认：运行所有实验（按字典序排序）
        if [ "$has_test" = true ]; then
            log_info "测试模式运行所有配置"
        else
            log_info "未指定参数，将运行所有实验（按字典序排序）"
        fi
        # RT-DETR v2 train_adapter（按字典序）
        for key in $(printf '%s\n' "${!RTDETRV2_ADAPTER_CONFIGS[@]}" | sort); do
            local p="${RTDETRV2_ADAPTER_CONFIGS[$key]}"
            if filter_config "$p"; then
                CONFIGS_TO_RUN+=("$p")
            fi
        done
        # CaS-DETR 第一阶段实验（按字典序）
        for key in $(printf '%s\n' "${!CaS_DETR_CONFIGS[@]}" | sort); do
            local p="${CaS_DETR_CONFIGS[$key]}"
            if filter_config "$p"; then
                CONFIGS_TO_RUN+=("$p")
            fi
        done
        # YOLOv5实验
        for key in $(printf '%s\n' "${!YOLOV5_CONFIGS[@]}" | sort); do
            local p="${YOLOV5_CONFIGS[$key]}"
            CONFIGS_TO_RUN+=("$p")
        done
        # YOLOv8实验
        for key in $(printf '%s\n' "${!YOLOV8_CONFIGS[@]}" | sort); do
            local p="${YOLOV8_CONFIGS[$key]}"
            # YOLOv8不使用backbone过滤（它有自己的模型大小）
            CONFIGS_TO_RUN+=("$p")
        done
        # YOLOv12实验
        for key in $(printf '%s\n' "${!YOLOV12_CONFIGS[@]}" | sort); do
            local p="${YOLOV12_CONFIGS[$key]}"
            CONFIGS_TO_RUN+=("$p")
        done
        # YOLOX 实验
        for key in $(printf '%s\n' "${!YOLOX_CONFIGS[@]}" | sort); do
            local p="${YOLOX_CONFIGS[$key]}"
            CONFIGS_TO_RUN+=("$p")
        done
        # torchvision Faster R-CNN（对照）
        for key in $(printf '%s\n' "${!FASTER_RCNN_CONFIGS[@]}" | sort); do
            local p="${FASTER_RCNN_CONFIGS[$key]}"
            CONFIGS_TO_RUN+=("$p")
        done
        # Deformable-DETR实验
        for key in $(printf '%s\n' "${!DEFORMABLE_DETR_CONFIGS[@]}" | sort); do
            local p="${DEFORMABLE_DETR_CONFIGS[$key]}"
            if filter_config "$p"; then
                CONFIGS_TO_RUN+=("$p")
            fi
        done
    elif [ "$1" == "--custom" ]; then
        log_info "使用自定义配置列表"
        shift
        CONFIGS_TO_RUN=("$@")
    elif [ "$1" == "--keys" ]; then
        log_info "使用键名选择配置"
        shift
        CONFIGS_TO_RUN=()
        for k in "$@"; do
            if [[ -n "${NAME_TO_PATH[$k]}" ]]; then
                CONFIGS_TO_RUN+=("${NAME_TO_PATH[$k]}")
            else
                log_warning "未知键名: $k"
            fi
        done
        if [ ${#CONFIGS_TO_RUN[@]} -eq 0 ]; then
            log_error "--keys 未匹配到任何配置"
            exit 1
        fi
    elif [ "$1" == "--select" ]; then
        log_info "交互式选择配置"
        echo -e "${YELLOW}可选配置:${NC}"
        local i=1
        for cfg in "${all_configs_paths[@]}"; do
            echo -e "  ${CYAN}[$i]${NC} $cfg"
            ((i++))
        done
        read -p "请输入要运行的序号(支持逗号与区间，如 1,3-5): " selection
        select_by_indices all_configs_paths "$selection"
        if [ ${#CONFIGS_TO_RUN[@]} -eq 0 ]; then
            log_error "未选择任何配置"
            exit 1
        fi
    elif [ "$1" == "--rerun-failed" ]; then
        shift
        local target_log_dir="$1"
        if [ -z "$target_log_dir" ]; then
            local base_dir="$SCRIPT_DIR/logs"
            if [ ! -d "$base_dir" ]; then
                log_error "未找到日志目录: $base_dir"
                exit 1
            fi
            target_log_dir=$(ls -1d "$base_dir"/batch_experiments_* 2>/dev/null | sort | tail -n 1)
            if [ -z "$target_log_dir" ]; then
                log_error "未找到任何批量日志目录"
                exit 1
            fi
        fi
        log_info "从日志目录收集失败实验: $target_log_dir"
        if ! collect_failed_from_logs "$target_log_dir"; then
            exit 1
        fi
        if [ ${#CONFIGS_TO_RUN[@]} -eq 0 ]; then
            log_warning "没有需要重跑的失败实验"
            exit 0
        fi
    else
        log_error "未知参数: $1"
        echo "使用方法："
        echo "  ./run_batch_experiments.sh                                 # 运行所有实验（完整epochs）"
        echo "  ./run_batch_experiments.sh --test                          # 测试模式：所有配置各跑2个epoch"
        echo "  ./run_batch_experiments.sh --rt-detr                       # 与 --rtdetrv2 相同，仅 RT-DETR v2"
        echo "  ./run_batch_experiments.sh --rtdetrv2                      # 官方 rtdetrv2_pytorch + train_adapter（默认 --cas-eval）"
        echo "  ./run_batch_experiments.sh --cas_detr                      # 只运行新的 CaS-DETR 第一阶段消融（5个）"
        echo "  ./run_batch_experiments.sh --yolov5                        # 只运行YOLOv5"
        echo "  ./run_batch_experiments.sh --yolov8                        # 只运行YOLOv8"
        echo "  ./run_batch_experiments.sh --yolov12                       # 只运行YOLOv12"
        echo "  ./run_batch_experiments.sh --yolox                         # 只运行 YOLOX"
        echo "  ./run_batch_experiments.sh --yolo                          # 一键 YOLOv5+v8+v12+YOLOX"
        echo "  ./run_batch_experiments.sh --yolo --s                      # 仅 s 规模（两数据集）"
        echo "  ./run_batch_experiments.sh --yolo --n --dairv2x            # 仅 DAIR 的 n 规模"
        echo "  ./run_batch_experiments.sh --fasterrcnn                    # 只运行 torchvision Faster R-CNN"
        echo "  ./run_batch_experiments.sh --deim                           # 只运行 DEIM-S（DAIR + UA-DETRAC）"
        echo "  ./run_batch_experiments.sh --dfine                          # 只运行 D-FINE-S（DAIR + UA-DETRAC）"
        echo "  ./run_batch_experiments.sh --deformable-detr               # 只运行Deformable-DETR"
        echo "  ./run_batch_experiments.sh --test --rt-detr                # 测试模式只跑 RT-DETR v2"
        echo "  ./run_batch_experiments.sh --test --cas_detr               # 测试模式只运行 CaS-DETR"
        echo "  ./run_batch_experiments.sh --test --yolov5                 # 测试模式只运行YOLOv5"
        echo "  ./run_batch_experiments.sh --test --yolov8                 # 测试模式只运行YOLOv8"
        echo "  ./run_batch_experiments.sh --test --yolov12                # 测试模式只运行YOLOv12"
        echo "  ./run_batch_experiments.sh --test --yolox                  # 测试模式只运行 YOLOX"
        echo "  ./run_batch_experiments.sh --test --fasterrcnn             # 测试模式只运行 Faster R-CNN"
        echo "  ./run_batch_experiments.sh --test --deformable-detr        # 测试模式只运行Deformable-DETR"
        echo "  ./run_batch_experiments.sh --rtdetrv2 --cas_detr               # 运行多个实验类型（可叠加）"
        echo "  ./run_batch_experiments.sh --test --rtdetrv2 --cas_detr          # 测试模式运行多个类型"
        echo "  ./run_batch_experiments.sh --r18                           # 只运行R18"
        echo "  ./run_batch_experiments.sh --r18                           # 只运行R18"
        echo "  ./run_batch_experiments.sh --n                             # 只运行所有 n 规模 YOLO（v5/v8/v12）"
        echo "  ./run_batch_experiments.sh --s                             # 只运行所有 s 规模 YOLO / YOLOX"
        echo "  ./run_batch_experiments.sh --m                             # 只运行所有 m 规模 YOLO / YOLOX"
        echo "  ./run_batch_experiments.sh --k0.5                          # 只跑路径名含 ratio0.5 的配置"
        echo "  ./run_batch_experiments.sh --k0.7                          # 只运行 Keep Ratio 0.7"
        echo "  ./run_batch_experiments.sh --core                          # 只运行核心实验（CaS-DETR moe+cass, DAIR）"
        echo "  ./run_batch_experiments.sh --custom cfg1.yaml cfg2.yaml    # 指定配置文件路径"
        echo "  ./run_batch_experiments.sh --keys rtdetrv2-r18-dairv2x casdeim-moe-only-dairv2x   # 通过键名选择"
        echo "  ./run_batch_experiments.sh --dairv2x                       # 数据集筛：仅 DAIR-V2X（可叠 --rtdetrv2 等）"
        echo "  ./run_batch_experiments.sh --uadetrac                      # 数据集筛：仅 UA-DETRAC"
        echo "  ./run_batch_experiments.sh --dataset dairv2x --rtdetrv2     # 推荐：数据集 + 实验类型（顺序任意；--rt-detr 同 --rtdetrv2）"
        echo "  ./run_batch_experiments.sh --dataset dairv2x,uadetrac       # 同传 --dairv2x --uadetrac（不筛）"
        echo "  ./run_batch_experiments.sh --select                        # 交互式选择"
        echo "  ./run_batch_experiments.sh --rerun-failed [LOG_DIR]        # 重跑失败实验"
        echo "  ./run_batch_experiments.sh --yes --cas_detr                 # 非交互一键跑 CaS-DETR"
        echo "  ./run_batch_experiments.sh --dairv2x --rtdetrv2             # 仅 DAIR-V2X 的 RT-DETR v2"
        echo "  ./run_batch_experiments.sh --dataset dairv2x --rtdetr       # 同上（--dataset 写法）"
        echo "  ./run_batch_experiments.sh --dairv2x --cas_detr             # 仅 DAIR-V2X 的 CaS-DETR 第一阶段消融"
        echo "  ./run_batch_experiments.sh --yolo --s                      # 同上（推荐简写）"
        echo "  ./run_batch_experiments.sh --yolov5 --yolov8 --yolov12 --yolox --s  # 跑所有 s 规模 YOLO / YOLOX"
        exit 1
    fi

    apply_dataset_scope_filter_to_configs
    apply_model_size_filter_to_configs
}

# 运行单个实验
run_single_experiment() {
    local config_path_raw=$1
    local dataset_name="${2:-}"
    local config_path="$config_path_raw"
    local rtdetr_adapter_dataset=""
    if [[ "$config_path_raw" == *"@"* ]]; then
        config_path="${config_path_raw%%@*}"
        rtdetr_adapter_dataset="${config_path_raw#*@}"
    fi
    local exp_name
    exp_name=$(basename "$config_path")
    exp_name="${exp_name%.*}"
    local exp_display="$exp_name"
    if [ -n "$dataset_name" ]; then
        exp_display="${exp_name}@${dataset_name}"
    fi
    local exp_dir=$(dirname "$config_path")
    
    # 检查配置文件是否存在（Deformable-DETR 使用 Python 脚本；路径相对 experiments 根目录）
    if [ ! -f "$SCRIPT_DIR/$config_path" ]; then
        if [[ "$config_path" == *.py ]]; then
            log_error "训练脚本不存在: $config_path"
        else
            log_error "配置文件不存在: $SCRIPT_DIR/$config_path"
        fi
        SKIPPED_EXPERIMENTS=$((SKIPPED_EXPERIMENTS + 1))
        return 1
    fi

    # 官方 RT-DETR v2：rtdetrv2_pytorch/tools/train_adapter.py（默认 --cas-eval，与 CaS 评估对齐）
    if [[ -n "$rtdetr_adapter_dataset" ]] && [[ "$config_path" == RT-DETR/rtdetrv2_pytorch/* ]]; then
        TOTAL_EXPERIMENTS=$((TOTAL_EXPERIMENTS + 1))
        echo ""
        echo -e "${PURPLE}========================================${NC}"
        echo -e "${PURPLE}实验 [$TOTAL_EXPERIMENTS/$TOTAL_PLANNED_RUNS]: ${exp_name}@${rtdetr_adapter_dataset} (RT-DETRv2 train_adapter)${NC}"
        echo -e "${PURPLE}========================================${NC}"
        log_info "RT-DETR v2 / train_adapter: $config_path_raw"
        local start_time=$(date +%s)
        if command -v python3 &> /dev/null; then
            python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
            sleep 1
        fi
        local original_dir=$(pwd)
        cd "$SCRIPT_DIR/RT-DETR/rtdetrv2_pytorch"
        set +e
        local yml_rel="${config_path#RT-DETR/rtdetrv2_pytorch/}"
        local out_tag="batch_${exp_name}_${rtdetr_adapter_dataset}"
        local adapter_cmd=(
            "$PYTHON_BIN" tools/train_adapter.py
            -c "$yml_rel"
            --dataset "$rtdetr_adapter_dataset"
            --output-dir "outputs/${out_tag}"
            --experiment-name "${exp_name}_${rtdetr_adapter_dataset}"
        )
        if [ "${RTDETRV2_CAS_EVAL:-1}" != "0" ]; then
            adapter_cmd+=(--cas-eval)
        fi
        if [ -n "${DAIRV2X_DATA_ROOT:-}" ] && [ "$rtdetr_adapter_dataset" = "dairv2x" ]; then
            adapter_cmd+=(--data-root "${DAIRV2X_DATA_ROOT}")
        fi
        if [ -n "${UADETRAC_DATA_ROOT:-}" ] && [ "$rtdetr_adapter_dataset" = "uadetrac" ]; then
            adapter_cmd+=(--data-root "${UADETRAC_DATA_ROOT}")
        fi
        if [ "$TEST_MODE" = true ]; then
            adapter_cmd+=(-u epoches=2)
        fi
        "${adapter_cmd[@]}"
        local exit_code=$?
        set -e
        cd "$original_dir"
        if command -v python3 &> /dev/null; then
            python3 -c "import torch; torch.cuda.empty_cache(); import gc; gc.collect()" 2>/dev/null || true
            sleep 2
        fi
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        local duration_formatted=$(printf '%02d:%02d:%02d' $((duration/3600)) $((duration%3600/60)) $((duration%60)))
        if [ $exit_code -eq 0 ]; then
            log_success "✓ RT-DETRv2 完成: ${exp_name}@${rtdetr_adapter_dataset} (耗时: $duration_formatted)"
            SUCCESSFUL_EXPERIMENTS=$((SUCCESSFUL_EXPERIMENTS + 1))
        else
            log_error "✗ RT-DETRv2 | ${exp_name}@${rtdetr_adapter_dataset} 失败 (退出码: $exit_code, 耗时: $duration_formatted)"
            FAILED_EXPERIMENTS=$((FAILED_EXPERIMENTS + 1))
        fi
        return $exit_code
    fi

    TOTAL_EXPERIMENTS=$((TOTAL_EXPERIMENTS + 1))
    
    echo ""
    echo -e "${PURPLE}========================================${NC}"
    echo -e "${PURPLE}实验 [$TOTAL_EXPERIMENTS/$TOTAL_PLANNED_RUNS]: $exp_display${NC}"
    echo -e "${PURPLE}========================================${NC}"
    log_info "开始实验: $config_path"
    if [ -n "$dataset_name" ]; then
        log_info "数据集: $dataset_name"
    fi
    
    # 确定训练脚本路径
    local YOLO_VERSION=""
    if [[ "$exp_name" == yolov5* ]]; then
        TRAIN_SCRIPT="yolo/train.py"
        WORK_DIR="yolo"
        YOLO_VERSION="5"
    elif [[ "$exp_name" == yolov8* ]]; then
        TRAIN_SCRIPT="yolo/train.py"
        WORK_DIR="yolo"
        YOLO_VERSION="8"
    elif [[ "$exp_name" == yolov12* ]]; then
        TRAIN_SCRIPT="yolo/train.py"
        WORK_DIR="yolo"
        YOLO_VERSION="12"
    elif [[ "$exp_name" == yolox* ]]; then
        TRAIN_SCRIPT="yolo/train_yolox.py"
        WORK_DIR="yolo"
        YOLO_VERSION=""
    elif [[ "$exp_name" == fasterrcnn* ]]; then
        TRAIN_SCRIPT="yolo/train_fasterrcnn.py"
        WORK_DIR="yolo"
        YOLO_VERSION=""
    elif [[ "$config_path" == DEIM/* ]]; then
        TRAIN_SCRIPT="train.py"
        WORK_DIR="DEIM"
    elif [[ "$config_path" == D-FINE/* ]]; then
        TRAIN_SCRIPT="train.py"
        WORK_DIR="D-FINE"
    elif [[ "$config_path" == CaS-DETR/* ]]; then
        TRAIN_SCRIPT="train.py"
        WORK_DIR="CaS-DETR"
    elif [[ "$exp_dir" == *"deformable-detr"* ]]; then
        # Deformable-DETR 使用 Python 脚本而不是 YAML 配置
        TRAIN_SCRIPT="deformable-detr/train_deformable_r18.py"
        WORK_DIR="deformable-detr"
    else
        TRAIN_SCRIPT="train.py"
        WORK_DIR="CaS-DETR"
    fi
    
    # 记录开始时间
    local start_time=$(date +%s)
    
    # 清理GPU缓存（在实验开始前清理，避免前一个实验的内存残留）
    if command -v python3 &> /dev/null; then
        python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
        sleep 1  # 给GPU一点时间释放内存
    fi
    
    # 运行训练（让训练脚本自己管理日志输出）
    # 保存原始目录以便正确返回
    local original_dir=$(pwd)
    cd "$WORK_DIR"
    set +e  # 临时允许错误
    
    # DEIM / CaS-DETR / D-FINE: train.py -c <yml>；整网微调路径默认写在 yaml 的 tuning
    if [[ "$WORK_DIR" == "DEIM" ]] || [[ "$WORK_DIR" == "CaS-DETR" ]] || [[ "$WORK_DIR" == "D-FINE" ]]; then
        local fw_flag="deim"
        [[ "$WORK_DIR" == "CaS-DETR" ]] && fw_flag="casdeim"
        [[ "$WORK_DIR" == "D-FINE" ]] && fw_flag="dfine"
        local yml_rel="${config_path#${WORK_DIR}/}"  # e.g. configs/deim_dfine/deim_hgnetv2_s_dairv2x.yml

        local pretrained_arg=""
        if [[ "$WORK_DIR" == "DEIM" ]] && [[ -n "${DEIM_TUNING_CKPT:-}" ]]; then
            pretrained_arg="-t ${DEIM_TUNING_CKPT}"
        elif [[ "$WORK_DIR" == "CaS-DETR" ]] && [[ -n "${DEIM_TUNING_CKPT:-}" ]]; then
            pretrained_arg="-t ${DEIM_TUNING_CKPT}"
        elif [[ "$WORK_DIR" == "D-FINE" ]] && [[ -n "${DFINE_TUNING_CKPT:-}" ]]; then
            pretrained_arg="-t ${DFINE_TUNING_CKPT}"
        fi

        if [ "$TEST_MODE" = true ]; then
            "$PYTHON_BIN" train.py -c "$yml_rel" $pretrained_arg --test-only 2>&1 || true
            log_warning "DEIM/CaS-DETR/D-FINE test-only: 跳过完整训练，仅验证配置可加载"
        else
            "$PYTHON_BIN" train.py -c "$yml_rel" $pretrained_arg
        fi
        local train_exit=$?

        # 训练成功后，运行 CaS 兼容评估
        if [ $train_exit -eq 0 ] && [ "$TEST_MODE" != true ]; then
            cd "$original_dir"
            log_info "运行 CaS 兼容评估: $config_path"
            "$PYTHON_BIN" common/eval_deim_dfine.py \
                --framework "$fw_flag" \
                --config "$config_path" \
                --model-name "$exp_name" \
                --device cuda 2>&1 || log_warning "CaS 评估失败（不影响训练结果）"
            cd "$WORK_DIR"
        fi

    # Deformable-DETR 使用 Python 脚本，不需要 --config 参数
    elif [[ "$exp_dir" == *"deformable-detr"* ]]; then
        # 如果是测试模式，传递 --epochs 2 参数
        if [ "$TEST_MODE" = true ]; then
            "$PYTHON_BIN" train_deformable_r18.py --epochs 2
        else
            "$PYTHON_BIN" train_deformable_r18.py
        fi
    # 其他实验使用 YAML 配置文件
    elif [[ "$TRAIN_SCRIPT" == *train_yolox.py* ]] && [ "$TEST_MODE" = true ]; then
        "$PYTHON_BIN" train_yolox.py --config "../$config_path" --epochs 2
    elif [[ "$TRAIN_SCRIPT" == *train_yolox.py* ]]; then
        "$PYTHON_BIN" train_yolox.py --config "../$config_path"
    elif [[ "$exp_name" == fasterrcnn* ]] && [ "$TEST_MODE" = true ]; then
        local FRCNN_DS="dairv2x"
        [[ "$exp_name" == *uadetrac* ]] && FRCNN_DS="uadetrac"
        "$PYTHON_BIN" train_fasterrcnn.py --config "../$config_path" --dataset "$FRCNN_DS" --epochs 2
    elif [[ "$exp_name" == fasterrcnn* ]]; then
        local FRCNN_DS="dairv2x"
        [[ "$exp_name" == *uadetrac* ]] && FRCNN_DS="uadetrac"
        "$PYTHON_BIN" train_fasterrcnn.py --config "../$config_path" --dataset "$FRCNN_DS"
    elif [ -n "$YOLO_VERSION" ] && [ "$TEST_MODE" = true ]; then
        "$PYTHON_BIN" train.py --version "$YOLO_VERSION" --config "../$config_path" --epochs 2
    elif [ -n "$YOLO_VERSION" ]; then
        "$PYTHON_BIN" train.py --version "$YOLO_VERSION" --config "../$config_path"
    elif [ "$TEST_MODE" = true ]; then
        "$PYTHON_BIN" train.py --config "../$config_path" --epochs 2
    else
        "$PYTHON_BIN" train.py --config "../$config_path"
    fi
    
    # DEIM/CaS-DETR/D-FINE 在 if 分支内已有 train_exit；其余分支用 $?
    if [[ "$WORK_DIR" == "DEIM" ]] || [[ "$WORK_DIR" == "CaS-DETR" ]] || [[ "$WORK_DIR" == "D-FINE" ]]; then
        local exit_code=${train_exit:-$?}
    else
        local exit_code=$?
    fi
    set -e
    cd "$original_dir"
    
    # 清理GPU缓存（防止内存泄漏影响后续实验）
    if command -v python3 &> /dev/null; then
        python3 -c "import torch; torch.cuda.empty_cache(); import gc; gc.collect()" 2>/dev/null || true
        sleep 2  # 给GPU更多时间释放内存
    fi
    
    # 记录结束时间
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local duration_formatted=$(printf '%02d:%02d:%02d' $((duration/3600)) $((duration%3600/60)) $((duration%60)))
    
    # 检查训练结果
    if [ $exit_code -eq 0 ]; then
        log_success "✓ 实验完成: $exp_display (耗时: $duration_formatted)"
        SUCCESSFUL_EXPERIMENTS=$((SUCCESSFUL_EXPERIMENTS + 1))
    else
        log_error "✗ $exp_display | 失败 (退出码: $exit_code, 耗时: $duration_formatted)"
        FAILED_EXPERIMENTS=$((FAILED_EXPERIMENTS + 1))
    fi
    
    return $exit_code
}

# 生成最终报告
generate_report() {
    # 仅输出到终端，不生成文件
    echo ""
    echo -e "${CYAN}=========================================${NC}"
    echo -e "${CYAN}批量实验完成！${NC}"
    echo -e "${CYAN}=========================================${NC}"
    echo -e "${GREEN}成功: $SUCCESSFUL_EXPERIMENTS${NC} | ${RED}失败: $FAILED_EXPERIMENTS${NC} | ${YELLOW}跳过: $SKIPPED_EXPERIMENTS${NC}"
    echo ""
    echo -e "${BLUE}提示: 实验结果（包括mAP等指标）已保存在各训练脚本生成的日志目录中${NC}"
    echo -e "${BLUE}      - CaS-DETR 消融日志: CaS-DETR/outputs/ablation/${NC}"
    echo -e "${BLUE}      - YOLO统一日志: yolo/logs/${NC}"
    echo -e "${BLUE}      - DEIM日志: DEIM/outputs/${NC}"
    echo -e "${BLUE}      - D-FINE日志: D-FINE/output/${NC}"
    echo -e "${BLUE}      - Deformable-DETR日志: deformable-detr/work_dirs/${NC}"
    echo -e "${BLUE}      - RT-DETR v2（train_adapter + --cas-eval）: RT-DETR/rtdetrv2_pytorch/outputs/batch_*/${NC}"
}

calculate_total_planned_runs() {
    TOTAL_PLANNED_RUNS=${#CONFIGS_TO_RUN[@]}
}

# 主函数
main() {
    cd "$SCRIPT_DIR" || { log_error "无法切换到脚本目录: $SCRIPT_DIR"; exit 1; }

    echo -e "${CYAN}"
    echo "╔════════════════════════════════════════════╗"
    echo "║     RT-DETR 批量实验运行系统              ║"
    echo "║     方案2: 鲁棒批量运行                   ║"
    echo "╚════════════════════════════════════════════╝"
    echo -e "${NC}"
    
    # 解析参数
    parse_arguments "$@"
    calculate_total_planned_runs
    
    # 显示将要运行的实验
    echo -e "${YELLOW}将运行以下 ${#CONFIGS_TO_RUN[@]} 个配置，实际执行 ${TOTAL_PLANNED_RUNS} 个实验:${NC}"
    local i=1
    for config in "${CONFIGS_TO_RUN[@]}"; do
        echo -e "  ${CYAN}[$i]${NC} $config"
        ((i++))
    done
    echo ""
    
    # 如果是测试模式，显示额外提示
    if [ "$TEST_MODE" = true ]; then
        echo -e "${YELLOW}⚠️  测试模式：所有配置将只运行2个epoch进行快速验证${NC}"
        echo -e "${YELLOW}   预计总耗时: ~30-60分钟（${TOTAL_PLANNED_RUNS}个实验 × 2 epochs）${NC}"
        echo -e "${YELLOW}   验证通过后，可使用以下命令运行完整训练：${NC}"
        echo -e "${CYAN}   ./run_batch_experiments.sh                # 运行所有实验（完整epochs）${NC}"
        echo ""
    fi
    
    # 确认开始（--yes / -y 跳过）
    if [ "$SKIP_CONFIRM" != true ]; then
        read -p "是否开始批量实验? [Y/n] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]] && [[ -n $REPLY ]]; then
            log_warning "用户取消操作"
            exit 0
        fi
    else
        log_info "已启用 --yes，跳过确认，直接开始批量实验"
    fi
    
    # 禁用CSV结果文件
    # echo "实验名称,状态,耗时,完成时间" > "$BATCH_LOG_DIR/results.csv"
    
    # 记录全局开始时间
    local global_start_time=$(date +%s)
    
    for config in "${CONFIGS_TO_RUN[@]}"; do
        run_single_experiment "$config" || true
    done
    
    # 计算总耗时
    local global_end_time=$(date +%s)
    local total_duration=$((global_end_time - global_start_time))
    local total_duration_formatted=$(printf '%02d:%02d:%02d' $((total_duration/3600)) $((total_duration%3600/60)) $((total_duration%60)))
    
    echo ""
    log_info "所有实验完成! 总耗时: $total_duration_formatted"
    
    # 生成最终报告
    generate_report
    
    # 返回状态码
    if [ $FAILED_EXPERIMENTS -gt 0 ]; then
        exit 1
    else
        exit 0
    fi
}

# 运行主函数
main "$@"
