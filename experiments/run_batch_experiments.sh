#!/bin/bash

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
#   ./run_batch_experiments.sh --rt-detr                       # 只运行RT-DETR实验
#   ./run_batch_experiments.sh --moe-rtdetr                    # 只运行MOE-RTDETR实验
#   ./run_batch_experiments.sh --dset                          # 只运行DSET实验
#   ./run_batch_experiments.sh --yolov8                        # 只运行YOLOv8实验
#   ./run_batch_experiments.sh --test --rt-detr                # 测试模式只运行RT-DETR
#   ./run_batch_experiments.sh --test --moe-rtdetr             # 测试模式只运行MOE-RTDETR
#   ./run_batch_experiments.sh --test --dset                   # 测试模式只运行DSET
#   ./run_batch_experiments.sh --test --yolov8                 # 测试模式只运行YOLOv8
#   ./run_batch_experiments.sh --r18                           # 只运行ResNet-18实验
#   ./run_batch_experiments.sh --r34                           # 只运行ResNet-34实验
#   ./run_batch_experiments.sh --r18 --r34                     # 运行R18和R34实验
#   ./run_batch_experiments.sh --custom cfg1.yaml cfg2.yaml    # 自定义配置列表
#   ./run_batch_experiments.sh --keys rt-detr-r18 moe6-r34     # 使用内置键名选择
#   ./run_batch_experiments.sh --select                        # 交互式选择待运行配置
#   ./run_batch_experiments.sh --rerun-failed [LOG_DIR]        # 自动选择上次失败的实验
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
BATCH_LOG_DIR="$SCRIPT_DIR/logs/batch_experiments_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BATCH_LOG_DIR"

# 全局变量
TOTAL_EXPERIMENTS=0
SUCCESSFUL_EXPERIMENTS=0
FAILED_EXPERIMENTS=0
SKIPPED_EXPERIMENTS=0
TEST_MODE=false  # 测试模式标志

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$BATCH_LOG_DIR/batch_summary.log"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$BATCH_LOG_DIR/batch_summary.log"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$BATCH_LOG_DIR/batch_summary.log"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$BATCH_LOG_DIR/batch_summary.log"
}

# 定义所有可用的实验配置
declare -A RT_DETR_CONFIGS=(
    ["rt-detr-r18"]="rt-detr/configs/rtdetr_presnet18.yaml"
    ["rt-detr-r34"]="rt-detr/configs/rtdetr_presnet34.yaml"
)

declare -A MOE_RTDETR_CONFIGS=(
    ["moe2-r18"]="moe-rtdetr/configs/moe2_presnet18.yaml"
    ["moe2-r34"]="moe-rtdetr/configs/moe2_presnet34.yaml"
    ["moe3-r18"]="moe-rtdetr/configs/moe3_presnet18.yaml"
    ["moe3-r34"]="moe-rtdetr/configs/moe3_presnet34.yaml"
    ["moe6-r18"]="moe-rtdetr/configs/moe6_presnet18.yaml"
    ["moe6-r34"]="moe-rtdetr/configs/moe6_presnet34.yaml"
)

declare -A DSET_CONFIGS=(
    ["dset2-r18"]="dset/configs/dset2_r18.yaml"
    ["dset2-r34"]="dset/configs/dset2_r34.yaml"
    ["dset3-r18"]="dset/configs/dset3_r18.yaml"
    ["dset3-r34"]="dset/configs/dset3_r34.yaml"
    ["dset6-r18"]="dset/configs/dset6_r18.yaml"
    ["dset6-r34"]="dset/configs/dset6_r34.yaml"
    ["dset8-r18"]="dset/configs/dset8_r18.yaml"
    ["dset8-r34"]="dset/configs/dset8_r34.yaml"
)

declare -A YOLOV8_CONFIGS=(
    ["yolov8s"]="yolov8/configs/yolov8s_dairv2x.yaml"
    ["yolov8m"]="yolov8/configs/yolov8m_dairv2x.yaml"
    ["yolov8l"]="yolov8/configs/yolov8l_dairv2x.yaml"
)

# 构建全部配置列表与名称映射
all_configs_paths=()
declare -A NAME_TO_PATH

build_all_configs() {
    all_configs_paths=()
    NAME_TO_PATH=()
    for key in "${!RT_DETR_CONFIGS[@]}"; do
        local p="${RT_DETR_CONFIGS[$key]}"
        all_configs_paths+=("$p")
        local b
        b=$(basename "$p" .yaml)
        NAME_TO_PATH["$key"]="$p"
        NAME_TO_PATH["$b"]="$p"
    done
    for key in "${!MOE_RTDETR_CONFIGS[@]}"; do
        local p="${MOE_RTDETR_CONFIGS[$key]}"
        all_configs_paths+=("$p")
        local b
        b=$(basename "$p" .yaml)
        NAME_TO_PATH["$key"]="$p"
        NAME_TO_PATH["$b"]="$p"
    done
    for key in "${!DSET_CONFIGS[@]}"; do
        local p="${DSET_CONFIGS[$key]}"
        all_configs_paths+=("$p")
        local b
        b=$(basename "$p" .yaml)
        NAME_TO_PATH["$key"]="$p"
        NAME_TO_PATH["$b"]="$p"
    done
    for key in "${!YOLOV8_CONFIGS[@]}"; do
        local p="${YOLOV8_CONFIGS[$key]}"
        all_configs_paths+=("$p")
        local b
        b=$(basename "$p" .yaml)
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
    
    # 首先检查特殊标志（--test 和 backbone选择）
    local args=("$@")
    local has_test=false
    local has_r18=false
    local has_r34=false
    local has_r50=false
    local filtered_args=()
    
    for arg in "${args[@]}"; do
        if [ "$arg" == "--test" ]; then
            has_test=true
            TEST_MODE=true
        elif [ "$arg" == "--r18" ]; then
            has_r18=true
        elif [ "$arg" == "--r34" ]; then
            has_r34=true
        elif [ "$arg" == "--r50" ]; then
            has_r50=true
        else
            filtered_args+=("$arg")
        fi
    done
    
    # 如果设置了测试模式，显示提示
    if [ "$has_test" = true ]; then
        log_info "测试模式：每个实验只跑2个epoch进行快速验证"
    fi
    
    # 如果指定了backbone过滤，应用过滤逻辑
    local backbone_filter=""
    if [ "$has_r18" = true ] || [ "$has_r34" = true ] || [ "$has_r50" = true ]; then
        local selected_backbones=()
        [ "$has_r18" = true ] && selected_backbones+=("R18")
        [ "$has_r34" = true ] && selected_backbones+=("R34")
        [ "$has_r50" = true ] && selected_backbones+=("R50")
        backbone_filter=$(IFS='+'; echo "${selected_backbones[*]}")
        log_info "Backbone过滤: $backbone_filter"
    fi
    
    # 根据过滤后的参数决定运行哪些配置
    set -- "${filtered_args[@]}"
    
    # 过滤配置函数
    filter_by_backbone() {
        local config_path="$1"
        if [ -z "$backbone_filter" ]; then
            return 0  # 无过滤，全部通过
        fi
        # 支持新格式 (r18/r34) 和旧格式 (presnet18/presnet34)
        if [ "$has_r18" = true ] && ([[ "$config_path" == *"r18"* ]] || [[ "$config_path" == *"presnet18"* ]]); then
            return 0
        fi
        if [ "$has_r34" = true ] && ([[ "$config_path" == *"r34"* ]] || [[ "$config_path" == *"presnet34"* ]]); then
            return 0
        fi
        if [ "$has_r50" = true ] && ([[ "$config_path" == *"r50"* ]] || [[ "$config_path" == *"presnet50"* ]]); then
            return 0
        fi
        return 1
    }
    
    # 收集所有指定的实验类型（支持多个参数叠加）
    local has_rt_detr=false
    local has_moe_rtdetr=false
    local has_dset=false
    local has_yolov8=false
    
    for arg in "$@"; do
        case "$arg" in
            --rt-detr)
                has_rt_detr=true
                ;;
            --moe-rtdetr)
                has_moe_rtdetr=true
                ;;
            --dset)
                has_dset=true
                ;;
            --yolov8)
                has_yolov8=true
                ;;
        esac
    done
    
    # 如果指定了实验类型，只运行指定的类型（支持多个）
    if [ "$has_rt_detr" = true ] || [ "$has_moe_rtdetr" = true ] || [ "$has_dset" = true ] || [ "$has_yolov8" = true ]; then
        # 显示将要运行的类型
        local selected_types=()
        [ "$has_rt_detr" = true ] && selected_types+=("RT-DETR")
        [ "$has_moe_rtdetr" = true ] && selected_types+=("MOE-RTDETR")
        [ "$has_dset" = true ] && selected_types+=("DSET")
        [ "$has_yolov8" = true ] && selected_types+=("YOLOv8")
        local types_str=$(IFS='+'; echo "${selected_types[*]}")
        if [ "$has_test" = true ]; then
            log_info "测试模式：运行指定实验类型（按字典序排序）: $types_str"
        else
            log_info "运行指定实验类型（按字典序排序）: $types_str"
        fi
        
        # 根据指定的类型添加配置
        if [ "$has_rt_detr" = true ]; then
            for key in $(printf '%s\n' "${!RT_DETR_CONFIGS[@]}" | sort); do
                local p="${RT_DETR_CONFIGS[$key]}"
                if filter_by_backbone "$p"; then
                    CONFIGS_TO_RUN+=("$p")
                fi
            done
        fi
        
        if [ "$has_moe_rtdetr" = true ]; then
            for key in $(printf '%s\n' "${!MOE_RTDETR_CONFIGS[@]}" | sort); do
                local p="${MOE_RTDETR_CONFIGS[$key]}"
                if filter_by_backbone "$p"; then
                    CONFIGS_TO_RUN+=("$p")
                fi
            done
        fi
        
        if [ "$has_dset" = true ]; then
            for key in $(printf '%s\n' "${!DSET_CONFIGS[@]}" | sort); do
                local p="${DSET_CONFIGS[$key]}"
                if filter_by_backbone "$p"; then
                    CONFIGS_TO_RUN+=("$p")
                fi
            done
        fi
        
        if [ "$has_yolov8" = true ]; then
            for key in $(printf '%s\n' "${!YOLOV8_CONFIGS[@]}" | sort); do
                local p="${YOLOV8_CONFIGS[$key]}"
                # YOLOv8不使用backbone过滤（它有自己的模型大小）
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
        # RT-DETR实验（按字典序）
        for key in $(printf '%s\n' "${!RT_DETR_CONFIGS[@]}" | sort); do
            local p="${RT_DETR_CONFIGS[$key]}"
            if filter_by_backbone "$p"; then
                CONFIGS_TO_RUN+=("$p")
            fi
        done
        # MOE-RTDETR实验（按字典序：moe2→moe3→moe6）
        for key in $(printf '%s\n' "${!MOE_RTDETR_CONFIGS[@]}" | sort); do
            local p="${MOE_RTDETR_CONFIGS[$key]}"
            if filter_by_backbone "$p"; then
                CONFIGS_TO_RUN+=("$p")
            fi
        done
        # DSET实验（按字典序：dset2→dset3→dset6）
        for key in $(printf '%s\n' "${!DSET_CONFIGS[@]}" | sort); do
            local p="${DSET_CONFIGS[$key]}"
            if filter_by_backbone "$p"; then
                CONFIGS_TO_RUN+=("$p")
            fi
        done
        # YOLOv8实验（按字典序：yolov8l→yolov8m→yolov8s）
        for key in $(printf '%s\n' "${!YOLOV8_CONFIGS[@]}" | sort); do
            local p="${YOLOV8_CONFIGS[$key]}"
            # YOLOv8不使用backbone过滤（它有自己的模型大小）
            CONFIGS_TO_RUN+=("$p")
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
        echo "  ./run_batch_experiments.sh --rt-detr                       # 只运行RT-DETR"
        echo "  ./run_batch_experiments.sh --moe-rtdetr                    # 只运行MOE-RTDETR"
        echo "  ./run_batch_experiments.sh --dset                          # 只运行DSET"
        echo "  ./run_batch_experiments.sh --yolov8                        # 只运行YOLOv8"
        echo "  ./run_batch_experiments.sh --test --rt-detr                # 测试模式只运行RT-DETR"
        echo "  ./run_batch_experiments.sh --test --moe-rtdetr             # 测试模式只运行MOE-RTDETR"
        echo "  ./run_batch_experiments.sh --test --dset                   # 测试模式只运行DSET"
        echo "  ./run_batch_experiments.sh --test --yolov8                 # 测试模式只运行YOLOv8"
        echo "  ./run_batch_experiments.sh --rt-detr --moe-rtdetr --dset   # 运行多个实验类型（可叠加）"
        echo "  ./run_batch_experiments.sh --test --rt-detr --dset          # 测试模式运行多个类型"
        echo "  ./run_batch_experiments.sh --r18                           # 只运行R18"
        echo "  ./run_batch_experiments.sh --r34                           # 只运行R34"
        echo "  ./run_batch_experiments.sh --r18 --r34                     # 运行R18+R34"
        echo "  ./run_batch_experiments.sh --custom cfg1.yaml cfg2.yaml    # 指定配置文件路径"
        echo "  ./run_batch_experiments.sh --keys rt-detr-r18 moe6-r34     # 通过键名选择"
        echo "  ./run_batch_experiments.sh --select                        # 交互式选择"
        echo "  ./run_batch_experiments.sh --rerun-failed [LOG_DIR]        # 重跑失败实验"
        exit 1
    fi
}

# 运行单个实验
run_single_experiment() {
    local config_path=$1
    local exp_name=$(basename "$config_path" .yaml)
    local exp_dir=$(dirname "$config_path")
    
    TOTAL_EXPERIMENTS=$((TOTAL_EXPERIMENTS + 1))
    
    echo ""
    echo -e "${PURPLE}========================================${NC}"
    echo -e "${PURPLE}实验 [$TOTAL_EXPERIMENTS/${#CONFIGS_TO_RUN[@]}]: $exp_name${NC}"
    echo -e "${PURPLE}========================================${NC}"
    log_info "开始实验: $config_path"
    
    # 检查配置文件是否存在
    if [ ! -f "$config_path" ]; then
        log_error "配置文件不存在: $config_path"
        SKIPPED_EXPERIMENTS=$((SKIPPED_EXPERIMENTS + 1))
        return 1
    fi
    
    # 确定训练脚本路径
    if [[ "$exp_dir" == *"yolov8"* ]]; then
        TRAIN_SCRIPT="yolov8/train.py"
        WORK_DIR="yolov8"
    elif [[ "$exp_dir" == *"dset"* ]]; then
        TRAIN_SCRIPT="dset/train.py"
        WORK_DIR="dset"
    elif [[ "$exp_dir" == *"rt-detr"* ]] && [[ "$exp_dir" != *"moe"* ]]; then
        TRAIN_SCRIPT="rt-detr/train.py"
        WORK_DIR="rt-detr"
    else
        TRAIN_SCRIPT="moe-rtdetr/train.py"
        WORK_DIR="moe-rtdetr"
    fi
    
    # 记录开始时间
    local start_time=$(date +%s)
    
    # 清理GPU缓存（在实验开始前清理，避免前一个实验的内存残留）
    if command -v python3 &> /dev/null; then
        python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
        sleep 1  # 给GPU一点时间释放内存
    fi
    
    # 运行训练（让训练脚本自己管理日志输出）
    cd "$WORK_DIR"
    set +e  # 临时允许错误
    
    # 如果是测试模式，传递--epochs 2参数
    if [ "$TEST_MODE" = true ]; then
        python train.py --config "../$config_path" --epochs 2
    else
        python train.py --config "../$config_path"
    fi
    
    local exit_code=$?
    set -e
    cd ..
    
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
        log_success "✓ 实验完成: $exp_name (耗时: $duration_formatted)"
        SUCCESSFUL_EXPERIMENTS=$((SUCCESSFUL_EXPERIMENTS + 1))
        echo "$exp_name,SUCCESS,$duration_formatted,$(date '+%Y-%m-%d %H:%M:%S')" >> "$BATCH_LOG_DIR/results.csv"
    else
        log_error "✗ $exp_name | 失败 (退出码: $exit_code, 耗时: $duration_formatted)"
        FAILED_EXPERIMENTS=$((FAILED_EXPERIMENTS + 1))
        echo "$exp_name,FAILED,$duration_formatted,$(date '+%Y-%m-%d %H:%M:%S')" >> "$BATCH_LOG_DIR/results.csv"
    fi
    
    return $exit_code
}

# 生成最终报告
generate_report() {
    local report_file="$BATCH_LOG_DIR/report.txt"
    
    {
        echo "========================================="
        echo "批量实验最终报告"
        echo "========================================="
        echo ""
        echo "完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "日志目录: $BATCH_LOG_DIR"
        echo ""
        echo "实验统计:"
        echo "  总实验数:   $TOTAL_EXPERIMENTS"
        echo "  成功:       $SUCCESSFUL_EXPERIMENTS"
        echo "  失败:       $FAILED_EXPERIMENTS"
        echo "  跳过:       $SKIPPED_EXPERIMENTS"
        echo ""
        
        if [ $SUCCESSFUL_EXPERIMENTS -gt 0 ]; then
            echo "✓ 成功的实验:"
            # 使用awk解析CSV（处理可能包含逗号的字段）
            awk -F',' '
            NR>1 && $2=="SUCCESS" {
                printf "  - %s (耗时: %s)\n", $1, $3
            }
            ' "$BATCH_LOG_DIR/results.csv" 2>/dev/null
            echo ""
        fi
        
        if [ $FAILED_EXPERIMENTS -gt 0 ]; then
            echo "✗ 失败的实验:"
            grep "FAILED" "$BATCH_LOG_DIR/results.csv" 2>/dev/null | while IFS=',' read -r name status duration timestamp; do
                echo "  - $name"
            done
            echo ""
        fi
        
        echo ""
        echo "CSV结果: $BATCH_LOG_DIR/results.csv"
        echo "========================================="
    } | tee "$report_file"
    
    # 彩色输出到终端
    echo ""
    echo -e "${CYAN}=========================================${NC}"
    echo -e "${CYAN}批量实验完成！${NC}"
    echo -e "${CYAN}=========================================${NC}"
    echo -e "${GREEN}成功: $SUCCESSFUL_EXPERIMENTS${NC} | ${RED}失败: $FAILED_EXPERIMENTS${NC} | ${YELLOW}跳过: $SKIPPED_EXPERIMENTS${NC}"
    echo ""
    echo -e "${BLUE}提示: 实验结果（包括mAP等指标）已保存在各训练脚本生成的日志目录中${NC}"
    echo -e "${BLUE}      - RT-DETR日志: rt-detr/logs/${NC}"
    echo -e "${BLUE}      - MOE-RTDETR日志: moe-rtdetr/logs/${NC}"
    echo -e "${BLUE}      - DSET日志: dset/logs/${NC}"
    echo -e "${BLUE}      - YOLOv8日志: yolov8/logs/${NC}"
    echo ""
    echo -e "${BLUE}完整报告: $report_file${NC}"
    echo -e "${BLUE}CSV结果: $BATCH_LOG_DIR/results.csv${NC}"
}

# 主函数
main() {
    echo -e "${CYAN}"
    echo "╔════════════════════════════════════════════╗"
    echo "║     RT-DETR 批量实验运行系统              ║"
    echo "║     方案2: 鲁棒批量运行                   ║"
    echo "╚════════════════════════════════════════════╝"
    echo -e "${NC}"
    
    # 解析参数
    parse_arguments "$@"
    
    # 显示将要运行的实验
    echo -e "${YELLOW}将运行以下 ${#CONFIGS_TO_RUN[@]} 个实验:${NC}"
    local i=1
    for config in "${CONFIGS_TO_RUN[@]}"; do
        echo -e "  ${CYAN}[$i]${NC} $config"
        ((i++))
    done
    echo ""
    
    # 如果是测试模式，显示额外提示
    if [ "$TEST_MODE" = true ]; then
        echo -e "${YELLOW}⚠️  测试模式：所有配置将只运行2个epoch进行快速验证${NC}"
        echo -e "${YELLOW}   预计总耗时: ~30-60分钟（${#CONFIGS_TO_RUN[@]}个实验 × 2 epochs）${NC}"
        echo -e "${YELLOW}   验证通过后，可使用以下命令运行完整训练：${NC}"
        echo -e "${CYAN}   ./run_batch_experiments.sh                # 运行所有实验（完整epochs）${NC}"
        echo ""
    fi
    
    # 确认开始
    read -p "是否开始批量实验? [Y/n] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]] && [[ -n $REPLY ]]; then
        log_warning "用户取消操作"
        exit 0
    fi
    
    # 初始化CSV结果文件
    echo "实验名称,状态,耗时,完成时间" > "$BATCH_LOG_DIR/results.csv"
    
    # 记录全局开始时间
    local global_start_time=$(date +%s)
    
    # 顺序运行所有实验
    for config in "${CONFIGS_TO_RUN[@]}"; do
        run_single_experiment "$config" || true  # 失败继续
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

