#!/bin/bash
#
# 自动读取 JSON 配置并运行 generate_benchmark_table.py
# 默认使用 generate_benchmark_table_dset.json
#
# 用法:
#   ./run_benchmark.sh                    # 使用默认 JSON
#   ./run_benchmark.sh other.json         # 指定 JSON 文件
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEFAULT_JSON="${SCRIPT_DIR}/generate_benchmark_table_dset.json"
JSON_PATH="${1:-$DEFAULT_JSON}"

# 相对路径转绝对路径
[[ "$JSON_PATH" != /* ]] && JSON_PATH="${SCRIPT_DIR}/${JSON_PATH}"

if [[ ! -f "$JSON_PATH" ]]; then
    echo "错误: JSON 文件不存在: $JSON_PATH"
    exit 1
fi

echo "使用配置: $JSON_PATH"
echo ""

cd "$PROJECT_ROOT" || exit 1
python3 experiments/analysis/generate_benchmark_table.py --models_config "$JSON_PATH"
