#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="yolov8:latest"
NAME="yolov8_dev"
# 默认同本仓库根目录；可用 YOLOV8_WORKDIR 覆盖
WORKDIR="${YOLOV8_WORKDIR:-$SCRIPT_DIR}"
PORT_JUPYTER=8888
PORT_TENSORBOARD=6006
SHM_SIZE="4g"

# 解决 32G 环境下 Dataloader 多进程与 OpenMP 线程池冲突导致的 libgomp 报错
export OMP_NUM_THREADS=1

mkdir -p "$WORKDIR"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "[ERROR] 镜像 $IMAGE 不存在，请先构建："
  echo "  docker build -t $IMAGE \$HOME/containers/tsdet"
  exit 1
fi

if docker ps -a --format '{{.Names}}' | grep -wq "$NAME"; then
  echo "[INFO] 容器已存在：$NAME"
  if ! docker ps --format '{{.Names}}' | grep -wq "$NAME"; then
    echo "[INFO] 启动容器..."
    docker start "$NAME" >/dev/null
  fi
  exec docker exec -it "$NAME" bash
fi

echo "[INFO] 创建并进入容器：$NAME"
exec docker run --gpus all -it --name "$NAME" \
  --shm-size="$SHM_SIZE" \
  -v "$WORKDIR:/workspace" \
  -p "$PORT_JUPYTER:8888" -p "$PORT_TENSORBOARD:6006" \
  "$IMAGE"
