#!/usr/bin/env bash
set -euo pipefail

IMAGE="rtdetr:latest"           # 你用 docker build -t rtdetr:latest . 构建的镜像
NAME="rtdetr_dev"               # 容器名
WORKDIR="$HOME/proj/CaS_DETR"   # 主机项目目录
HOST_ROOT="$HOME/proj"
CONTAINER_WORKDIR="/root/autodl-tmp/CaS_DETR"
PORT_JUPYTER=8899               # 避免和 yolov8 冲突，给个不同端口
PORT_TENSORBOARD=6007
SHM_SIZE="4g"

mkdir -p "$WORKDIR"

# 镜像存在性检查
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "[ERROR] 镜像 $IMAGE 不存在。先构建："
  echo "  cd \$HOME/containers/rtdetr && docker build -t $IMAGE ."
  exit 1
fi

# 若容器已存在：启动/进入
if docker ps -a --format '{{.Names}}' | grep -wq "$NAME"; then
  if ! docker ps --format '{{.Names}}' | grep -wq "$NAME"; then
    echo "[INFO] 启动容器 $NAME ..."
    docker start "$NAME" >/dev/null
  fi
  exec docker exec -it -w "$CONTAINER_WORKDIR" "$NAME" bash
fi

# 首次创建并进入
echo "[INFO] 创建并进入容器：$NAME"
exec docker run --gpus all -it --name "$NAME" \
  --shm-size="$SHM_SIZE" \
  -v "$HOST_ROOT:/root/autodl-tmp" \
  -w "$CONTAINER_WORKDIR" \
  -p "$PORT_JUPYTER:8888" -p "$PORT_TENSORBOARD:6006" \
  "$IMAGE"
