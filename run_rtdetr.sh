#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="rtdetr:latest"
NAME="rtdetr_dev"
# 主机项目目录：默认同脚本所在仓库根；可用 RTDETR_WORKDIR 覆盖
WORKDIR="${RTDETR_WORKDIR:-$SCRIPT_DIR}"
DOCKER_BUILD_DIR="$SCRIPT_DIR/containers/rtdetr"
# 把「主机上 CaS_DETR 的父目录」挂到容器内 /root/autodl-tmp
HOST_ROOT="$(cd "$WORKDIR/.." && pwd)"
CONTAINER_WORKDIR="/root/autodl-tmp/CaS_DETR"
PORT_JUPYTER=8899
PORT_TENSORBOARD=6007
SHM_SIZE="4g"

export OMP_NUM_THREADS=1

# 无真实终端时去掉 -t，否则报错：the input device is not a TTY
if [[ -t 0 && -t 1 ]]; then
  DOCKER_TTY=(-it)
else
  DOCKER_TTY=(-i)
  echo "[INFO] 当前无 TTY（例如 IDE 后台运行），将不分配伪终端；进入 shell 请在本机终端执行："
  echo "  docker exec -it -w $CONTAINER_WORKDIR $NAME bash"
fi

mkdir -p "$WORKDIR"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "[ERROR] 镜像 $IMAGE 不存在。先构建："
  echo "  docker build -t $IMAGE \"$DOCKER_BUILD_DIR\""
  exit 1
fi

if docker ps -a --format '{{.Names}}' | grep -wq "$NAME"; then
  if ! docker ps --format '{{.Names}}' | grep -wq "$NAME"; then
    echo "[INFO] 启动容器 $NAME ..."
    docker start "$NAME" >/dev/null
  fi
  exec docker exec "${DOCKER_TTY[@]}" -w "$CONTAINER_WORKDIR" "$NAME" bash
fi

echo "[INFO] 创建并进入容器：$NAME"
exec docker run --gpus all "${DOCKER_TTY[@]}" --name "$NAME" \
  --shm-size="$SHM_SIZE" \
  -v "$HOST_ROOT:/root/autodl-tmp" \
  -w "$CONTAINER_WORKDIR" \
  -p "$PORT_JUPYTER:8888" -p "$PORT_TENSORBOARD:6006" \
  "$IMAGE"
