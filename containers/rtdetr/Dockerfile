# RT-DETR / CaS_DETR experiments
# python:3.10-slim 拉自 daocloud 镜像，PyTorch cu124 wheel 自带 CUDA 运行库
FROM docker.m.daocloud.io/python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

RUN pip install --no-cache-dir \
    "faster-coco-eval>=1.6.6" \
    pycocotools \
    PyYAML \
    tensorboard \
    pandas \
    scipy \
    numpy \
    onnx \
    onnxruntime-gpu \
    opencv-python \
    Pillow \
    matplotlib \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

CMD ["/bin/bash"]
