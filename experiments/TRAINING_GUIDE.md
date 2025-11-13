# 🚀 训练指南

## 快速开始

### RT-DETR 训练

```bash
cd experiments/rt-detr/

# 方法1: 使用默认配置 (PResNet50)
./run_training.sh

# 方法2: 指定配置文件
./run_training.sh configs/rtdetr_presnet50.yaml

# 方法3: 使用轻量级模型
./run_training.sh configs/rtdetr_presnet18.yaml

# 方法4: 使用高精度模型
./run_training.sh configs/rtdetr_presnet101.yaml
```

### MOE-RTDETR 训练

```bash
cd experiments/moe-rtdetr/

# 方法1: 使用默认配置 (6专家 + PResNet50)
./run_training.sh

# 方法2: 指定配置文件
./run_training.sh configs/moe6_presnet50.yaml

# 方法3: 使用3专家配置
./run_training.sh configs/moe3_presnet50.yaml

# 方法4: 轻量级6专家模型
./run_training.sh configs/moe6_presnet18.yaml
```

## 配置文件列表

### RT-DETR (4个配置)

| 配置文件 | Backbone | Batch Size | 训练时间 | 推荐场景 |
|---------|----------|-----------|---------|---------|
| `rtdetr_presnet18.yaml` | PResNet18 | 128 | 最快 | 快速原型 |
| `rtdetr_presnet34.yaml` | PResNet34 | 96 | 较快 | 平衡选择 |
| `rtdetr_presnet50.yaml` | PResNet50 | 80 | 标准 | **默认推荐** |
| `rtdetr_presnet101.yaml` | PResNet101 | 64 | 较慢 | 最高精度 |

### MOE-RTDETR (8个配置)

#### 6专家配置 (Config A)

| 配置文件 | Backbone | Batch Size | 推荐场景 |
|---------|----------|-----------|---------|
| `moe6_presnet18.yaml` | PResNet18 | 96 | 轻量级MoE |
| `moe6_presnet34.yaml` | PResNet34 | 80 | 平衡MoE |
| `moe6_presnet50.yaml` | PResNet50 | 64 | **默认推荐** |
| `moe6_presnet101.yaml` | PResNet101 | 48 | 最高精度 |

#### 3专家配置 (Config B)

| 配置文件 | Backbone | Batch Size | 推荐场景 |
|---------|----------|-----------|---------|
| `moe3_presnet18.yaml` | PResNet18 | 112 | 快速MoE |
| `moe3_presnet34.yaml` | PResNet34 | 96 | 平衡MoE |
| `moe3_presnet50.yaml` | PResNet50 | 80 | 推荐MoE |
| `moe3_presnet101.yaml` | PResNet101 | 64 | 高精度MoE |

## 命令行参数覆盖

所有配置文件的参数都可以通过命令行覆盖：

```bash
# 修改训练轮数
./run_training.sh configs/rtdetr_presnet50.yaml --epochs 100

# 修改批次大小
./run_training.sh configs/rtdetr_presnet50.yaml --batch_size 64

# 修改随机种子
./run_training.sh configs/rtdetr_presnet50.yaml --seed 3407

# 多个参数
./run_training.sh configs/moe6_presnet50.yaml \
    --epochs 150 \
    --batch_size 48 \
    --seed 42 \
    --top_k 2
```

## 批量训练脚本

### 训练所有 RT-DETR 配置

```bash
#!/bin/bash
cd experiments/rt-detr/

for config in configs/rtdetr_presnet*.yaml
do
    echo "Training with $config"
    ./run_training.sh $config
done
```

### 训练所有 MOE6 配置

```bash
#!/bin/bash
cd experiments/moe-rtdetr/

for config in configs/moe6_presnet*.yaml
do
    echo "Training with $config"
    ./run_training.sh $config
done
```

### 公平对比实验

```bash
#!/bin/bash
# 对比 RT-DETR vs MOE-RTDETR (相同backbone，相同seed)

SEED=42
BACKBONE="presnet50"

echo "=== 训练 RT-DETR ===" 
cd experiments/rt-detr/
./run_training.sh configs/rtdetr_${BACKBONE}.yaml --seed $SEED

echo "=== 训练 MOE-RTDETR (6专家) ==="
cd ../moe-rtdetr/
./run_training.sh configs/moe6_${BACKBONE}.yaml --seed $SEED

echo "=== 训练 MOE-RTDETR (3专家) ==="
./run_training.sh configs/moe3_${BACKBONE}.yaml --seed $SEED
```

## 恢复训练

从检查点恢复训练：

```bash
# RT-DETR
./run_training.sh configs/rtdetr_presnet50.yaml \
    --resume_from_checkpoint logs/rtdetr_20250101_120000/latest_checkpoint.pth

# MOE-RTDETR
./run_training.sh configs/moe6_presnet50.yaml \
    --resume_from_checkpoint logs/moe_rtdetr_20250101_120000/latest_checkpoint.pth
```

## 使用预训练权重

```bash
# 确保预训练权重文件存在
ls pretrained/

# 使用配置文件中指定的预训练权重
./run_training.sh configs/rtdetr_presnet50.yaml

# 或通过命令行覆盖
./run_training.sh configs/rtdetr_presnet50.yaml \
    --pretrained_weights pretrained/custom_weights.pth
```

## 监控训练

### 查看实时日志

```bash
# RT-DETR
tail -f logs/rtdetr_*/training.log

# MOE-RTDETR
tail -f logs/moe_rtdetr_*/training.log
```

### 查看训练曲线

训练过程中会自动生成训练曲线图：

```bash
# RT-DETR
ls logs/rtdetr_*/training_curves.png

# MOE-RTDETR
ls logs/moe_rtdetr_*/training_curves.png
```

## GPU显存不足处理

如果遇到OOM错误，按以下优先级调整：

1. **降低batch size**
```bash
./run_training.sh configs/rtdetr_presnet50.yaml --batch_size 48
```

2. **使用更小的模型**
```bash
./run_training.sh configs/rtdetr_presnet34.yaml
```

3. **减少专家数（MOE-RTDETR）**
```bash
# 从6专家改为3专家
./run_training.sh configs/moe3_presnet50.yaml
```

## 训练时间预估

基于 vGPU 48G，DAIR-V2X数据集 (200 epochs)：

| 配置 | 训练时间 |
|------|---------|
| RT-DETR + PResNet18 | ~6h |
| RT-DETR + PResNet50 | ~10h |
| RT-DETR + PResNet101 | ~16h |
| MOE6 + PResNet18 | ~9h |
| MOE6 + PResNet50 | ~14h |
| MOE6 + PResNet101 | ~22h |
| MOE3 + PResNet50 | ~12h |

## 常见问题

### Q1: 配置文件找不到？

```bash
# 检查配置文件列表
ls configs/

# 应该看到所有配置文件
cd experiments/rt-detr/
ls configs/rtdetr_*.yaml

cd experiments/moe-rtdetr/
ls configs/moe*.yaml
```

### Q2: 如何查看所有可用参数？

```bash
python train.py --help
```

### Q3: 如何使用确定性模式？

```bash
./run_training.sh configs/rtdetr_presnet50.yaml --deterministic
```

注意：确定性模式会降低训练速度10-30%。

## 推荐训练流程

### 步骤1: 快速验证（使用轻量级模型）

```bash
# 先用小模型跑几个epoch验证代码和数据
./run_training.sh configs/rtdetr_presnet18.yaml --epochs 5
```

### 步骤2: 标准训练（推荐配置）

```bash
# 验证通过后，使用标准配置训练完整模型
./run_training.sh configs/rtdetr_presnet50.yaml
```

### 步骤3: 对比实验（可选）

```bash
# 对比不同架构
./run_training.sh configs/rtdetr_presnet50.yaml --seed 42
./run_training.sh configs/moe6_presnet50.yaml --seed 42
```

### 步骤4: 精度优化（可选）

```bash
# 使用更大模型追求最高精度
./run_training.sh configs/rtdetr_presnet101.yaml
```

