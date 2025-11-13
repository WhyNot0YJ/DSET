# DSET: Dual-Sparse Expert Transformer for V2X Object Detection

## 📋 目录

- [概述](#概述)
- [核心创新点](#核心创新点)
- [架构设计](#架构设计)
- [环境配置](#环境配置)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [技术细节](#技术细节)
- [代码逻辑验证](#代码逻辑验证)
- [训练技巧](#训练技巧)
- [测试流程](#测试流程)
- [故障排除](#故障排除)
- [实验结果](#实验结果)
- [常见问题](#常见问题)

---

## 概述

**DSET (Dual-Sparse Expert Transformer)** 是一种高效的目标检测模型，专门为V2X路测单元中的交通参与者识别设计。通过结合**Token Pruning**和**Patch-MoE**两种稀疏机制，DSET在保持高精度的同时显著降低了计算复杂度。

### 核心创新点

#### 1. Token Pruning（Token剪枝）
- 在Encoder输入前使用**可学习的重要性预测器**评估token重要性
- 剪枝冗余tokens，减少进入Transformer的token数量（默认保留70%）
- 支持**渐进式训练**：从不剪枝逐渐过渡到目标剪枝比例
- 理论计算量减少：**30%**

#### 2. Patch-MoE（空间专家混合）
- 在Encoder的FFN层使用Mixture-of-Experts
- 每个token动态选择少数几个专家处理（top-2）
- 针对空间特征的稀疏专家激活
- 理论计算量减少：**50%**

#### 3. Decoder MoE（解码器专家混合）
- Decoder FFN层的自适应专家混合
- 支持多种专家数量和top-k配置
- 增强模型表达能力的同时保持高效

**双稀疏协同效果**：理论总计算量降至 **0.7 × 0.5 = 35%**

---

## 架构设计

### 整体架构图

```
输入图像 [B, 3, 640, 640]
   ↓
┌─────────────────────────────────┐
│  Backbone (PResNet/HGNetv2)     │
│  提取多尺度特征                  │
└─────────────────────────────────┘
   ↓
┌─────────────────────────────────┐
│  Token Pruning                  │
│  - 可学习重要性预测器            │
│  - 保留top 70%重要tokens        │
│  - 渐进式启用（warmup 10 epoch）│
└─────────────────────────────────┘
   ↓ [B, 0.7*HW, C]
┌─────────────────────────────────┐
│  HybridEncoder (DSET)           │
│  ├─ Input Projection            │
│  ├─ Patch-MoE Transformer       │
│  │   ├─ Self-Attention          │
│  │   └─ Patch-MoE FFN           │
│  │       (4 experts, top-2)     │
│  ├─ FPN融合                     │
│  └─ PAN融合                     │
└─────────────────────────────────┘
   ↓
┌─────────────────────────────────┐
│  RT-DETR Decoder (with MoE)     │
│  ├─ Self-Attention              │
│  ├─ Cross-Attention             │
│  └─ Adaptive Expert FFN         │
│      (6 experts, top-3)         │
└─────────────────────────────────┘
   ↓
┌─────────────────────────────────┐
│  Detection Head                 │
│  输出: Boxes + Class Scores     │
└─────────────────────────────────┘
```

### 双稀疏工作流程

```
训练时:
1. Backbone提取特征
2. Token Pruning评估并剪枝（epoch < 10时渐进式）
3. Patch-MoE处理保留的tokens
4. Decoder MoE生成检测结果
5. 计算损失：
   - Detection Loss（主损失）
   - Decoder MoE Balance Loss
   - Encoder MoE Balance Loss
   - Token Pruning Loss（可选）

推理时:
1. Backbone提取特征
2. Token Pruning剪枝
3. Patch-MoE处理
4. Decoder MoE生成结果
5. 直接输出（无损失计算）
```

---

## 环境配置

### 系统要求

- Python >= 3.8
- PyTorch >= 1.10
- CUDA >= 11.0 (推荐)
- GPU: 至少8GB显存（训练），4GB显存（推理）

### 依赖安装

```bash
cd dual-moe-rtdetr
pip install -r requirements.txt
```

主要依赖：
```
torch>=1.10.0
torchvision>=0.11.0
pycocotools
pyyaml
numpy
opencv-python
matplotlib
```

### 数据集准备

DAIR-V2X数据集目录结构：

```
datasets/DAIR-V2X/
├── cooperative-vehicle-infrastructure/
│   ├── vehicle-side/
│   │   ├── image/
│   │   │   ├── 000001.jpg
│   │   │   └── ...
│   │   └── label/
│   │       ├── 000001.json
│   │       └── ...
│   └── infrastructure-side/
│       ├── image/
│       └── label/
└── ...
```

---

## 快速开始

### 基础训练

```bash
# 使用PResNet50（推荐配置）
python train.py --config configs/dset_presnet50.yaml

# 使用PResNet18（轻量级）
python train.py --config configs/dset_presnet18.yaml
```

### 命令行参数训练

```bash
python train.py \
  --backbone presnet50 \
  --data_root datasets/DAIR-V2X \
  --epochs 200 \
  --batch_size 32 \
  --pretrained_lr 1e-5 \
  --new_lr 1e-4
```

### 断点续训

```bash
python train.py \
  --config configs/dset_presnet50.yaml \
  --resume_from_checkpoint logs/dset_rtdetr_r50_20240101_120000/latest_checkpoint.pth
```

### 小规模测试（推荐首次运行）

```bash
# 测试2个epoch，确认代码运行正常
python train.py \
  --config configs/dset_presnet50.yaml \
  --epochs 2 \
  --batch_size 4
```

---

## 配置说明

### DSET双稀疏配置

配置文件示例（`configs/dset_presnet50.yaml`）：

```yaml
model:
  # Decoder MoE配置
  num_experts: 6           # Decoder专家数量
  top_k: 3                 # Decoder top-k选择
  
  # DSET双稀疏配置
  dset:
    # Token Pruning配置
    use_token_pruning: true
    token_keep_ratio: 0.7                # 保留70%的tokens
    token_pruning_warmup_epochs: 10      # 渐进式启用
    
    # Patch-MoE配置
    use_patch_moe: true
    patch_moe_num_experts: 4             # Encoder专家数量
    patch_moe_top_k: 2                   # Encoder top-k
```

### 关键参数详解

| 参数 | 描述 | 推荐值 | 影响 |
|------|------|--------|------|
| `token_keep_ratio` | Token保留比例 | 0.6-0.7 | 值越小，剪枝越激进，计算量越低但可能影响精度 |
| `token_pruning_warmup_epochs` | Token Pruning warmup | 10 | 值越大，训练越稳定但收敛可能较慢 |
| `patch_moe_num_experts` | Patch-MoE专家数 | 4 | 影响Encoder表达能力和计算量 |
| `patch_moe_top_k` | Patch-MoE top-k | 2 | 值越大，激活专家越多，计算量越大 |
| `num_experts` | Decoder MoE专家数 | 6 | 影响Decoder表达能力 |
| `top_k` | Decoder MoE top-k | 3 | 建议>=2，避免专家退化 |
| `moe_balance_weight` | MoE负载均衡权重 | 0.05 | 代码中自动调整，一般无需修改 |

### 配置文件对比

| 配置 | Backbone | Token Keep | Patch-MoE | Decoder MoE | 适用场景 |
|------|----------|------------|-----------|-------------|----------|
| `dset_presnet50.yaml` | PResNet50 | 0.7 (4 experts, top-2) | 6 experts, top-3 | 标准训练 | 平衡精度和效率 |
| `dset_presnet18.yaml` | PResNet18 | 0.6 (2 experts, top-1) | 3 experts, top-2 | 轻量级 | 资源受限场景 |

---

## 技术细节

### Token Pruning模块

**位置**: `src/zoo/rtdetr/token_pruning.py`

**核心组件**:

1. **LearnableImportancePredictor**
   - 轻量级MLP（256 -> 128 -> 1）
   - 预测每个token的重要性分数
   - 可学习，通过反向传播优化

2. **TokenPruner**
   - 基于重要性分数选择top-k tokens
   - 渐进式剪枝策略：
     ```python
     progress = (epoch - warmup_epochs) / warmup_epochs
     current_ratio = 1.0 - progress * (1.0 - keep_ratio)
     ```
   - 保持空间顺序（对indices排序）

3. **SpatialTokenPruner** (可选)
   - 考虑空间先验（中心/边缘权重）
   - 适用于特定应用场景

**关键实现**:
```python
# 1. 预测重要性
importance_scores = importance_predictor(tokens)  # [B, N]

# 2. 选择top-k
_, top_indices = torch.topk(importance_scores, num_keep, dim=-1)

# 3. 排序保持空间顺序
top_indices_sorted, _ = torch.sort(top_indices, dim=-1)

# 4. 收集保留的tokens
pruned_tokens = tokens[batch_indices, top_indices_sorted]
```

### Patch-MoE模块

**位置**: `src/zoo/rtdetr/moe_components.py`

**核心组件**:

1. **AdaptiveRouter**
   - 线性层：hidden_dim -> num_experts
   - Softmax + Top-K选择
   - 权重归一化

2. **SpecialistNetwork**
   - 标准两层FFN
   - d_model -> dim_feedforward -> d_model
   - 支持多种激活函数（ReLU/GELU/SiLU）

3. **PatchMoELayer**
   - 整合Router + Experts
   - 稀疏激活（只计算top-k专家）
   - 加权融合输出

**关键实现**:
```python
# 1. 路由决策
router_probs = F.softmax(router_logits, dim=-1)
expert_weights, expert_indices = torch.topk(router_probs, top_k, dim=-1)

# 2. 稀疏计算
for expert_id in unique_experts:
    expert_mask = (expert_indices == expert_id).any(dim=-1)
    expert_output = experts[expert_id](tokens[expert_mask])
    output[expert_mask] += expert_output * weights[expert_mask]
```

### HybridEncoder集成

**位置**: `src/zoo/rtdetr/hybrid_encoder.py`

**主要修改**:

1. **TransformerEncoderLayer**
   - 添加`use_moe`参数
   - FFN层可选择Patch-MoE或标准FFN
   - 缓存router信息用于负载均衡损失

2. **HybridEncoder.forward()**
   - Token Pruning在encoder前执行
   - 位置编码选择对应的kept tokens
   - 特征图恢复（zero-padding策略）
   - 返回encoder_info（包含统计信息）

3. **set_epoch()方法**
   - 传递epoch到所有token_pruners
   - 支持渐进式训练

**特征图恢复策略**:
```python
# 剪枝的位置用0填充
full_memory = torch.zeros(B, H*W, C, device=memory.device)
full_memory[batch_idx, kept_indices] = memory
# 在后续FPN/PAN中通过卷积融合得到补偿
```

### 损失函数设计

**总损失**:
```python
total_loss = detection_loss + 
             moe_balance_weight * (decoder_moe_loss + encoder_moe_loss) +
             0.001 * token_pruning_loss
```

**各损失说明**:

1. **Detection Loss** (主损失)
   - Hungarian matching
   - VFL + BBox + GIoU
   - 权重最大

2. **MoE Balance Loss**
   - 确保专家均衡使用
   - Switch Transformer风格：`num_experts * sum(f_i * P_i)`
   - Decoder和Encoder分别计算

3. **Token Pruning Loss** (辅助)
   - 稀疏性约束
   - 多样性约束
   - 权重很小（0.001）

---

## 代码逻辑验证

### ✅ 已验证的核心逻辑

#### 1. Token Pruning模块
- ✅ 渐进式剪枝策略正确实现
- ✅ Token选择逻辑正确（topk + sort）
- ✅ 位置信息处理正确
- ⚠️ 特征图恢复策略较简单（zero-padding），在实际训练中验证效果

#### 2. Patch-MoE模块
- ✅ 路由器逻辑正确（Softmax + topk + 归一化）
- ✅ 专家稀疏计算正确
- ✅ 形状处理正确（支持多种输入格式）
- ✅ 负载均衡损失计算正确

#### 3. HybridEncoder集成
- ✅ Token Pruning集成正确
- ✅ Patch-MoE集成正确
- ✅ 位置编码选择正确
- ✅ set_epoch方法正确传递

#### 4. DSET模型
- ✅ 模型初始化正确
- ✅ 前向传播逻辑正确
- ✅ 损失计算正确
- ✅ 损失权重动态调整合理

#### 5. Trainer
- ✅ 配置加载正确
- ✅ 渐进式训练正确实现
- ✅ 损失统计正确
- ✅ 日志输出完整

### 🔍 关键代码路径

**训练时前向传播**:
```
Backbone(images)
  ↓
HybridEncoder.forward(feats, return_encoder_info=True)
  ├─ Input Projection
  ├─ Token Pruning (if enabled, epoch > warmup)
  │   └─ 保留70% tokens
  ├─ Transformer Encoder with Patch-MoE
  │   ├─ Self-Attention
  │   └─ Patch-MoE FFN (4 experts, top-2)
  ├─ 特征图恢复（zero-padding）
  ├─ FPN/PAN融合
  └─ return (outs, encoder_info)
  ↓
RTDETRTransformerv2 (with Decoder MoE)
  ├─ Input Processing
  ├─ Decoder Layers (6 experts, top-3)
  └─ Detection Head
  ↓
Loss Computation
  ├─ Detection Loss
  ├─ Decoder MoE Loss
  ├─ Encoder MoE Loss
  └─ Token Pruning Loss
```

---

## 训练技巧

### 1. 渐进式剪枝策略

**推荐设置**:
```yaml
dset:
  token_pruning_warmup_epochs: 10
```

**工作原理**:
- Epoch 0-10: 剪枝比例从0%逐渐增加到30%
- Epoch 11+: 稳定在30%剪枝

**预期行为**:
```
Epoch 0:  keep_ratio = 1.0    (不剪枝)
Epoch 5:  keep_ratio = 0.85   (15%剪枝)
Epoch 10: keep_ratio = 0.7    (30%剪枝)
Epoch 15: keep_ratio = 0.7    (稳定)
```

### 2. MoE负载均衡

**自动调整机制**:
```python
if top_k == 1:
    balance_weight = 0.1  # 更强约束
else:
    balance_weight = 0.05  # 适度约束
```

**监控指标**:
- 理想：各专家使用率接近均匀（6个专家各约16.7%）
- 警告：某专家使用率 > 50%（MoE退化）
- 解决：增加balance_weight或top_k

### 3. 学习率设置

**差异化学习率**:
```yaml
training:
  pretrained_lr: 1e-5    # Backbone + Encoder
  new_lr: 1e-4           # MoE + Pruning组件
```

**原因**:
- 预训练部分需要微调（小学习率）
- 新增部分需要充分训练（大学习率）

### 4. 数据增强

**Mosaic增强**:
```yaml
training:
  use_mosaic: true
```
提升模型鲁棒性，特别是对小目标检测

### 5. 梯度裁剪

```yaml
training:
  clip_max_norm: 10.0
```
防止梯度爆炸，确保训练稳定

---

## 测试流程

### 阶段1：基础功能测试

```bash
# 1. 测试配置文件加载
python -c "import yaml; config=yaml.safe_load(open('configs/dset_presnet50.yaml')); print('Config OK')"

# 2. 测试模型创建
python -c "from train import DSETRTDETR; model=DSETRTDETR(); print('Model OK')"

# 3. 运行测试脚本
python test_dset.py
```

**预期输出**:
```
============================================================
测试DSET模型
============================================================
1. 创建DSET模型...
✓ 模型创建成功
  - Token Pruning: True
  - Patch-MoE: True
  - Decoder MoE: 6 experts

2. 创建测试输入...
✓ 输入创建成功: images torch.Size([2, 3, 640, 640])

3. 测试推理模式（不剪枝）...
✓ 推理模式成功

4. 测试训练模式（Token Pruning + Patch-MoE）...
✓ 训练模式成功
  - total_loss: 12.3456
  - Token Pruning Ratios: ['25.00%']

5. 测试反向传播...
✓ 反向传播成功
✓ 梯度正常

============================================================
✓ 所有测试通过！DSET模型运行正常。
============================================================
```

### 阶段2：小规模训练测试

```bash
# 2个epoch，小batch，快速验证
python train.py --config configs/dset_presnet50.yaml --epochs 2 --batch_size 4
```

**观察要点**:
1. 没有运行时错误
2. Token Pruning Ratio逐渐增加
3. 损失正常下降
4. 专家使用率相对均衡

### 阶段3：完整训练

```bash
# 正式训练
python train.py --config configs/dset_presnet50.yaml
```

**监控指标**:
- Training Loss: 应持续下降
- mAP: 应逐渐提升
- Token Pruning Ratio: epoch 0-10从0%到30%，之后稳定
- Expert Usage Rate: 各专家接近16.7%

---

## 故障排除

### 错误1: `AttributeError: 'HybridEncoder' object has no attribute 'set_epoch'`

**原因**: Token Pruning未启用

**解决**:
```yaml
model:
  dset:
    use_token_pruning: true  # 确保启用
```

### 错误2: `RuntimeError: shape mismatch in pos_embed selection`

**原因**: Position embedding维度不匹配

**调试**:
```python
# 在hybrid_encoder.py中添加打印
print(f"pos_embed shape: {pos_embed.shape}")
print(f"kept_indices shape: {kept_indices.shape}")
print(f"src_flatten shape: {src_flatten.shape}")
```

**解决**: 确保pos_embed是[1, HW, C]或[B, HW, C]格式

### 错误3: `Loss is NaN`

**可能原因**:
1. 剪枝比例过高（keep_ratio < 0.5）
2. MoE balance weight过大
3. 学习率过大
4. 数据异常

**解决方案**:
```yaml
# 1. 降低剪枝强度
dset:
  token_keep_ratio: 0.7  # 或更高

# 2. 降低学习率
training:
  pretrained_lr: 5e-6
  new_lr: 5e-5

# 3. 增加warmup
training:
  warmup_epochs: 5
```

### 错误4: 专家使用严重不均衡

**现象**: 某个专家使用率 > 50%

**解决**:
```yaml
# 在代码中手动调整balance_weight
# train.py line 395-397
if hasattr(self.decoder, 'moe_top_k'):
    moe_balance_weight = 0.1  # 增加到0.1或0.15
```

### 错误5: CUDA Out of Memory

**解决**:
```yaml
# 1. 减小batch size
training:
  batch_size: 16  # 或更小

# 2. 减少worker数量
misc:
  num_workers: 4

# 3. 使用梯度累积（需修改代码）
```

### 错误6: 训练速度很慢

**检查**:
1. 是否使用了GPU？
2. num_workers是否合理？
3. 数据预处理是否成为瓶颈？

**优化**:
```yaml
misc:
  num_workers: 8          # 根据CPU核心数调整
  pin_memory: true        # 确保启用
  prefetch_factor: 2      # 预取因子
```

---

## 实验结果

### DAIR-V2X数据集

| 模型 | Backbone | mAP@0.5 | mAP@0.75 | mAP@[0.5:0.95] | FPS | 参数量 | 计算量 |
|------|----------|---------|----------|----------------|-----|--------|--------|
| RT-DETR | PResNet50 | - | - | - | - | - | 100% |
| MoE-RTDETR | PResNet50 | - | - | - | - | - | ~70% |
| **DSET** | **PResNet50** | **-** | **-** | **-** | **-** | **-** | **~35%** |

*注：实验结果将在训练完成后补充*

### 理论计算量分析

| 组件 | 计算量 | 说明 |
|------|--------|------|
| Token Pruning | 0.7× | 保留70% tokens |
| Patch-MoE (Encoder) | 0.5× | Top-2激活（4个专家中的2个） |
| Decoder MoE | 0.5× | Top-3激活（6个专家中的3个） |
| **总体** | **~35%** | 0.7 × 0.5 × 1.0 ≈ 35% |

### 预期训练曲线

```
Loss趋势:
├─ Detection Loss: 持续下降（主导）
├─ Decoder MoE Loss: 初期高（~2.0），后期稳定（~1.0）
├─ Encoder MoE Loss: 类似Decoder MoE
└─ Total Loss: 跟随Detection Loss

Token Pruning Ratio:
├─ Epoch 0-10: 0% → 30% (渐进式)
└─ Epoch 11+: 30% (稳定)

Expert Usage:
├─ 理想: 各专家 ~16.7% (6个专家)
└─ 可接受: 10%-25%范围内
```

---

## 常见问题

### Q1: Token Pruning会影响精度吗？

**A**: 适当的剪枝比例（0.6-0.7）通常不会显著影响精度，原因：
1. 冗余tokens被剪枝（如背景区域）
2. 重要tokens得到保留（如目标区域）
3. 渐进式训练确保稳定
4. FPN/PAN融合提供一定补偿

反而可能通过减少冗余信息提升泛化能力。

### Q2: Patch-MoE和Decoder MoE有什么区别？

**A**: 
| 维度 | Patch-MoE | Decoder MoE |
|------|-----------|-------------|
| 位置 | Encoder | Decoder |
| 处理对象 | 空间patch tokens | Query tokens |
| 专家数量 | 2-4个（较少） | 6个（较多） |
| Top-K | 1-2 | 2-3 |
| 作用 | 局部特征提取 | 目标级别建模 |

### Q3: 如何选择专家数量？

**A**: 
- **Encoder (Patch-MoE)**: 2-4个专家
  - 原因：Encoder处理空间特征，需要保持轻量
  - 推荐：4 experts, top-2
  
- **Decoder (MoE)**: 3-6个专家
  - 原因：Decoder需要更强表达能力
  - 推荐：6 experts, top-3

- **经验法则**: top_k ≥ 2，避免专家退化

### Q4: 训练不稳定怎么办？

**A**: 按顺序尝试：
1. **增加warmup epochs**
   ```yaml
   dset:
     token_pruning_warmup_epochs: 15  # 从10增加到15
   ```

2. **降低剪枝强度**
   ```yaml
   dset:
     token_keep_ratio: 0.75  # 从0.7增加到0.75
   ```

3. **增加MoE balance weight**
   ```python
   # train.py line 395
   moe_balance_weight = 0.1  # 从0.05增加到0.1
   ```

4. **降低学习率**
   ```yaml
   training:
     pretrained_lr: 5e-6
     new_lr: 5e-5
   ```

### Q5: 如何在自己的数据集上使用DSET？

**A**: 需要修改：
1. **数据集类** (`src/data/dataset/`)
   - 实现自定义Dataset类
   - 返回格式：images, targets
   
2. **类别数量** (`train.py`)
   ```python
   num_classes = your_num_classes  # 修改类别数
   ```
   
3. **配置文件**
   ```yaml
   data:
     data_root: "path/to/your/dataset"
   ```

### Q6: 推理速度如何？

**A**: DSET设计目标是在保持精度的同时提升推理速度：
- Token Pruning: 减少30% tokens → 加速encoder
- Patch-MoE: 稀疏激活 → 减少50% encoder FFN计算
- Decoder MoE: 稀疏激活 → 减少50% decoder FFN计算
- **预期**: 相比标准RT-DETR提速1.5-2×（待实测验证）

### Q7: 能否禁用某些稀疏机制？

**A**: 可以，配置灵活：

**只使用Token Pruning**:
```yaml
dset:
  use_token_pruning: true
  use_patch_moe: false
```

**只使用Patch-MoE**:
```yaml
dset:
  use_token_pruning: false
  use_patch_moe: true
```

**标准MoE（无双稀疏）**:
```yaml
dset:
  use_token_pruning: false
  use_patch_moe: false
# Decoder MoE仍然保留
```

---

## 引用

如果您使用了DSET，请引用：

```bibtex
@article{dset2024,
  title={DSET: Dual-Sparse Expert Transformer for Efficient V2X Object Detection},
  author={Your Name},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2024}
}
```

## 许可证

本项目基于RT-DETR开发，遵循相同的许可证。

## 致谢

- [RT-DETR](https://github.com/lyuwenyu/RT-DETR) - 基础检测框架
- [DAIR-V2X](https://thudair.baai.ac.cn/index) - 数据集
- Switch Transformer - MoE设计灵感
- Vision Transformer - Token Pruning灵感

---

## 项目结构

```
dual-moe-rtdetr/
├── configs/                      # 配置文件
│   ├── dset_presnet50.yaml      # PResNet50配置（推荐）
│   └── dset_presnet18.yaml      # PResNet18配置（轻量级）
├── src/
│   ├── zoo/rtdetr/
│   │   ├── token_pruning.py     # Token Pruning模块
│   │   ├── moe_components.py    # MoE组件（含Patch-MoE）
│   │   ├── hybrid_encoder.py    # DSET Encoder
│   │   ├── rtdetrv2_decoder.py  # RT-DETR Decoder (with MoE)
│   │   └── ...
│   ├── nn/backbone/             # Backbone网络
│   ├── data/                    # 数据加载
│   ├── optim/                   # 优化器和调度器
│   └── misc/                    # 可视化等工具
├── train.py                     # 训练脚本
├── test_dset.py                # 测试脚本
├── run_training.sh             # 启动脚本
├── README.md                   # 本文档
└── requirements.txt            # 依赖列表
```

---

## 联系方式

如有问题或建议，欢迎：
- 提Issue
- 发邮件
- 参与讨论

**最后更新**: 2024年11月

**状态**: ✅ 代码已验证，准备就绪

---

## 🚀 准备就绪检查清单

训练前确认：
- [ ] 数据集已准备并组织正确
- [ ] 依赖已安装（`pip install -r requirements.txt`）
- [ ] 配置文件已根据需求修改
- [ ] GPU可用且显存充足
- [ ] 已运行`test_dset.py`确认代码正常
- [ ] 已进行小规模测试（2 epochs）

**一切就绪，开始训练吧！** 🎉
