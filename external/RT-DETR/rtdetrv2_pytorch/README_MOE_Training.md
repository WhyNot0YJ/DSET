# MOE RT-DETR DAIR-V2X训练指南

## 概述

这是一个完整的MOE（Mixture of Experts）RT-DETR训练系统，专门用于在DAIR-V2X数据集上进行任务选择性目标检测训练。

## 文件结构

```
rtdetrv2_pytorch/
├── train_moe_rtdetr_dair_v2x.py    # 主训练脚本
├── configs/
│   └── moe_rtdetr_dair_v2x.yml     # 训练配置文件
├── run_moe_training.sh             # 批量训练脚本
├── src/nn/arch/
│   └── moe_rtdetr.py               # MOE模型架构
├── src/nn/criterion/
│   └── moe_criterion.py            # MOE损失函数
└── README_MOE_Training.md          # 本文档
```

## 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install torch torchvision torchaudio
pip install pyyaml numpy matplotlib seaborn

# 设置环境变量
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### 2. 数据准备

确保DAIR-V2X数据集位于以下路径：
```
datasets/DAIR-V2X/
├── metadata/
│   └── data_info.json
├── image/
├── label/
└── calib/
```

### 3. 预训练权重准备

RT-DETR预训练权重会在训练时自动通过`torch.hub`下载，无需手动准备任何文件。

### 4. 训练模型

#### 方法1：使用启动脚本（推荐）

```bash
# 运行所有配置的训练
./run_moe_training.sh
```

#### 方法2：单独训练

```bash
# 训练配置A：6个专家（按类别）
python train_moe_rtdetr_dair_v2x.py \
    --config A \
    --data_root datasets/DAIR-V2X \
    --epochs 100 \
    --batch_size 16 \
    --pretrained_lr 1e-5 \
    --new_lr 1e-4 \
    --top_k 2 \
    --pretrained_weights torch_hub

# 训练配置B：3个专家（按复杂度）
python train_moe_rtdetr_dair_v2x.py \
    --config B \
    --data_root datasets/DAIR-V2X \
    --epochs 100 \
    --batch_size 16 \
    --pretrained_lr 1e-5 \
    --new_lr 1e-4 \
    --top_k 2 \
    --pretrained_weights torch_hub

# 训练配置C：3个专家（按尺寸）
python train_moe_rtdetr_dair_v2x.py \
    --config C \
    --data_root datasets/DAIR-V2X \
    --epochs 100 \
    --batch_size 16 \
    --pretrained_lr 1e-5 \
    --new_lr 1e-4 \
    --top_k 2 \
    --pretrained_weights torch_hub
```

## 配置说明

### MOE配置

#### 配置A：6个专家（按类别）
- 专家0：car
- 专家1：truck
- 专家2：bus
- 专家3：person
- 专家4：bicycle
- 专家5：motorcycle

#### 配置B：3个专家（按复杂度）
- 专家0：vehicles (car, truck, bus)
- 专家1：people (person)
- 专家2：two_wheelers (bicycle, motorcycle)

#### 配置C：3个专家（按尺寸）
- 专家0：large_objects (truck, bus)
- 专家1：medium_objects (car)
- 专家2：small_objects (person, bicycle, motorcycle)

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--config` | A | MOE配置 (A/B/C) |
| `--data_root` | datasets/DAIR-V2X | 数据集路径 |
| `--epochs` | 100 | 训练轮数 |
| `--batch_size` | 16 | 批次大小 |
| `--pretrained_lr` | 1e-5 | 预训练组件学习率 |
| `--new_lr` | 1e-4 | 新组件学习率 |
| `--top_k` | 2 | 路由器Top-K选择 |

## 模型架构

### 完整架构

```
输入图像 [B, 3, H, W]
    ↓
┌─────────────────────────────────────┐
│        共享部分（所有专家共用）        │
├─────────────────────────────────────┤
│  Backbone (PResNet)                │
│  Encoder (HybridEncoder)           │
└─────────────────────────────────────┘
    ↓
共享特征 [B, seq_len, hidden_dim]
    ↓
┌─────────────────────────────────────┐
│           路由器 (Router)           │
├─────────────────────────────────────┤
│  Router Network                    │
│  Top-K Selection                   │
│  Load Balancing                    │
└─────────────────────────────────────┘
    ↓
专家选择 [B, seq_len, top_k]
    ↓
┌─────────────────────────────────────┐
│        专家网络 (Experts)           │
├─────────────────────────────────────┤
│  专家0-N: Expert Network + Detection │
└─────────────────────────────────────┘
    ↓
加权融合输出
```

### 关键组件

1. **路由器（Router）**：决定哪些专家应该被激活
2. **专家网络（Experts）**：每个专家专门处理特定任务
3. **检测头（Detection Heads）**：每个专家有独立的检测头
4. **负载均衡（Load Balancing）**：确保所有专家都被有效使用

## 训练过程

### 第一阶段：MOE联合训练

1. **共享特征提取**：Backbone + Encoder提取通用特征
2. **专家选择**：路由器决定激活哪些专家
3. **专家处理**：选中的专家处理特定任务
4. **损失计算**：每个专家计算自己的损失
5. **参数更新**：所有参数联合更新

### 损失函数

- **检测损失**：每个专家的检测损失
- **路由器损失**：负载均衡损失
- **总损失**：所有损失的和

## 输出文件

### 训练日志

```
logs/moe_rtdetr_YYYYMMDD_HHMMSS/
├── training.log              # 训练日志
├── config.yaml              # 配置文件
├── checkpoint_epoch_*.pth    # 检查点文件
└── best_model.pth           # 最佳模型
```

### 日志内容

- 训练和验证损失
- 每个专家的损失
- 路由器损失
- 学习率变化
- 专家使用情况

## 监控训练

### 关键指标

1. **总损失**：整体训练进度
2. **专家损失**：各专家的学习情况
3. **路由器损失**：负载均衡情况
4. **学习率**：优化器状态

### 日志示例

```
Epoch 0:
  训练损失: 2.3456
  验证损失: 2.1234
  路由器损失: 0.1234
  专家0: 训练=0.4567, 验证=0.4321
  专家1: 训练=0.3456, 验证=0.3210
  ...
```

## 故障排除

### 常见问题

1. **CUDA内存不足**
   ```bash
   # 减小batch_size
   --batch_size 8
   ```

2. **训练不收敛**
   ```bash
   # 调整学习率
   --pretrained_lr 5e-6
   --new_lr 5e-5
   ```

3. **专家不平衡**
   - 检查路由器损失
   - 调整负载均衡权重

### 调试建议

1. 使用小数据集进行快速测试
2. 监控每个专家的损失变化
3. 检查专家使用分布
4. 对比不同配置的结果

## 扩展功能

### 第二阶段：语言门控训练

训练完成后，可以继续训练语言门控机制：

```python
# 加载第一阶段训练的模型
model = CompleteMOERTDETR(config_name="A")
model.load_state_dict(torch.load("best_model.pth"))

# 添加语言门控
language_router = LanguageGatedRouter(...)
# 继续训练...
```

### 自定义配置

可以修改配置文件来调整训练参数：

```yaml
# configs/moe_rtdetr_dair_v2x.yml
model:
  config_name: "A"
  hidden_dim: 256
  top_k: 2

training:
  epochs: 100
  batch_size: 16
  pretrained_lr: 1e-5
  new_lr: 1e-4
```

## 性能优化

### 训练加速

1. **使用GPU**：确保CUDA可用
2. **多进程数据加载**：增加num_workers
3. **混合精度训练**：使用torch.cuda.amp
4. **梯度累积**：处理大batch_size

### 内存优化

1. **减小batch_size**
2. **使用梯度检查点**
3. **清理中间变量**
4. **使用CPU卸载**

## 训练监控

### 实时监控指标

训练过程中会输出详细的日志信息，包括：

#### 基础指标
- **总损失 (Total Loss)**: 整体训练损失
- **验证损失 (Validation Loss)**: 验证集上的损失
- **验证准确率 (Validation Accuracy)**: 类别预测准确率

#### MOE特定指标
- **路由器损失 (Router Loss)**: 负载均衡损失
- **专家损失 (Expert Losses)**: 每个专家的个体损失
- **专家使用率 (Expert Usage Rate)**: 每个专家的使用频率
- **路由熵 (Routing Entropy)**: 路由策略的多样性

#### 训练日志示例
```
Epoch 1:
  训练损失: 0.6211
  验证损失: 0.5987
  验证准确率: 0.2345 (123/524)
  路由器损失: 0.0123
  专家损失: ['0.1567', '0.1456', '0.1345', '0.1234']
  专家使用率: ['0.250', '0.250', '0.250', '0.250']
  路由熵: 1.2345
```

### 监控工具

#### 1. 实时监控脚本
```bash
# 实时监控训练进度
python monitor_training.py --log_file moe_training.log --watch --interval 30

# 生成训练指标图表
python monitor_training.py --log_file moe_training.log --plot --save_plot training_metrics.png
```

#### 2. 结果分析脚本
```bash
# 分析训练结果
python analyze_results.py --checkpoint checkpoints/moe_rtdetr_best.pth --output analysis_report.json
```

#### 3. 训练指标解读

**损失指标**:
- 总损失下降表示模型整体性能提升
- 路由器损失下降表示专家负载更均衡
- 专家损失差异反映专家专业化程度

**准确率指标**:
- 验证准确率反映模型分类性能
- 与基准模型对比评估MOE效果

**专家使用指标**:
- 专家使用率均匀表示负载均衡良好
- 路由熵反映专家选择的多样性
- 理想状态：各专家使用率相近，路由熵适中(1.0-1.5)

## 联系信息

如有问题或建议，请联系项目维护者。

---

**注意**：这是一个完整的训练系统，包含了MOE架构的所有关键组件。训练完成后，您将获得一个可以在DAIR-V2X数据集上进行任务选择性检测的模型。
