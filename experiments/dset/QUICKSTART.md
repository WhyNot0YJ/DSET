# DSET 快速开始指南

## 🚀 3分钟快速启动

### 1. 安装依赖
```bash
cd dual-moe-rtdetr
pip install -r requirements.txt
```

### 2. 准备数据集
确保DAIR-V2X数据集在 `datasets/DAIR-V2X/` 目录下

### 3. 测试代码
```bash
# 运行测试脚本（确保代码正常）
python test_dset.py
```

### 4. 小规模测试（推荐首次运行）
```bash
# 2个epoch快速验证
python train.py --config configs/dset_presnet50.yaml --epochs 2 --batch_size 4
```

### 5. 正式训练
```bash
# 标准配置
python train.py --config configs/dset_presnet50.yaml

# 或使用脚本
bash run_training.sh
```

---

## 📊 核心创新点

### 双稀疏机制

1. **Token Pruning (30%剪枝)**
   - 在Encoder前剪枝冗余tokens
   - 保留70%重要tokens
   - 渐进式启用（warmup 10 epochs）

2. **Patch-MoE (50%激活)**
   - Encoder中的稀疏专家
   - 4个专家，top-2激活
   - 动态路由选择

3. **Decoder MoE (50%激活)**
   - Decoder中的稀疏专家
   - 6个专家，top-3激活

**总体计算量**: 0.7 × 0.5 ≈ **35%** 🎯

---

## 📁 配置文件

| 文件 | Backbone | 适用场景 |
|------|----------|----------|
| `dset_presnet50.yaml` | PResNet50 | **推荐**：标准训练 |
| `dset_presnet18.yaml` | PResNet18 | 轻量级/资源受限 |

---

## 🔧 关键参数

```yaml
model:
  dset:
    use_token_pruning: true        # 启用Token Pruning
    token_keep_ratio: 0.7          # 保留70% tokens
    token_pruning_warmup_epochs: 10 # 渐进式warmup
    
    use_patch_moe: true            # 启用Patch-MoE
    patch_moe_num_experts: 4       # Encoder专家数
    patch_moe_top_k: 2             # Encoder top-k
```

---

## 📈 训练监控

### 关键指标

1. **Token Pruning Ratio**
   - Epoch 0-10: 0% → 30% (渐进式)
   - Epoch 11+: 稳定在30%

2. **专家使用率**
   - 理想：各专家 ~16.7% (6个专家)
   - 警告：某专家 > 50% (需调整)

3. **损失曲线**
   - Detection Loss: 持续下降
   - MoE Loss: 初期高，后期稳定在~1.0

---

## ❓ 常见问题

### Q: 训练不稳定？
**A**: 增加warmup epochs或降低剪枝强度
```yaml
token_pruning_warmup_epochs: 15
token_keep_ratio: 0.75
```

### Q: CUDA OOM？
**A**: 减小batch size
```yaml
training:
  batch_size: 16  # 或更小
```

### Q: 专家使用不均衡？
**A**: 增加balance weight（修改train.py line 395）
```python
moe_balance_weight = 0.1  # 从0.05增加到0.1
```

---

## 📚 详细文档

完整文档请查看: **[README.md](README.md)**

包含：
- 技术细节
- 代码逻辑验证
- 完整故障排除
- 更多训练技巧

---

## ✅ 准备清单

开始训练前确认：
- [ ] 数据集已准备
- [ ] 依赖已安装
- [ ] 测试通过 (`python test_dset.py`)
- [ ] GPU可用
- [ ] 配置文件已检查

**Ready to train!** 🎉

