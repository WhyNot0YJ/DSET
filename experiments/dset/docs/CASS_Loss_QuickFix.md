# CASSLoss 空间偏置问题 - 快速修复指南

## 🔴 问题确认

经过代码分析，确认 CASSLoss 存在**空间密度偏置**，导致 Mask 向图像远端（左上角）偏移。

---

## 📊 核心问题定位

### 问题代码位置

**文件**：`src/zoo/rtdetr/patch_level_pruning.py`

1. **损失归一化问题**（第 557-558 行）：
   ```python
   if reduction == 'mean':
       return loss.mean()  # ❌ 对所有 patches 简单平均，未考虑 object 数量和大小
   ```

2. **Target Mask 合并策略**（第 496 行）：
   ```python
   merged_mask, _ = torch.max(box_masks, dim=0)  # ❌ 多目标 max 合并，高密度区域被放大
   ```

3. **损失计算入口**（第 564-587 行）：
   ```python
   def compute_cass_loss(self, pred_scores, target_mask, reduction='mean'):
       # ❌ 直接对 patch-level 损失取平均，无 object-level 归一化
   ```

---

## ⚡ 快速修复方案

### 方案 A：临时缓解（立即可用）

**修改**：降低 CASS Loss 权重，或调整扩张比例

在配置文件（如 `configs/dset4_r18_ratio0.3.yaml`）中：

```yaml
dset:
  use_cass: true
  cass_loss_weight: 0.01  # 从 0.05 降低到 0.01
  cass_expansion_ratio: 0.3  # 从 0.8 降低到 0.3（更保守的扩张）
```

**原理**：减少 CASS Loss 的影响，让检测损失的主导作用更强。

---

### 方案 B：对象级别归一化（推荐，需要代码修改）

**修改文件**：`src/zoo/rtdetr/patch_level_pruning.py`

在 `PatchLevelPruner` 类中添加新方法：

```python
def compute_cass_loss_object_normalized(
    self,
    pred_scores: torch.Tensor,
    gt_bboxes: List[torch.Tensor],
    feat_shape: Tuple[int, int],
    img_shape: Tuple[int, int]
) -> torch.Tensor:
    """
    按对象计算损失并归一化，避免密度偏置
    """
    device = pred_scores.device
    B = pred_scores.shape[0]
    
    if B != len(gt_bboxes):
        raise ValueError(f"Batch size mismatch: {B} vs {len(gt_bboxes)}")
    
    total_loss = 0.0
    total_objects = 0
    
    for b_idx in range(B):
        bboxes = gt_bboxes[b_idx]
        if bboxes is None or len(bboxes) == 0:
            continue
        
        # 确保 bboxes 是 2D
        if bboxes.dim() == 1:
            bboxes = bboxes.unsqueeze(0)
        
        # 为每个 bbox 单独计算损失
        for bbox in bboxes:
            # 生成单个 bbox 的 target mask
            single_box_mask = self.generate_soft_target_mask(
                gt_bboxes=[bbox.unsqueeze(0)],
                feat_shape=feat_shape,
                img_shape=img_shape,
                device=device
            )  # [1, num_patches]
            
            # 计算该对象的损失
            obj_loss = self.compute_cass_loss(
                pred_scores[b_idx:b_idx+1],
                single_box_mask,
                reduction='mean'
            )
            
            total_loss = total_loss + obj_loss
            total_objects += 1
    
    if total_objects == 0:
        return torch.tensor(0.0, device=device, requires_grad=False)
    
    # 按对象数量归一化
    return total_loss / total_objects
```

然后修改 `compute_cass_loss_from_info` 方法（第 652-684 行）：

```python
def compute_cass_loss_from_info(
    self,
    info: Dict,
    gt_bboxes: List[torch.Tensor],
    feat_shape: Tuple[int, int],
    img_shape: Tuple[int, int]
) -> torch.Tensor:
    """
    使用对象级别归一化计算 CASS loss
    """
    if 'patch_importance_scores' not in info or info['patch_importance_scores'] is None:
        return torch.tensor(0.0, requires_grad=False)
    
    pred_scores = info['patch_importance_scores']
    
    # 使用新的对象级别归一化方法
    loss = self.compute_cass_loss_object_normalized(
        pred_scores=pred_scores,
        gt_bboxes=gt_bboxes,
        feat_shape=feat_shape,
        img_shape=img_shape
    )
    
    return loss
```

---

### 方案 C：Area-Weighted 损失（更精细）

如果需要进一步平衡大小目标，可以在方案 B 基础上加入 area 加权：

```python
# 在 compute_cass_loss_object_normalized 中
for bbox in bboxes:
    # 计算 bbox area（用于加权）
    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    area_normalized = area / (img_shape[0] * img_shape[1])  # 归一化到 [0, 1]
    
    # ... 计算 obj_loss ...
    
    # 按 area 加权（可选：也可以不加权，只用对象数量归一化）
    total_loss = total_loss + obj_loss * area_normalized
    # 或：total_loss = total_loss + obj_loss  # 只用对象数量归一化
```

---

## 🔍 验证步骤

### 1. 对比实验

修改前后，对比以下指标：

```python
# 在训练循环中添加统计
spatial_distribution = {
    'top_left_loss': [],      # 左上角区域损失
    'bottom_loss': [],         # 下方区域损失
    'large_obj_loss': [],      # 大目标损失
    'small_obj_loss': []       # 小目标损失
}
```

### 2. 可视化 Mask 分布

使用现有的 `_debug_visualize_mask` 方法（第 589-651 行），或添加自定义可视化：

```python
# 在 compute_cass_loss_from_info 中添加
if self.training and random.random() < 0.01:  # 1% 采样率
    self._debug_visualize_mask(target_mask, pred_scores)
```

### 3. 监控训练指标

观察以下指标是否改善：
- **Detection Loss**：大目标的检测损失是否下降
- **Mask 分布**：可视化 mask 是否更均匀
- **验证集 mAP**：特别是大目标的 AP

---

## 📝 配置建议

修复后，建议的配置参数：

```yaml
dset:
  use_cass: true
  cass_loss_weight: 0.05  # 可以恢复到原始值
  cass_expansion_ratio: 0.3  # 建议降低（当前 0.8 可能过大）
  cass_min_size: 2.0  # 保持不变
  use_subpixel_offset: true  # 保持启用
  use_focal_loss: true  # 可以尝试 false（使用 MSE），看效果
  cass_focal_alpha: 2.0
  cass_focal_beta: 4.0
```

---

## ⚠️ 注意事项

1. **向后兼容性**：如果修改了 `compute_cass_loss_from_info` 的接口，需要确保所有调用点都更新。

2. **性能影响**：方案 B（对象级别归一化）会增加计算量（每个 bbox 单独生成 mask），但通常可接受。

3. **渐进式修复**：建议先实施方案 A（降低权重），验证问题是否缓解，再实施方案 B。

4. **Warmup 机制**：确保 CASS Loss 在 warmup 期间仍然禁用（代码已实现，无需修改）。

---

## 🔗 相关文件

- **核心实现**：`src/zoo/rtdetr/patch_level_pruning.py`
- **损失集成**：`train.py`（第 487-549 行）
- **配置示例**：`configs/dset4_r18_ratio0.3.yaml`

