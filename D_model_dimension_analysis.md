# 报告：D_model 维度不匹配分析

## 1. D_model 配置与实际值

| 名称 | 预期值 (来自配置) | 实际值 (代码运行时) | 备注 |
| :--- | :--- | :--- | :--- |
| hidden_dim (D_model) | **256** | **256** (从 `self.config['model']['hidden_dim']` 读取) | 配置文件中明确设置为 256 |
| DSETRTDETR.hidden_dim | 256 | 256 (从配置传入) | `train.py:631` 传入 `hidden_dim=self.config['model']['hidden_dim']` |
| HybridEncoder.hidden_dim | 256 | 256 (从 DSETRTDETR 传入) | `train.py:252` 传入 `hidden_dim=self.hidden_dim` |

**代码路径追踪：**
- 配置文件：`experiments/dset/configs/dset4_r18.yaml:15` → `hidden_dim: 256`
- 模型创建：`train.py:631` → `hidden_dim=self.config['model']['hidden_dim']` → **256**
- Encoder 创建：`train.py:252` → `hidden_dim=self.hidden_dim` → **256**
- HybridEncoder 初始化：`hybrid_encoder.py:248` → `self.hidden_dim = hidden_dim` → **256**

## 2. HybridEncoder 通道投影分析

| 模块 | 输入通道 (in_channel) | 输出通道 (hidden_dim) | 备注 |
| :--- | :--- | :--- | :--- |
| input_proj[0] (Conv2d) | 128 (P3) | 256 | `hybrid_encoder.py:267` - `nn.Conv2d(in_channel, hidden_dim, ...)` |
| input_proj[1] (Conv2d) | 256 (P4) | 256 | 同上 |
| input_proj[2] (Conv2d) | **512 (P5)** | **256** | **用于 use_encoder_idx=[2]，处理 P5 特征** |

**代码位置：**
- `hybrid_encoder.py:260-273` - 创建所有 input_proj 层
- `hybrid_encoder.py:267` - `nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False)`
- 对于 `use_encoder_idx=[2]`，使用 `input_proj[2]`，输入通道 512，输出通道 256

## 3. 1039/1050 追踪结果 (【关键】)

| 维度 | 实际值 (报错信息) | 来源（代码行及变量） |
| :--- | :--- | :--- |
| **张量 A** (特征通道) | **1039** | `hybrid_encoder.py:422` - `src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)`<br>**预期通道数：256**<br>**实际通道数：1039**<br>**可能原因：**<br>1. `proj_feats[enc_ind].shape[1]` 不是 256<br>2. `input_proj[2]` 的输出通道被修改<br>3. 预训练权重加载时修改了通道数 |
| **张量 B** (位置编码) | **1050** | `hybrid_encoder.py:460/465/469` - `build_2d_sincos_position_embedding(w_pruned, h_pruned, self.hidden_dim, ...)`<br>**预期通道数：256**<br>**实际通道数：1050**<br>**可能原因：**<br>1. `self.hidden_dim` 在运行时不是 256<br>2. `build_2d_sincos_position_embedding` 函数内部计算错误<br>3. 传入的 `embed_dim` 参数不是 `self.hidden_dim` |

### 详细代码追踪

#### 张量 A (src_flatten) 的通道数来源：
```python
# hybrid_encoder.py:410
proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
# proj_feats[2] 形状: [B, 256, H, W] (预期)

# hybrid_encoder.py:420-422
for i, enc_ind in enumerate(self.use_encoder_idx):  # enc_ind = 2
    h, w = proj_feats[enc_ind].shape[2:]  # 获取 H, W
    src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
    # 预期: src_flatten.shape = [B, H*W, 256]
    # 实际报错: src_flatten.shape[2] = 1039 ❌
```

#### 张量 B (pos_embed) 的通道数来源：
```python
# hybrid_encoder.py:460/465/469
pos_embed = self.build_2d_sincos_position_embedding(
    w_pruned, h_pruned, self.hidden_dim, self.pe_temperature)
# 预期: pos_embed.shape = [1, H*W, 256]
# 实际报错: pos_embed.shape[2] = 1050 ❌

# hybrid_encoder.py:364-379 (build_2d_sincos_position_embedding)
def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.):
    # embed_dim 应该是 self.hidden_dim = 256
    # 但实际传入的 embed_dim 可能是 1050
    pos_dim = embed_dim // 4  # 如果 embed_dim=1050, pos_dim=262
    # ... 计算 ...
    return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)
    # 输出通道 = 4 * pos_dim = 4 * (embed_dim // 4) = embed_dim
    # 如果 embed_dim=1050, 输出通道 = 1050
```

## 4. 问题分析

### 可能的原因：

1. **预训练权重加载问题**
   - 如果预训练权重中的 `input_proj[2]` 的输出通道不是 256，加载后可能修改了通道数
   - 检查点：`train.py:689-773` - `_load_pretrained_weights` 方法

2. **hidden_dim 在运行时被修改**
   - 虽然代码中 `self.hidden_dim = hidden_dim`，但可能在某个地方被重新赋值
   - 检查点：搜索所有 `self.hidden_dim =` 的赋值

3. **配置读取错误**
   - 虽然配置文件中是 256，但可能在读取时被修改
   - 检查点：`train.py:631` - 确认 `self.config['model']['hidden_dim']` 的实际值

4. **维度计算错误**
   - `build_2d_sincos_position_embedding` 函数可能在某些边界情况下计算出错
   - 检查点：`hybrid_encoder.py:364-379`

## 5. 可能的原因分析

### 原因 1：预训练权重通道数不匹配
- **现象**：预训练权重中的 `encoder.input_proj.2.conv.weight` 形状可能是 `[1039, 512, 1, 1]`
- **影响**：虽然 `load_state_dict(strict=False)` 会跳过不匹配的参数，但如果权重被部分加载，可能导致通道数异常
- **检查方法**：在 `train.py:732` 后检查 `missing_keys` 中是否包含 `encoder.input_proj` 相关参数

### 原因 2：hidden_dim 在运行时被意外修改
- **现象**：`self.hidden_dim` 在初始化后可能被某个操作修改
- **检查方法**：在 `hybrid_encoder.py:248` 和 `hybrid_encoder.py:460` 处打印 `self.hidden_dim` 的值

### 原因 3：input_proj 输出通道数异常
- **现象**：`proj_feats[enc_ind]` 的通道数不是预期的 256
- **检查方法**：在 `hybrid_encoder.py:410` 后检查 `proj_feats[enc_ind].shape[1]`

### 原因 4：build_2d_sincos_position_embedding 计算错误
- **现象**：传入的 `embed_dim` 参数不是 `self.hidden_dim`
- **检查方法**：在 `hybrid_encoder.py:460` 前检查传入的 `self.hidden_dim` 值

## 6. 数值分析

### 1039 和 1050 的特性：
- **1039**：质数，不能被 4 整除（1039 % 4 = 3）
- **1050**：可以被 2、3、5、7 等整除，但**不能被 4 整除**（1050 % 4 = 2）

### 关键发现：
- `build_2d_sincos_position_embedding` 函数要求 `embed_dim % 4 == 0`（`hybrid_encoder.py:370`）
- 如果 `self.hidden_dim = 1050`，调用 `build_2d_sincos_position_embedding` 时会触发断言错误
- 这意味着如果报错信息显示 `pos_embed.shape[2] = 1050`，那么：
  1. 要么断言被跳过了（不应该发生）
  2. 要么 `pos_embed` 不是通过 `build_2d_sincos_position_embedding` 创建的
  3. 要么报错信息中的维度不是通道维度

### 可能的情况：
1. **1039** 可能来自某个计算错误，导致 `proj_feats[enc_ind].shape[1]` 不是 256
2. **1050** 可能来自某个硬编码值或计算错误，但不符合 `build_2d_sincos_position_embedding` 的要求

## 7. 建议的调试步骤

### 步骤 1：在模型初始化时添加调试输出
```python
# train.py:631 后
print(f"[DEBUG] Model hidden_dim: {model.hidden_dim}")
print(f"[DEBUG] Encoder hidden_dim: {model.encoder.hidden_dim}")
print(f"[DEBUG] input_proj[2] output channels: {model.encoder.input_proj[2][0].out_channels}")
```

### 步骤 2：在 HybridEncoder.forward 中添加调试输出
```python
# hybrid_encoder.py:410 后
print(f"[DEBUG] proj_feats[{enc_ind}].shape: {proj_feats[enc_ind].shape}")

# hybrid_encoder.py:422 后
print(f"[DEBUG] src_flatten.shape: {src_flatten.shape}")
print(f"[DEBUG] src_flatten channels: {src_flatten.shape[2]}")

# hybrid_encoder.py:460 前
print(f"[DEBUG] self.hidden_dim: {self.hidden_dim}")
print(f"[DEBUG] w_pruned: {w_pruned}, h_pruned: {h_pruned}")
```

### 步骤 3：检查预训练权重加载
```python
# train.py:732 后
print(f"[DEBUG] Missing keys containing input_proj:")
for key in missing_keys:
    if 'input_proj' in key:
        print(f"  - {key}: {filtered_state_dict.get(key, 'NOT IN STATE_DICT').shape if key in filtered_state_dict else 'NOT FOUND'}")
```

### 步骤 4：验证配置读取
```python
# train.py:631 前
print(f"[DEBUG] Config hidden_dim: {self.config['model']['hidden_dim']}")
print(f"[DEBUG] Config type: {type(self.config['model']['hidden_dim'])}")
```

