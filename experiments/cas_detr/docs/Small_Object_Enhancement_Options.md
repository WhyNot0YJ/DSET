# CaS_DETR 小目标增强方案记录

## 1. 目的

本文记录两个适合在当前 `experiments/cas_detr` 代码基础上继续演进的结构方向：

1. **高分辨率细节分支**
2. **Small-object Query 初始化**

这两个方向都不是继续围绕 token 剪枝本身做微调，而是从**小目标信息保真**和**小目标候选进入 decoder 的概率**两个角度补足当前架构的短板。

文档重点回答以下问题：

- 当前代码里，这两个方向分别应该接到哪里
- 为什么它们对 `AP_small` 更有帮助
- 最小可用版本应该怎么做
- 训练、推理、显存、工程复杂度会增加多少
- 两者相比，哪个更适合先做

---

## 2. 当前代码现状

### 2.1 当前主线改动集中在什么地方

当前 `CaS_DETR` 对 RT-DETR 的核心改动集中在：

- `HybridEncoder` 中的 **Token Pruning**
- `TokenLevelPruner` 中的 **CASS**
- `CAIPPredictor` 中的 **动态 keep_ratio**
- `RTDETRTransformerv2` decoder FFN 中的 **MoE**

主数据流可以概括为：

`Backbone -> HybridEncoder -> FPN/PAN -> RTDETRTransformerv2 -> Detection Heads`

其中，真正与小目标最相关的两个薄弱点是：

1. **细粒度高分辨率信息没有被单独增强**
2. **decoder 初始 query 选择没有专门照顾小目标**

### 2.2 当前 encoder 的关键行为

`HybridEncoder.forward()` 会先将 `use_encoder_idx` 指定的特征层展平并拼接，再统一做 token pruning 与 encoder 计算，之后 scatter 回完整网格，再进入 FPN/PAN。

这意味着当前架构更偏向：

- 在已有 token 里挑重点
- 压缩 encoder 计算
- 保留整体精度

它并没有显式做这两件事：

- 给更细尺度的局部纹理单独补强
- 给小目标保留专门的 decoder query 入口

### 2.3 当前 decoder query 是怎么初始化的

`RTDETRTransformerv2._get_decoder_input()` 会在 encoder memory 上预测分类分数和框，再通过 `_select_topk()` 直接选出 top-k query。

这个机制对显著目标、分数更早抬升的目标更有利。小目标常见问题不是后期 refine 不出来，而是**早期候选压根没进去**。

---

## 3. 方案一：高分辨率细节分支

### 3.1 核心目标

这个方向的目标是：

- 在进入 FPN/PAN 或 decoder 之前，显式保留并增强更细尺度特征中的边缘、纹理和局部几何信息
- 降低小目标在深层语义特征中被过度平滑、过度抽象的风险

一句话概括：

**先让小目标“看得更清楚”，再谈如何挑 token 和挑 query。**

### 3.2 为什么当前代码需要它

当前配置中，典型 encoder 输入是：

- `use_encoder_idx: [1, 2]`

也就是主要依赖中高层特征。虽然这些层有更强语义，但对小目标来说，往往存在两个问题：

1. **空间分辨率偏低**
2. **小目标纹理和轮廓被语义化表示冲淡**

即使后面有 CASS 和 CAIP，它们解决的是：

- 哪些 token 更重要
- 在复杂场景里多留一些 token

但它们并不直接创造新的高频细节信息。

### 3.3 适合当前仓库的接入位置

最合适的入口是 `HybridEncoder.forward()` 中 `proj_feats` 刚构建完的时候。

推荐接入点：

1. `proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]` 之后
2. 在 top-down FPN 融合前
3. 仅作用于最细输出，或者仅向最细输出注入

### 3.4 推荐的最小实现

推荐做一个轻量 detail branch，不要第一版就改 backbone，也不要第一版就让它参与 token pruning。

建议结构：

```text
proj_feats[0]
  -> 3x3 depthwise conv
  -> GELU
  -> 5x5 depthwise conv
  -> 1x1 conv
  -> detail_feat
```

然后有两种最稳的接法：

#### 方案 A：注入最细 FPN 输出

```text
outs[0] = outs[0] + alpha * detail_feat
```

特点：

- 改动最小
- 不改 token pruning 逻辑
- 最容易先验证是否对 `AP_small` 有帮助

#### 方案 B：作为额外融合输入

```text
inner_out = Fuse(upsample_feat, feat_low, detail_feat)
```

特点：

- 效果上限更高
- 但 neck 结构也要一起调整

### 3.5 可以考虑的增强模块

按推荐顺序：

1. **DWConv + 大核 DWConv**
   - 最轻量
   - 对局部纹理增强直接有效
   - 最适合先做

2. **RFB / ASPP-lite**
   - 能补局部与稍大感受野
   - 对中小目标边缘与上下文平衡更好

3. **轻量门控注意力**
   - 如 SE、EMA、Coordinate Attention
   - 适合在不明显增加 FLOPs 的情况下增强通道重标定

4. **DCN 或可变形局部块**
   - 可能更强
   - 但工程复杂度、训练不稳定性、实现成本都更高

### 3.6 最小实现需要改哪些文件

核心只需要：

- `experiments/cas_detr/src/zoo/rtdetr/hybrid_encoder.py`

通常还会补：

- `experiments/cas_detr/train.py`
  - 配置透传新参数
- `experiments/cas_detr/configs/*.yaml`
  - 新增 detail branch 开关与超参

### 3.7 建议新增配置项

建议放在 `model.cas_detr` 下，便于与当前增强模块归类一致：

```yaml
model:
  cas_detr:
    use_detail_branch: true
    detail_branch_in_index: 0
    detail_branch_kernel_sizes: [3, 5]
    detail_branch_hidden_ratio: 1.0
    detail_branch_fuse_mode: add
    detail_branch_fuse_weight: 1.0
```

### 3.8 优点

- 对小目标最直接，逻辑上最合理
- 与当前 token pruning、CAIP、MoE 都不冲突
- 可以先只增强最细特征，不破坏主干结构
- 最有机会直接提升 `AP_small` 和 `AR_small`

### 3.9 风险

- 更高分辨率分支通常会增加显存和一点延迟
- 如果增强块过重，可能拉低整体速度优势
- 如果融合设计不当，可能把噪声也带入最细层

### 3.10 训练与推理代价

粗略判断：

- **参数量**：小幅增加
- **GFLOPs**：小到中等增加
- **显存**：中等增加，取决于最细分支分辨率
- **推理延迟**：小幅增加

### 3.11 推荐分与理由

**推荐分：9.4 / 10**

理由：

- 最直接补小目标最缺的东西
- 对现有主线干扰小
- 不要求先推翻当前剪枝设计
- 第一版就有希望看到 `AP_small` 改善

---

## 4. 方案二：Small-object Query 初始化

### 4.1 核心目标

这个方向的目标是：

- 在 decoder 初始化阶段，显式给小目标留出候选入口
- 提高小目标被 query 早期命中的概率
- 提升小目标的 recall，尤其是 `AR_small`

一句话概括：

**不是等 decoder 自己发现小目标，而是一开始就让一部分 query 去找小目标。**

### 4.2 为什么当前代码需要它

当前 `RTDETRTransformerv2._select_topk()` 是对 encoder memory 上的分类分数直接做 top-k。

这种方式天然更偏向：

- 响应更强的目标
- 面积更大、得分更容易抬升的目标
- 更容易形成稳定 proposal 的目标

而小目标经常会在这个阶段吃亏，因为它们：

- 特征响应更弱
- 初始框更不稳定
- 分类置信度更晚抬升

因此，单纯全局 top-k 往往会让小目标在 decoder 入口就被挤掉。

### 4.3 适合当前仓库的接入位置

最合适的入口是：

- `experiments/cas_detr/src/zoo/rtdetr/rtdetrv2_decoder.py`

尤其是：

- `_get_decoder_input()`
- `_select_topk()`

这里已经具备实现小目标 query 保留机制所需的全部信息：

- `memory`
- `spatial_shapes`
- `anchors`
- 每个 token 所属 level 的顺序

### 4.4 推荐的最小实现

最推荐第一版做 **保底配额式 small-object query 初始化**，而不是直接上完整双 query bank。

推荐逻辑：

1. 先从更高分辨率 level 中保留一部分 query
2. 剩余 query 再走原来的全局 top-k

例如：

- 总 query 数：300
- `small_object_query_ratio = 0.2`
- 其中 60 个 query 从高分辨率 level 保底选出
- 另外 240 个 query 继续从全局 top-k 产生

### 4.5 第一版推荐实现细节

#### 方案 A：Level-aware 保底配额

根据 `spatial_shapes` 计算每个 level 在扁平 memory 中的起止区间，然后：

- 选定最高分辨率 level 区间
- 在该区间内做一次 top-k
- 再在全局范围内做剩余 top-k
- 最后拼接并去重

这是最容易接进当前代码、风险最可控的实现。

#### 方案 B：小框先验偏置

在 encoder 输出候选阶段，对更小 anchor 或更细 level 的候选加一个偏置分数，再做 top-k。

优点：

- 改动更小

缺点：

- 可解释性稍弱
- 不如保底配额直观

#### 方案 C：完整双 query bank

将 query 分成：

- general queries
- small-object queries

并可进一步为 small-object queries 设计单独的内容初始化或参考框初始化。

优点：

- 更有研究味
- 上限更高

缺点：

- 改动更大
- 需要更多消融和训练稳定性验证

### 4.6 最小实现需要改哪些文件

核心只需要：

- `experiments/cas_detr/src/zoo/rtdetr/rtdetrv2_decoder.py`

通常还会补：

- `experiments/cas_detr/train.py`
  - 配置透传
- `experiments/cas_detr/configs/*.yaml`
  - 配置新增 small-object query 相关开关

### 4.7 建议新增配置项

建议放在 decoder 相关配置附近：

```yaml
model:
  small_object_query:
    enabled: true
    reserve_ratio: 0.2
    reserve_level: 0
    selection_mode: level_reserve
    bias_to_small_anchors: false
```

或者如果不希望新增根级字段，也可放在 `model.cas_detr` 下统一管理。

### 4.8 优点

- 改动范围小
- 与当前主干兼容性高
- 最容易保留当前速度和显存特征
- 对小目标 recall 可能有明显帮助

### 4.9 风险

- 如果保底比例设太大，会挤占大目标和普通目标 query
- 如果最高分辨率层噪声较多，可能引入低质量 proposal
- 如果只做 query 端，不补细节分支，收益可能受限于输入特征本身的信息量

### 4.10 训练与推理代价

粗略判断：

- **参数量**：几乎不变，除非增加新的 query 头
- **GFLOPs**：几乎不变
- **显存**：几乎不变
- **推理延迟**：变化极小

### 4.11 推荐分与理由

**推荐分：8.5 / 10**

理由：

- 改动更少
- 对当前 decoder 入口非常友好
- 工程上适合快速验证
- 但它更像“更好地使用已有特征”，不如高分辨率细节分支那样直接补足表征能力

---

## 5. 两个方案的多维度对比

| 维度 | 高分辨率细节分支 | Small-object Query 初始化 |
|---|---|---|
| 核心作用阶段 | Encoder / Neck 之前与之中 | Decoder 输入阶段 |
| 主要解决问题 | 小目标细节信息不足 | 小目标候选进不去 decoder |
| 对 `AP_small` 潜力 | 很高 | 中高 |
| 对 `AR_small` 潜力 | 高 | 很高 |
| 参数量增加 | 小到中 | 很小 |
| GFLOPs 增加 | 小到中 | 很小 |
| 实现复杂度 | 中 | 低到中 |
| 对现有代码侵入性 | 中 | 低 |
| 与 token pruning 冲突 | 低 | 低 |
| 与 CAIP / MoE 冲突 | 低 | 低 |
| 第一版实验成本 | 中 | 低 |
| 推荐优先级 | 更高 | 更适合先快试 |

---

## 6. 哪个更适合先做

### 6.1 如果目标是先做最小改动、尽快出结果

优先做：

**Small-object Query 初始化**

原因：

- 只需要改 decoder query 选择逻辑
- 基本不动主干
- 更适合快速做一轮对照实验

### 6.2 如果目标是更有机会直接提升 `AP_small`

优先做：

**高分辨率细节分支**

原因：

- 直接补小目标最缺的高分辨率信息
- 对表征层面更本质
- 更有机会带来稳定收益

### 6.3 如果最终想把小目标效果做得更完整

推荐顺序：

1. 先做 **Small-object Query 初始化**，快速验证小目标候选入口是否是瓶颈
2. 再做 **高分辨率细节分支**，增强输入到 neck 和 decoder 的小目标信息
3. 如果两者都有收益，再考虑把两者联动

---

## 7. 推荐的两阶段实验方案

### 阶段一：低风险快试

目标：

- 在不明显增加复杂度的前提下，先观察小目标 recall 是否改善

建议：

- 只做 `Small-object Query 初始化`
- 保底 query 比例从 `0.1`、`0.2`、`0.3` 做三组
- 重点看：
  - `AP_small`
  - `AR_small`
  - overall `mAP`
  - 是否出现大目标性能回落

### 阶段二：主线增强

目标：

- 真正补足小目标表征能力

建议：

- 加 `高分辨率细节分支`
- 先用最轻的 `DWConv + 大核 DWConv + add fuse`
- 不要第一版就和 pruning 深度绑定
- 重点看：
  - `AP_small`
  - `AR_small`
  - 推理延迟
  - 显存变化

### 阶段三：联动验证

如果前两步各自有效，再组合验证：

- `detail branch only`
- `small-query only`
- `detail branch + small-query`

重点看两者收益是否叠加，而不是互相替代。

---

## 8. 最后的结论

### 8.1 高分辨率细节分支

这是当前最值得认真投入的小目标结构增强方向。

它的价值在于：

- 不只是重新分配现有 token
- 而是补回小目标真正缺的细粒度信息

如果目标是冲 `AP_small`，它是优先级最高的方案。

### 8.2 Small-object Query 初始化

这是当前最适合先做的小改动方案。

它的价值在于：

- 不改变主干大逻辑
- 很容易快速验证“小目标是不是在 decoder 入口就已经吃亏”

如果目标是低风险、快速做实验，它是优先级最高的方案。

### 8.3 实用建议

若只做一个：

- 想**更快试验**，先做 `Small-object Query 初始化`
- 想**更可能直接涨 AP_small**，先做 `高分辨率细节分支`

若最终希望形成一条更完整的小目标增强路线：

- 这两个方向值得组合推进

