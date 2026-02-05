# DSET vs Baselines 终极实验评估报告
生成时间: 2025-12-05
---

## 1. 实验配置审计 (Configuration Audit)

### 1.1 输入分辨率 (Input Resolution)

| Model | Input Resolution | Status |
|:---|:---|:---|
| DSET | 1280 | ✅ |
| DSET | 1280 | ✅ |
| DSET | 1280 | ✅ |
| DSET | 1280 | ✅ |
| DSET | 1280 | ✅ |
| DSET | 1280 | ✅ |
| RT-DETR | 1280 | ✅ |
| RT-DETR | 1280 | ✅ |
| YOLOV8 | 1280 | ✅ |
| YOLOV8 | 1280 | ✅ |
| YOLOV10 | 1280 | ✅ |

### 1.2 验证设置 (Validation Settings)

| Model | conf_thres | max_det | Status |
|:---|:---|:---|:---|
| DSET | 0.001 | 100 | ✅ |
| DSET | 0.001 | 100 | ✅ |
| DSET | 0.001 | 100 | ✅ |
| DSET | 0.001 | 100 | ✅ |
| DSET | 0.001 | 100 | ✅ |
| DSET | 0.001 | 100 | ✅ |
| RT-DETR | 0.001 | 100 | ✅ |
| RT-DETR | 0.001 | 100 | ✅ |
| YOLOV8 | 0.001 | 100 | ✅ |
| YOLOV8 | 0.001 | 100 | ✅ |
| YOLOV10 | 0.001 | 100 | ✅ |

### 1.3 数据增强 (Data Augmentation)

| Model | Mosaic | Mixup | Status |
|:---|:---|:---|:---|
| DSET | 0.0 | None | ✅ |
| DSET | 0.0 | None | ✅ |
| DSET | 0.0 | None | ✅ |
| DSET | 0.0 | None | ✅ |
| DSET | 0.0 | None | ✅ |
| DSET | 0.0 | None | ✅ |
| RT-DETR | None | None | ✅ |
| RT-DETR | 0.0 | None | ✅ |
| YOLOV8 | None | None | ✅ |
| YOLOV8 | None | None | ✅ |
| YOLOV10 | None | None | ✅ |

## 2. 核心性能榜单 (SOTA Comparison)

| Model Name | Best mAP 50-95 | mAP 50 | Best Epoch | Total Epochs |
|:---|:---|:---|:---|:---|
| **yolo_v10l_20251202_112836** | **0.73** | 0.92 | 94 | 200 |
| yolo_v8l_20251201_150106 | 0.72 | 0.92 | 67 | 200 |
| DSET-6 (R34) | 0.70 | 0.89 | 125 | 200 |
| DSET-48 (R34) | 0.70 | 0.89 | 130 | 200 |
| yolo_v8s_20251201_110541 | 0.69 | 0.89 | 99 | 200 |
| DSET-4 (R34) | 0.69 | 0.88 | 135 | 200 |
| RT-DETR (R34) | 0.69 | 0.88 | 140 | 200 |
| DSET-48 (R18) | 0.65 | 0.87 | 125 | 200 |
| DSET-6 (R18) | 0.65 | 0.87 | 130 | 200 |
| RT-DETR (R18) | 0.65 | 0.87 | 149 | 200 |
| DSET-4 (R18) | 0.65 | 0.86 | 105 | 200 |

## 3. 细粒度优势分析 (Critical for Paper)

### 3.1 R18 Backbone 对比

#### 3.1.1 小目标组 (Small Objects)

| Model | Pedestrian AP | Cyclist AP |
|:---|:---|:---|
| DSET-48 (R18) | **0.58** | **0.65** |
| DSET-6 (R18) | **0.59** | **0.66** |
| RT-DETR (R18) | 0.58 | 0.63 |
| DSET-4 (R18) | 0.57 | 0.63 |

#### 3.1.2 DSET vs RT-DETR R18 提升分析 (Small Objects)

| Model | Pedestrian AP | vs RT-DETR R18 | Cyclist AP | vs RT-DETR R18 |
|:---|:---|:---|:---|:---|
| DSET-48 (R18) | 0.58 | +0.64% | 0.65 | +3.45% |
| DSET-6 (R18) | 0.59 | +0.98% | 0.66 | +4.33% |
| DSET-4 (R18) | 0.57 | -2.21% | 0.63 | +0.45% |

#### 3.1.3 困难类别组 (Hard Categories)

| Model | Van AP | Truck AP |
|:---|:---|:---|
| DSET-48 (R18) | N/A | **0.51** |
| DSET-6 (R18) | N/A | **0.52** |
| RT-DETR (R18) | N/A | **0.53** |
| DSET-4 (R18) | N/A | 0.51 |

### 3.2 R34 Backbone 对比

#### 3.2.1 小目标组 (Small Objects)

| Model | Pedestrian AP | Cyclist AP |
|:---|:---|:---|
| DSET-6 (R34) | **0.63** | **0.69** |
| DSET-48 (R34) | **0.64** | 0.69 |
| DSET-4 (R34) | 0.61 | 0.68 |
| RT-DETR (R34) | 0.62 | 0.68 |

#### 3.2.2 DSET vs RT-DETR R34 提升分析 (Small Objects)

| Model | Pedestrian AP | vs RT-DETR R34 | Cyclist AP | vs RT-DETR R34 |
|:---|:---|:---|:---|:---|
| DSET-6 (R34) | 0.63 | +1.43% | 0.69 | +1.72% |
| DSET-48 (R34) | 0.64 | +2.37% | 0.69 | +1.62% |
| DSET-4 (R34) | 0.61 | -0.98% | 0.68 | -0.01% |

#### 3.2.3 困难类别组 (Hard Categories)

| Model | Van AP | Truck AP |
|:---|:---|:---|
| DSET-6 (R34) | N/A | **0.56** |
| DSET-48 (R34) | N/A | 0.54 |
| DSET-4 (R34) | N/A | 0.54 |
| RT-DETR (R34) | N/A | 0.54 |

## 4. 动态计算与稀疏性 (Sparsity Analysis)

| Model | Token Keep Ratio | Pruning Ratio |
|:---|:---|:---|
| yolo_v10l_20251202_112836 | N/A | N/A |
| yolo_v8l_20251201_150106 | N/A | N/A |
| DSET-6 (R34) | 0.90 | 0.10 |
| DSET-48 (R34) | 0.90 | 0.10 |
| yolo_v8s_20251201_110541 | N/A | N/A |
| DSET-4 (R34) | 0.90 | 0.10 |
| RT-DETR (R34) | N/A | N/A |
| DSET-48 (R18) | 0.90 | 0.10 |
| DSET-6 (R18) | 0.90 | 0.10 |
| RT-DETR (R18) | N/A | N/A |
| DSET-4 (R18) | 0.90 | 0.10 |

## 5. 异常与Debug信息

### 5.1 实验状态

| Model | Status | Notes |
|:---|:---|:---|
| yolo_v10l_20251202_112836 | ✅ Completed | Best at epoch 94 |
| yolo_v8l_20251201_150106 | ✅ Completed | Best at epoch 67 |
| DSET-6 (R34) | ✅ Completed | Best at epoch 125 |
| DSET-48 (R34) | ✅ Completed | Best at epoch 130 |
| yolo_v8s_20251201_110541 | ✅ Completed | Best at epoch 99 |
| DSET-4 (R34) | ✅ Completed | Best at epoch 135 |
| RT-DETR (R34) | ✅ Completed | Best at epoch 140 |
| DSET-48 (R18) | ✅ Completed | Best at epoch 125 |
| DSET-6 (R18) | ✅ Completed | Best at epoch 130 |
| RT-DETR (R18) | ✅ Completed | Best at epoch 149 |
| DSET-4 (R18) | ✅ Completed | Best at epoch 105 |
