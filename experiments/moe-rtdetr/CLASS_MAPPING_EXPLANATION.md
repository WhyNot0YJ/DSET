# 类别映射的作用位置和流程

## 训练时的数据流程

### 1. 数据加载流程

```
DataLoader
  ↓
Dataset.__getitem__(idx)  [第96行]
  ↓
_load_annotations(annotation_path)  [第107行调用，第158行定义]
  ↓
  读取JSON标注文件
  ↓
  对每个标注对象：
    ann["type"] = "Pedestrian"  (原始标注)
    ↓
    class_name = "pedestrian"  (转小写)
    ↓
    【类别映射在这里起作用！】[第171-194行]
    ↓
    class_mapping["pedestrian"] = "person"
    ↓
    class_id = self.class_to_id["person"] = 3
    ↓
    返回 processed_annotations[{'category_id': 3, ...}]
  ↓
_adjust_annotations()  [第111行]
  ↓
  调整坐标，但category_id保持不变
  ↓
target['labels'] = torch.tensor([3, 3, 0, 1, ...])  [第116行]
  ↓
传递给模型进行训练
```

### 2. 类别映射的具体位置

**文件：** `src/data/dataset/dairv2x_detection.py`

**函数：** `_load_annotations()` (第158-220行)

**关键代码：**
```python
# 第169行：从标注文件读取类别
class_name = ann["type"].lower()  # "Pedestrian" -> "pedestrian"

# 第171-194行：类别映射（这里起作用！）
class_mapping = {
    "pedestrian": "person",  # Pedestrian -> person
    "cyclist": "bicycle",     # Cyclist -> bicycle
    "motorcyclist": "motorcycle",
    "van": "truck",
}

# 第187-190行：应用映射
if class_name in class_mapping:
    mapped_name = class_mapping[class_name]  # "pedestrian" -> "person"
    class_id = self.class_to_id[mapped_name]  # "person" -> 3
```

### 3. 映射后的数据流向

```
_load_annotations() 返回:
  [
    {'category_id': 3, 'bbox': [...], ...},  # Pedestrian -> person (3)
    {'category_id': 0, 'bbox': [...], ...},  # Car -> car (0)
    {'category_id': 1, 'bbox': [...], ...},  # Van -> truck (1)
  ]
  ↓
_adjust_annotations() 调整坐标，category_id不变
  ↓
target['labels'] = torch.tensor([3, 0, 1])  # 传递给模型
  ↓
模型训练时使用这些labels计算损失
```

## 为什么之前有问题？

### 修复前（错误的）：
```python
class_name = ann["type"].lower()  # "pedestrian"
if class_name in self.class_to_id:  # "pedestrian" 不在 {"car", "truck", "bus", "person", "bicycle", "motorcycle"} 中
    class_id = self.class_to_id[class_name]
else:
    continue  # ❌ 跳过！Pedestrian被丢弃了
```

**结果：**
- Pedestrian → 被跳过，没有训练数据
- Cyclist → 被跳过，没有训练数据
- Motorcyclist → 被跳过，没有训练数据
- 模型从未学习过这些类别！

### 修复后（正确的）：
```python
class_name = ann["type"].lower()  # "pedestrian"
if class_name in self.class_to_id:
    class_id = self.class_to_id[class_name]
elif class_name in class_mapping:  # ✅ "pedestrian" 在映射表中
    mapped_name = class_mapping[class_name]  # "person"
    class_id = self.class_to_id[mapped_name]  # 3
```

**结果：**
- Pedestrian → person (3) ✅
- Cyclist → bicycle (4) ✅
- Motorcyclist → motorcycle (5) ✅
- 模型可以学习这些类别！

## 总结

**类别映射在 `_load_annotations()` 函数中起作用**，这个函数在每次数据加载时被调用（`__getitem__`），将原始标注文件的类别名称（如"Pedestrian"）映射到模型期望的类别ID（如3）。

如果映射不正确，这些类别的训练数据会被跳过，导致模型无法学习识别这些类别。

