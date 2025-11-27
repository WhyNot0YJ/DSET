# COCO数据集标准格式指南

## 标准COCO目录结构

标准的COCO数据集应该按照以下目录结构组织：

```
A9_coco/
├── images/
│   ├── train/          # 训练集图像（所有图像放在这里）
│   ├── val/            # 验证集图像（可选）
│   └── test/           # 测试集图像（可选）
└── annotations/
    ├── instances_train.json      # 训练集标注文件
    ├── instances_val.json        # 验证集标注文件（可选）
    └── image_info_test.json      # 测试集信息文件（可选）
```

## 对于A9数据集

由于A9数据集目前只有训练数据，建议的结构是：

```
A9_coco/
├── images/
│   └── train/          # 所有4202张图像
└── annotations/
    └── instances_train.json      # 合并后的标注文件（已生成：instances_train.json）
```

## 具体操作步骤

### 1. 创建目录结构

```bash
mkdir -p datasets/A9_coco/images/train
mkdir -p datasets/A9_coco/annotations
```

### 2. 复制图像文件

需要将所有图像从原始A9数据集复制到 `datasets/A9_coco/images/train/` 目录。

图像分布在以下位置：
- `datasets/ A9/a9_dataset_r02_s01/images/s110_camera_basler_south1_8mm/`
- `datasets/ A9/a9_dataset_r02_s01/images/s110_camera_basler_south2_8mm/`
- `datasets/ A9/a9_dataset_r02_s02/images/s110_camera_basler_south1_8mm/`
- `datasets/ A9/a9_dataset_r02_s02/images/s110_camera_basler_south2_8mm/`
- `datasets/ A9/a9_dataset_r02_s03/images/s110_camera_basler_south1_8mm/`
- `datasets/ A9/a9_dataset_r02_s03/images/s110_camera_basler_south2_8mm/`
- `datasets/ A9/a9_dataset_r02_s04/images/s110_camera_basler_south1_8mm/`
- `datasets/ A9/a9_dataset_r02_s04/images/s110_camera_basler_south2_8mm/`

可以使用以下命令批量复制：

```bash
# 方法1：使用find和cp
find "datasets/ A9" -name "*.jpg" -path "*/images/s110_camera_*/*" \
    -exec cp {} datasets/A9_coco/images/train/ \;

# 方法2：使用rsync（推荐，可以跳过已存在的文件）
rsync -av --ignore-existing \
    "datasets/ A9/"*/images/s110_camera_*/ \
    datasets/A9_coco/images/train/
```

### 3. 复制标注文件

```bash
cp instances_train.json datasets/A9_coco/annotations/instances_train.json
```

## 验证结构

完成后，目录结构应该是：

```
datasets/A9_coco/
├── images/
│   └── train/
│       ├── 1646667332_139503285_s110_camera_basler_south1_8mm.jpg
│       ├── 1646667317_755295260_s110_camera_basler_south1_8mm.jpg
│       └── ... (共4202张图像)
└── annotations/
    └── instances_train.json
```

## COCO JSON文件格式说明

`instances_train.json` 文件包含以下字段：

```json
{
  "info": {
    "description": "数据集描述",
    "version": "版本号"
  },
  "licenses": [],
  "images": [
    {
      "id": 1,
      "file_name": "图像文件名.jpg",
      "width": 1920,
      "height": 1200
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [x, y, width, height],  // COCO格式：[左上角x, 左上角y, 宽度, 高度]
      "area": 面积,
      "iscrowd": 0
    }
  ],
  "categories": [
    {
      "id": 1,
      "name": "car",
      "supercategory": "vehicle"
    }
  ]
}
```

## 注意事项

1. **图像文件名**：COCO格式中，`images` 数组里的 `file_name` 应该直接是文件名（不含路径），因为图像都在 `images/train/` 目录下。

2. **路径关系**：如果图像在 `images/train/image.jpg`，那么JSON中的 `file_name` 应该是 `image.jpg`（不是 `train/image.jpg`）。

3. **图像ID唯一性**：每个图像必须有唯一的 `id`，标注中的 `image_id` 必须对应图像的 `id`。

4. **标注ID唯一性**：每个标注必须有唯一的 `id`。

## 使用合并后的JSON文件

当前已生成的 `instances_train.json` 文件：
- 位置：项目根目录
- 包含：4202张图像，13141个标注，8个类别
- 可以直接复制到 `datasets/A9_coco/annotations/` 目录使用

## 验证数据集

可以使用以下命令验证：

```bash
# 检查图像数量
ls datasets/A9_coco/images/train/*.jpg | wc -l
# 应该输出：4202

# 检查JSON文件
python3 -c "
import json
with open('datasets/A9_coco/annotations/instances_train.json') as f:
    data = json.load(f)
    print(f'Images: {len(data[\"images\"])}')
    print(f'Annotations: {len(data[\"annotations\"])}')
"
```

