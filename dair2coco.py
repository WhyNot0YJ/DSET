import json
import os
import sys

# 尝试导入 tqdm
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterator): return iterator

# ================= 配置区域 =================
ROOT_PATH = "/root/autodl-tmp/datasets/DAIR-V2X"

# 1. 总索引文件 (包含所有图片信息)
DATA_INFO_PATH = "/root/autodl-tmp/datasets/DAIR-V2X/metadata/data_info.json"

# 2. 划分文件 (告诉我们要怎么分配) -> 请确认文件名是否正确，截图里是 split_data.json
SPLIT_FILE_PATH = os.path.join(ROOT_PATH, "metadata/split_data.json")

# 3. 输出文件路径
OUT_TRAIN_PATH = os.path.join(ROOT_PATH, "annotations/instances_train.json")
OUT_VAL_PATH = os.path.join(ROOT_PATH, "annotations/instances_val.json")

# 类别映射
CLASS_MAPPING = {
    "Car": 0, "Truck": 1, "Van": 2, "Bus": 3,
    "Pedestrian": 4, "Cyclist": 5, "Motorcyclist": 6, "Trafficcone": 7
}

def convert():
    # --- 第一步：读取划分信息 (split_data.json) ---
    print(f"正在读取划分文件: {SPLIT_FILE_PATH} ...")
    if not os.path.exists(SPLIT_FILE_PATH):
        print(f"[错误] 找不到 split_data.json，请检查路径！")
        return

    with open(SPLIT_FILE_PATH, 'r', encoding='utf-8') as f:
        split_data = json.load(f)
    
    # 提取 ID 列表，转成集合方便快速查找
    # 注意：split_data.json 里的 ID 是字符串 "000023" 还是数字？截图显示是字符串 "000023"
    train_ids = set(split_data.get('train', []))
    val_ids = set(split_data.get('val', []))
    
    print(f"  - 训练集数量: {len(train_ids)}")
    print(f"  - 验证集数量: {len(val_ids)}")

    # --- 第二步：读取总索引 (data_info.json) ---
    print(f"正在读取总索引: {DATA_INFO_PATH} ...")
    with open(DATA_INFO_PATH, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    if isinstance(raw_data, dict):
        data_list = list(raw_data.values())
    else:
        data_list = raw_data

    # --- 第三步：初始化两个 COCO 结构 ---
    # 定义通用的类别信息
    categories = [
        {"id": 0, "name": "Car"}, {"id": 1, "name": "Truck"},
        {"id": 2, "name": "Van"}, {"id": 3, "name": "Bus"},
        {"id": 4, "name": "Pedestrian"}, {"id": 5, "name": "Cyclist"},
        {"id": 6, "name": "Motorcyclist"}, {"id": 7, "name": "Trafficcone"}
    ]

    coco_train = {"images": [], "annotations": [], "categories": categories}
    coco_val = {"images": [], "annotations": [], "categories": categories}

    # 独立的计数器
    train_ann_id = 1
    val_ann_id = 1

    print(f"开始遍历 {len(data_list)} 张图片进行分拣...")
    print(f"强制读取标签路径: {ROOT_PATH}/annotations/camera/ ...")

    processed_count = 0

    for img_idx, item in enumerate(tqdm(data_list)):
        # 1. 获取图片 ID
        # image_path 格式通常是 "image/000001.jpg"
        # 我们需要提取 "000001" 这一部分来跟 split_data 进行匹配
        image_path = item.get('image_path', '')
        filename_with_ext = os.path.basename(image_path)      # 000001.jpg
        image_id_str = os.path.splitext(filename_with_ext)[0] # 000001
        
        # 2. 判断这张图属于哪个集
        target_coco = None
        target_ann_id = 0
        
        if image_id_str in train_ids:
            target_coco = coco_train
            target_ann_id = train_ann_id
        elif image_id_str in val_ids:
            target_coco = coco_val
            target_ann_id = val_ann_id
        else:
            # 既不在 train 也不在 val (可能是 test 集，或者是没被选中的数据)
            continue 

        # 3. 强制修正标签路径逻辑
        # 无论 data_info 里写什么，我们都去 annotations/camera/ 下找同名 json
        label_filename = image_id_str + ".json"
        label_path = os.path.join(ROOT_PATH, "annotations/camera", label_filename)

        if not os.path.exists(label_path):
            continue

        try:
            with open(label_path, 'r', encoding='utf-8') as lf:
                labels = json.load(lf)
        except Exception:
            continue
        
        processed_count += 1

        # 4. 写入 Image 信息
        width = 1920 
        height = 1080
        
        # 注意：Image ID 最好用数字。如果 image_id_str 是 "000001"，转 int 就是 1
        try:
            num_image_id = int(image_id_str)
        except:
            num_image_id = img_idx # 如果文件名不是纯数字，退化使用索引

        target_coco["images"].append({
            "id": num_image_id,
            "file_name": image_path,
            "width": width, "height": height
        })

        # 5. 写入 Annotation 信息
        for obj in labels:
            raw_type = obj.get('type')
            if raw_type not in CLASS_MAPPING: continue 
            
            bbox_2d = obj.get('2d_box', {})
            if not bbox_2d: continue

            try:
                xmin, ymin = float(bbox_2d['xmin']), float(bbox_2d['ymin'])
                xmax, ymax = float(bbox_2d['xmax']), float(bbox_2d['ymax'])
            except ValueError: continue

            w, h = xmax - xmin, ymax - ymin
            
            target_coco["annotations"].append({
                "id": target_ann_id, 
                "image_id": num_image_id,
                "category_id": CLASS_MAPPING[raw_type],
                "bbox": [xmin, ymin, w, h], "area": w * h,
                "iscrowd": 0, "segmentation": []
            })
            target_ann_id += 1
        
        # 更新回全局计数器
        if target_coco is coco_train:
            train_ann_id = target_ann_id
        else:
            val_ann_id = target_ann_id

    # --- 第四步：保存两个文件 ---
    os.makedirs(os.path.dirname(OUT_TRAIN_PATH), exist_ok=True)
    
    print(f"\n正在保存训练集: {OUT_TRAIN_PATH} ...")
    with open(OUT_TRAIN_PATH, 'w', encoding='utf-8') as out_f:
        json.dump(coco_train, out_f)
    print(f"  - 包含图片: {len(coco_train['images'])}")
    print(f"  - 包含标注: {len(coco_train['annotations'])}")

    print(f"\n正在保存验证集: {OUT_VAL_PATH} ...")
    with open(OUT_VAL_PATH, 'w', encoding='utf-8') as out_f:
        json.dump(coco_val, out_f)
    print(f"  - 包含图片: {len(coco_val['images'])}")
    print(f"  - 包含标注: {len(coco_val['annotations'])}")
    
    print(f"\n全部完成！总共处理有效图片: {processed_count} 张")

if __name__ == "__main__":
    convert()