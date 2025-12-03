
import sys
import os
import torch
from torch.utils.data import DataLoader
import cv2
import numpy as np
from pathlib import Path
import torchvision.transforms as T

# Setup path to allow imports from src
project_root = Path(__file__).parent.resolve()
if str(os.getcwd()) not in sys.path:
    sys.path.insert(0, os.getcwd())
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root.parent))

try:
    from src.data.dataset.dairv2x_detection import DAIRV2XDetection
except ImportError:
    # Fallback if running from root
    sys.path.append(str(Path("experiments/rt-detr").resolve()))
    from src.data.dataset.dairv2x_detection import DAIRV2XDetection

# BaseCollateFunction might be needed if we strictly follow the code, 
# but for this debug script we can just make CustomCollateFunction a standalone class or inherit object
class CustomCollateFunction:
    def __call__(self, batch):
        images, targets = zip(*batch)
        
        # 1. 处理图像 (保持 Tensor 格式)
        if not isinstance(images[0], torch.Tensor):
            processed_images = [T.functional.to_tensor(img) for img in images]
        else:
            processed_images = list(images)

        # 2. 计算 Batch 最大尺寸
        sizes = [img.shape[-2:] for img in processed_images]
        stride = 32
        max_h_raw = max(s[0] for s in sizes)
        max_w_raw = max(s[1] for s in sizes)
        # 向上取整到 32 倍数
        max_h = (max_h_raw + stride - 1) // stride * stride
        max_w = (max_w_raw + stride - 1) // stride * stride
        
        # 3. 创建画布并填充 (左上角对齐)
        batch_images = torch.zeros(len(processed_images), 3, max_h, max_w, 
                                   dtype=processed_images[0].dtype)
        
        for i, img in enumerate(processed_images):
            h, w = img.shape[-2:]
            batch_images[i, :, :h, :w] = img
            
        # 4. 根据最终 Batch 尺寸进行归一化
        new_targets = []
        for t in list(targets):
            # 复制 target 防止原地修改污染数据
            new_t = t.copy()
            boxes = new_t['boxes'] # 此时是绝对坐标 cx, cy, w, h
            
            # 手动归一化：除以 max_w 和 max_h
            # 格式是 cx, cy, w, h
            boxes[:, 0] = boxes[:, 0] / max_w
            boxes[:, 1] = boxes[:, 1] / max_h
            boxes[:, 2] = boxes[:, 2] / max_w
            boxes[:, 3] = boxes[:, 3] / max_h
            
            # 限制数值在 0-1 之间 (防止浮点溢出)
            boxes = torch.clamp(boxes, 0.0, 1.0)
            
            new_t['boxes'] = boxes
            new_targets.append(new_t)
        
        return batch_images, new_targets

def visualize_gt(loader, num_images=5):
    # 创建输出目录
    output_dir = Path("experiments/rt-detr/debug_vis")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"开始可视化 {num_images} 张 GT 图像...")
    print(f"结果将保存到: {output_dir.resolve()}")

    for i, (images, targets) in enumerate(loader):
        if i >= num_images: break
        
        # 取第一张图 (BatchSize=4 中的第0张)
        img_tensor = images[0] # [3, H, W]
        target = targets[0]
        
        # 反归一化图片
        # 注意：train.py 中 preprocess_image 使用了 normalize
        # DAIRV2XDetection dataset 应该输出了 normalized tensor (如果 transforms 包含 normalize)
        # 我们先假设 dataset 输出的是 [0,1] 的 tensor 或者经过 normalize 的
        # 检查 CustomCollateFunction，它处理的是 dataset 的输出。
        # 如果 dataset 输出的是 PIL 或未 normalized 的 Tensor，CustomCollateFunction 里的 to_tensor 会转为 [0,1]
        # 但 DAIRV2XDetection 通常会在内部做 transform。
        
        # 这里我们假设是 ImageNet normalize 过的
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        # 反归一化
        img = img_tensor * std + mean
        img = torch.clamp(img, 0, 1)
        
        # 转 numpy
        img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        H, W = img_np.shape[:2]
        print(f"Image {i}: {W}x{H}")
        
        # 画 GT 框
        boxes = target['boxes']
        print(f"GT Boxes: {len(boxes)}")
        
        for box in boxes:
            # CustomCollate 已经归一化到了 [0, 1] (相对于 padded image size H, W)
            # 所以直接乘 H, W 还原
            cx, cy, w, h = box.tolist()
            
            # cx, cy, w, h 是归一化的，还原为绝对坐标
            abs_cx = cx * W
            abs_cy = cy * H
            abs_w = w * W
            abs_h = h * H
            
            x1 = int(abs_cx - abs_w/2)
            y1 = int(abs_cy - abs_h/2)
            x2 = int(abs_cx + abs_w/2)
            y2 = int(abs_cy + abs_h/2)
            
            cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2) # 绿色框
            
        output_filename = output_dir / f"debug_gt_{i}.jpg"
        cv2.imwrite(str(output_filename), img_np)
        print(f"Saved {output_filename}")

if __name__ == "__main__":
    # 数据集路径
    # 注意：确保路径正确。用户环境是 /root/autodl-tmp/datasets/DAIR-V2X
    # 或者使用 train.py 默认的 datasets/DAIR-V2X
    # 我们使用 train.py 默认值或者用户提供的值
    data_root = "datasets/DAIR-V2X" 
    # 如果本地没有，尝试绝对路径
    if not os.path.exists(data_root):
        data_root = "/root/autodl-tmp/datasets/DAIR-V2X"
        
    if not os.path.exists(data_root):
        print(f"Warning: Data root {data_root} does not exist.")
    
    print(f"使用数据集路径: {data_root}")

    # 使用验证集配置 (不带 RandomCrop, 只有 Resize 和 Normalize)
    # DAIRV2XDetection 的 transform 逻辑在内部
    # target_size=640 (train.py 中 val_dataset 使用的是 640?)
    # 在 train.py:
    # val_dataset = DAIRV2XDetection(..., target_size=640, ...)
    
    dataset = DAIRV2XDetection(
        data_root=data_root, 
        split='val', 
        target_size=640, # 保持与 train.py 一致
        aug_brightness=0.0,
        aug_contrast=0.0,
        aug_saturation=0.0,
        aug_hue=0.0,
        aug_color_jitter_prob=0.0
    )
    
    loader = DataLoader(dataset, batch_size=4, collate_fn=CustomCollateFunction(), shuffle=False)
    visualize_gt(loader)

