import os
import re

projects = ['rt-detr', 'moe-rtdetr']
for proj in projects:
    # 1. Update dairv2x_detection.py imports
    path = f'experiments/{proj}/src/data/dataset/dairv2x_detection.py'
    if os.path.exists(path):
        with open(path, 'r') as f: content = f.read()
        content = content.replace("RandomResize, ConvertBoxes, Normalize, SanitizeBoundingBoxes\n)",
                                "RandomResize, ConvertBoxes, Normalize, SanitizeBoundingBoxes, PadToSize\n)")
        
        # Replace train transforms
        old_train = """                    RandomResize(scales=scales, max_size=train_max_size),
                    SanitizeBoundingBoxes(),
                    T.ToImage(),
                    T.ToDtype(torch.float32, scale=True),
                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ConvertBoxes(fmt='cxcywh', normalize=False)"""
        new_train = """                    T.Resize(size=640, max_size=640, antialias=True),
                    SanitizeBoundingBoxes(),
                    T.ToImage(),
                    T.ToDtype(torch.float32, scale=True),
                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    PadToSize(size=(640, 640), fill=0.0),
                    ConvertBoxes(fmt='cxcywh', normalize=False)"""
        content = content.replace(old_train, new_train)
        
        # Replace val transforms
        old_val = """                self.transforms = T.Compose([
                    T.Resize(size=720, max_size=1280, antialias=True),
                    T.ToImage(),
                    T.ToDtype(torch.float32, scale=True),
                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ConvertBoxes(fmt='cxcywh', normalize=False)
                ])"""
        new_val = """                self.transforms = T.Compose([
                    T.Resize(size=640, max_size=640, antialias=True),
                    T.ToImage(),
                    T.ToDtype(torch.float32, scale=True),
                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    PadToSize(size=(640, 640), fill=0.0),
                    ConvertBoxes(fmt='cxcywh', normalize=False)
                ])"""
        content = content.replace(old_val, new_val)
        with open(path, 'w') as f: f.write(content)

    # 2. Update batch_inference.py
    path = f'experiments/{proj}/batch_inference.py'
    if os.path.exists(path):
        with open(path, 'r') as f: content = f.read()
        old_infer_1 = """    orig_w, orig_h = image_pil.size
    im_size_min = min(orig_h, orig_w)
    im_size_max = max(orig_h, orig_w)
    target_short = 720
    scale = target_short / float(im_size_min)
    if round(scale * im_size_max) > target_size:
        scale = target_size / float(im_size_max)
    
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))
    
    resized_pil = image_pil.resize((new_w, new_h), resample=Image.BILINEAR)
    image_tensor = T.functional.to_tensor(resized_pil) 
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image_tensor = (image_tensor - mean) / std
    
    stride = 32
    padded_h = int(np.ceil(new_h / stride) * stride)
    padded_w = int(np.ceil(new_w / stride) * stride)
    
    padded_image = torch.zeros(3, padded_h, padded_w, dtype=torch.float32)
    padded_image[:, :new_h, :new_w] = image_tensor
    img_input = padded_image.unsqueeze(0)
    
    image_bgr_vis = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    
    meta = {
        'orig_size': torch.tensor([[orig_h, orig_w]]),
        'scale': scale,
        'padded_h': padded_h,
        'padded_w': padded_w,"""
        new_infer_1 = """    orig_w, orig_h = image_pil.size
    im_size_max = max(orig_h, orig_w)
    target_max = 640
    scale = target_max / float(im_size_max)
    
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))
    
    resized_pil = image_pil.resize((new_w, new_h), resample=Image.BILINEAR)
    image_tensor = T.functional.to_tensor(resized_pil) 
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image_tensor = (image_tensor - mean) / std
    
    padded_h = 640
    padded_w = 640
    
    padded_image = torch.zeros(3, padded_h, padded_w, dtype=torch.float32)
    padded_image[:, :new_h, :new_w] = image_tensor
    img_input = padded_image.unsqueeze(0)
    
    image_bgr_vis = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    
    meta = {
        'orig_size': torch.tensor([[orig_h, orig_w]]),
        'scale': scale,
        'padded_h': padded_h,
        'padded_w': padded_w,"""
        content = content.replace(old_infer_1, new_infer_1)
        
        old_infer_2 = """    orig_image = cv2.imread(str(orig_image_path))
    if orig_image is None:
        return None
    
    orig_h, orig_w = orig_image.shape[:2]
    input_h, input_w = img_tensor.shape[-2:]
    
    im_size_min = min(orig_h, orig_w)
    im_size_max = max(orig_h, orig_w)
    scale = 720 / float(im_size_min)
    if round(scale * im_size_max) > 1280:
        scale = 1280 / float(im_size_max)
        
    meta = {
        'orig_size': torch.tensor([[orig_h, orig_w]]),\n"""
        new_infer_2 = """    orig_image = cv2.imread(str(orig_image_path))
    if orig_image is None:
        return None
    
    orig_h, orig_w = orig_image.shape[:2]
    input_h, input_w = img_tensor.shape[-2:]
    
    im_size_max = max(orig_h, orig_w)
    scale = 640 / float(im_size_max)
        
    meta = {
        'orig_size': torch.tensor([[orig_h, orig_w]]),\n"""
        content = content.replace(old_infer_2, new_infer_2)
        with open(path, 'w') as f: f.write(content)

    # 3. Update train.py eval mappings
    path = f'experiments/{proj}/train.py'
    if os.path.exists(path):
        with open(path, 'r') as f: content = f.read()
        
        old_train_1 = """                # 转换为COCO格式
                if filtered_boxes.shape[0] > 0:
                    boxes_coco = torch.zeros_like(filtered_boxes)
                    if filtered_boxes.max() <= 1.0:
                        # 归一化坐标 -> 像素坐标
                        boxes_coco[:, 0] = (filtered_boxes[:, 0] - filtered_boxes[:, 2] / 2) * img_w
                        boxes_coco[:, 1] = (filtered_boxes[:, 1] - filtered_boxes[:, 3] / 2) * img_h
                        boxes_coco[:, 2] = filtered_boxes[:, 2] * img_w
                        boxes_coco[:, 3] = filtered_boxes[:, 3] * img_h
                    else:
                        boxes_coco = filtered_boxes.clone()
                    
                    # Clamp坐标
                    boxes_coco[:, 0] = torch.clamp(boxes_coco[:, 0], 0, img_w)
                    boxes_coco[:, 1] = torch.clamp(boxes_coco[:, 1], 0, img_h)
                    boxes_coco[:, 2] = torch.clamp(boxes_coco[:, 2], 1, img_w)
                    boxes_coco[:, 3] = torch.clamp(boxes_coco[:, 3], 1, img_h)"""
                    
        new_train_1 = """                # 转换为COCO格式
                if filtered_boxes.shape[0] > 0:
                    # 获取原始图像尺寸，映射回原始比例
                    orig_h, orig_w = targets[i]['orig_size'].tolist()
                    scale = img_w / float(max(orig_h, orig_w))  # img_w 实际上是 640
                    
                    boxes_coco = torch.zeros_like(filtered_boxes)
                    if filtered_boxes.max() <= 1.0:
                        # 归一化坐标 -> padded像素坐标 -> 原始图像坐标
                        boxes_coco[:, 0] = ((filtered_boxes[:, 0] - filtered_boxes[:, 2] / 2) * img_w) / scale
                        boxes_coco[:, 1] = ((filtered_boxes[:, 1] - filtered_boxes[:, 3] / 2) * img_h) / scale
                        boxes_coco[:, 2] = (filtered_boxes[:, 2] * img_w) / scale
                        boxes_coco[:, 3] = (filtered_boxes[:, 3] * img_h) / scale
                    else:
                        boxes_coco = filtered_boxes.clone() / scale
                    
                    # Clamp坐标
                    boxes_coco[:, 0] = torch.clamp(boxes_coco[:, 0], 0, orig_w)
                    boxes_coco[:, 1] = torch.clamp(boxes_coco[:, 1], 0, orig_h)
                    boxes_coco[:, 2] = torch.clamp(boxes_coco[:, 2], 1, orig_w)
                    boxes_coco[:, 3] = torch.clamp(boxes_coco[:, 3], 1, orig_h)"""
        
        old_train_2 = """                    true_boxes_coco = torch.zeros_like(true_boxes)
                    if max_val <= 1.0 + 1e-6:
                        true_boxes_coco[:, 0] = (true_boxes[:, 0] - true_boxes[:, 2] / 2) * img_w
                        true_boxes_coco[:, 1] = (true_boxes[:, 1] - true_boxes[:, 3] / 2) * img_h
                        true_boxes_coco[:, 2] = true_boxes[:, 2] * img_w
                        true_boxes_coco[:, 3] = true_boxes[:, 3] * img_h
                    else:
                        true_boxes_coco = true_boxes.clone()
                    
                    true_boxes_coco[:, 0] = torch.clamp(true_boxes_coco[:, 0], 0, img_w)
                    true_boxes_coco[:, 1] = torch.clamp(true_boxes_coco[:, 1], 0, img_h)
                    true_boxes_coco[:, 2] = torch.clamp(true_boxes_coco[:, 2], 1, img_w)
                    true_boxes_coco[:, 3] = torch.clamp(true_boxes_coco[:, 3], 1, img_h)"""
        
        new_train_2 = """                    # 获取原始图像尺寸，映射回原始比例
                    orig_h, orig_w = targets[i]['orig_size'].tolist()
                    scale = img_w / float(max(orig_h, orig_w))  # img_w 是 padded_size
                    
                    true_boxes_coco = torch.zeros_like(true_boxes)
                    if max_val <= 1.0 + 1e-6:
                        true_boxes_coco[:, 0] = ((true_boxes[:, 0] - true_boxes[:, 2] / 2) * img_w) / scale
                        true_boxes_coco[:, 1] = ((true_boxes[:, 1] - true_boxes[:, 3] / 2) * img_h) / scale
                        true_boxes_coco[:, 2] = (true_boxes[:, 2] * img_w) / scale
                        true_boxes_coco[:, 3] = (true_boxes[:, 3] * img_h) / scale
                    else:
                        true_boxes_coco = true_boxes.clone() / scale
                    
                    true_boxes_coco[:, 0] = torch.clamp(true_boxes_coco[:, 0], 0, orig_w)
                    true_boxes_coco[:, 1] = torch.clamp(true_boxes_coco[:, 1], 0, orig_h)
                    true_boxes_coco[:, 2] = torch.clamp(true_boxes_coco[:, 2], 1, orig_w)
                    true_boxes_coco[:, 3] = torch.clamp(true_boxes_coco[:, 3], 1, orig_h)"""
        
        content = content.replace(old_train_1, new_train_1)
        content = content.replace(old_train_2, new_train_2)
        with open(path, 'w') as f: f.write(content)
