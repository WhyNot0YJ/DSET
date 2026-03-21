import os
import glob
import re

def remove_crop_from_yaml(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Remove crop_min and crop_max lines from yaml
    content = re.sub(r'^\s*crop_min:\s*.*\n?', '', content, flags=re.MULTILINE)
    content = re.sub(r'^\s*crop_max:\s*.*\n?', '', content, flags=re.MULTILINE)
    
    with open(filepath, 'w') as f:
        f.write(content)

def remove_crop_from_python(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # In train.py: remove aug_config.get('crop_min') and similarly for crop_max
    content = re.sub(r'^\s*aug_crop_min\s*=\s*aug_config\.get\(\'crop_min\'.*\n?', '', content, flags=re.MULTILINE)
    content = re.sub(r'^\s*aug_crop_max\s*=\s*aug_config\.get\(\'crop_max\'.*\n?', '', content, flags=re.MULTILINE)
    content = re.sub(r'^\s*\'crop_min\':.*\n?', '', content, flags=re.MULTILINE)
    content = re.sub(r'^\s*\'crop_max\':.*\n?', '', content, flags=re.MULTILINE)
    
    # Remove kwarg passing
    content = re.sub(r'^\s*aug_crop_min\s*=.*,\n?', '', content, flags=re.MULTILINE)
    content = re.sub(r'^\s*aug_crop_max\s*=.*,\n?', '', content, flags=re.MULTILINE)
    
    with open(filepath, 'w') as f:
        f.write(content)

def remove_crop_from_dataset(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Remove aug_crop_min and aug_crop_max from __init__ kwargs
    content = re.sub(r'^\s*aug_crop_min:\s*float\s*=\s*0\.3,\n?', '', content, flags=re.MULTILINE)
    content = re.sub(r'^\s*aug_crop_max:\s*float\s*=\s*1\.0,\n?', '', content, flags=re.MULTILINE)
    
    # Also remove RandomIoUCrop in dairv2x_detection.py
    content = re.sub(r'^\s*RandomIoUCrop\(min_scale=aug_crop_min.*,\n?', '', content, flags=re.MULTILINE)
    
    with open(filepath, 'w') as f:
        f.write(content)


for d in ['cas_detr', 'rt-detr', 'moe-rtdetr']:
    for p in glob.glob(f'experiments/{d}/configs/*.yaml'):
        remove_crop_from_yaml(p)
    
    for p in glob.glob(f'experiments/{d}/train.py'):
        remove_crop_from_python(p)
        
    for p in glob.glob(f'experiments/{d}/src/data/dataset/dairv2x_detection.py'):
        remove_crop_from_dataset(p)

print("Done")
