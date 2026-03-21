import glob
import re

def remove_crops_dataset(p):
    with open(p, 'r') as f:
        c = f.read()

    # Remove aug_crop_min and aug_crop_max arguments
    c = re.sub(r' *aug_crop_min: float = 0\.3,\n?', '', c)
    c = re.sub(r' *aug_crop_max: float = 1\.0,\n?', '', c)
    
    # Remove RandomIoUCrop
    c = re.sub(r' *RandomIoUCrop\([^)]+\),\n?', '', c)

    with open(p, 'w') as f:
        f.write(c)

for p in glob.glob('experiments/*/src/data/dataset/dairv2x_detection.py'):
    remove_crops_dataset(p)
    
