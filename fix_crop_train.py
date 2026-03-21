import glob
import re

def remove_crops_py(p):
    with open(p, 'r') as f:
        c = f.read()

    # Remove aug_crop_min assignments
    c = re.sub(r' *aug_crop_min *= *aug_config\.get\(\'crop_min\'[^\n]*\n?', '', c)
    c = re.sub(r' *aug_crop_max *= *aug_config\.get\(\'crop_max\'[^\n]*\n?', '', c)
    
    # Remove kwargs in dataset init
    c = re.sub(r' *aug_crop_min=[^,\n]*,*\n?', '', c)
    c = re.sub(r' *aug_crop_max=[^,\n]*,*\n?', '', c)
    
    # Remove strings 'crop_min': 0.1
    c = re.sub(r' *\'crop_min\': *[0-9.]+,[^\n]*\n?', '', c)
    c = re.sub(r' *\'crop_max\': *[0-9.]+,[^\n]*\n?', '', c)

    with open(p, 'w') as f:
        f.write(c)

for p in glob.glob('experiments/*/train.py'):
    remove_crops_py(p)
    
