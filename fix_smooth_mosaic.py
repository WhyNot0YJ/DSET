import glob
import re

for d in ['cas_detr', 'rt-detr', 'moe-rtdetr']:
    # Update configs
    conf_paths = glob.glob(f'experiments/{d}/configs/*.yaml')
    for p in conf_paths:
        with open(p, 'r') as f:
            c = f.read()
            c = re.sub(r'mosaic:\s*0\.0', 'mosaic: 1.0', c)
        with open(p, 'w') as f:
            f.write(c)

print("Done")
