import os
import glob
import re

def update_criterion(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Add label_smoothing to __init__ if missing
    if 'def __init__' in content and 'label_smoothing: float = 0.0' not in content:
        content = re.sub(r'(def __init__\([^:]*?)(self,\s*weight_dict.*?):', r'\1\2, label_smoothing: float = 0.01:', content)
        content = re.sub(r'(def __init__\(.*?):', r'\1:\n        self.label_smoothing = label_smoothing', content, count=1)

    # Patch loss_labels_vfl
    vfl_pattern = r'(target = F\.one_hot\(target_classes, num_classes=self\.num_classes \+ 1\)\[\.\.\., :\-1\].*)(\n\s*target_score_o)'
    vfl_replace = r'\1\n        sm = getattr(self, "label_smoothing", 0.01)\n        if sm > 0.0:\n            target = target * (1.0 - sm) + sm / self.num_classes\n\2'
    content = re.sub(vfl_pattern, vfl_replace, content)

    # Patch loss_labels_focal
    focal_pattern = r'(target = F\.one_hot\(target_classes, num_classes=self\.num_classes \+ 1\)\[\.\.\., :\-1\]\.to\(src_logits\.dtype\).*)(\n\s*loss = )'
    focal_replace = r'\1\n        sm = getattr(self, "label_smoothing", 0.01)\n        if sm > 0.0:\n            target = target * (1.0 - sm) + sm / self.num_classes\n\2'
    content = re.sub(focal_pattern, focal_replace, content)

    with open(filepath, 'w') as f:
        f.write(content)

for d in ['cas_detr', 'rt-detr', 'moe-rtdetr']:
    # Update det_criterion.py instances
    crit_paths = glob.glob(f'experiments/{d}/src/nn/criterion/*criterion*.py')
    for p in crit_paths:
        update_criterion(p)
    
    # Update configs
    conf_paths = glob.glob(f'experiments/{d}/configs/*.yaml')
    for p in conf_paths:
        with open(p, 'r') as f:
            c = f.read()
            c = re.sub(r'mosaic:\s*0\.0', 'mosaic: 1.0', c)
        with open(p, 'w') as f:
            f.write(c)

print("Done")
