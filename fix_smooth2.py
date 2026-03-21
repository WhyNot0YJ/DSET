import re
import glob

def fix(p):
    with open(p, 'r') as f:
        c = f.read()
    
    # 替换 F.one_hot 在 loss_labels_vfl
    if 'sm = getattr(self, "label_smoothing", 0.01)' not in c:
        p1 = r'(target = F\.one_hot\(target_classes, num_classes=self\.num_classes \+ 1\)\[\.\.\., :\-1\])'
        r1 = r'\1\n        sm = getattr(self, "label_smoothing", 0.01)\n        if sm > 0.0:\n            target = target * (1.0 - sm) + sm / self.num_classes'
        c = re.sub(p1, r1, c)
        
        p2 = r'(target = F\.one_hot\(target_classes, num_classes=self\.num_classes \+ 1\)\[\.\.\., :\-1\]\.to\(src_logits\.dtype\))'
        c = re.sub(p2, r1, c)

        # 查找 __init__
        p3 = r'(def __init__\(self, weight_dict, num_classes,.*?)(\):)'
        r3 = r'\1, label_smoothing: float = 0.01\2\n        self.label_smoothing = label_smoothing'
        c = re.sub(p3, r3, c, flags=re.DOTALL)

    with open(p, 'w') as f:
        f.write(c)

for d in ['cas_detr', 'rt-detr', 'moe-rtdetr']:
    for p in glob.glob(f'experiments/{d}/src/nn/criterion/*criterion*.py'):
        fix(p)
