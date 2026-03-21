import glob

def fix(p):
    with open(p, 'r') as f:
        c = f.read()
    
    # Label vfl
    old_vfl = "        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]"
    new_vfl = """        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]
        
        # Apply label smoothing
        sm = getattr(self, 'label_smoothing', 0.01)
        if sm > 0.0:
            target = target.float() * (1.0 - sm) + sm / self.num_classes"""
    c = c.replace(old_vfl, new_vfl)

    # Label focal
    old_focal = "        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1].to(src_logits.dtype)"
    new_focal = """        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1].to(src_logits.dtype)
        
        # Apply label smoothing
        sm = getattr(self, 'label_smoothing', 0.01)
        if sm > 0.0:
            target = target.float() * (1.0 - sm) + sm / self.num_classes"""
    c = c.replace(old_focal, new_focal)

    with open(p, 'w') as f:
        f.write(c)

for d in ['cas_detr', 'rt-detr', 'moe-rtdetr']:
    for p in glob.glob(f'experiments/{d}/src/nn/criterion/*criterion*.py'):
        fix(p)
