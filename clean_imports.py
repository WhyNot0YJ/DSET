import os
projects = ['cas_detr', 'rt-detr', 'moe-rtdetr']
for proj in projects:
    path = f'experiments/{proj}/src/data/dataset/dairv2x_detection.py'
    if os.path.exists(path):
        with open(path, 'r') as f:
            content = f.read()
        # 清除不再使用的 RandomResize 导入
        content = content.replace("RandomResize, ConvertBoxes", "ConvertBoxes")
        with open(path, 'w') as f:
            f.write(content)
        print(f"Cleaned RandomResize in {proj}")
