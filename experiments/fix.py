import json

# 这里填你报错时正在用的验证集 json 路径
json_path = '/root/autodl-tmp/datasets/DAIR-V2X/annotations/instances_val.json'  # 举例，请修改

print(f"正在读取: {json_path}")
with open(json_path, 'r') as f:
    data = json.load(f)

if 'info' not in data:
    print("发现缺少 'info' 字段，正在修补...")
    data['info'] = {"description": "DAIR-V2X Fixed by AutoDL", "version": "1.0", "year": 2024}
    
    with open(json_path, 'w') as f:
        json.dump(data, f)
    print("修复完成！")
else:
    print("文件已有 'info' 字段，无需修复。")