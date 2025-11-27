# YOLOv8模型重新评估指南（max_det=100）

## 操作步骤

### 方法1：批量评估（推荐）

直接运行批量评估脚本，会自动评估所有3个模型：

```bash
cd /home/yujie/proj/task-selective-det/experiments/yolov8

bash reevaluate_all_models_maxdet100.sh
```

这个脚本会：
1. 自动找到所有 `best_model_*.pth` 文件
2. 使用 `max_det=100` 重新评估每个模型
3. 将结果保存到 `logs/reevaluation_maxdet100/` 目录

### 方法2：单个模型评估

如果你想单独评估某个模型：

```bash
cd /home/yujie/proj/task-selective-det/experiments/yolov8

# 评估YOLOv8-L
python3 reevaluate_with_maxdet100.py \
    --model best_model_yolov8l.pth \
    --data /root/autodl-fs/datasets/DAIR-V2X_YOLO/dairv2x.yaml \
    --max_det 100 \
    --device cuda

# 评估YOLOv8-M
python3 reevaluate_with_maxdet100.py \
    --model best_model_yolov8m.pth \
    --data /root/autodl-fs/datasets/DAIR-V2X_YOLO/dairv2x.yaml \
    --max_det 100 \
    --device cuda

# 评估YOLOv8-S
python3 reevaluate_with_maxdet100.py \
    --model best_model_yolov8s.pth \
    --data /root/autodl-fs/datasets/DAIR-V2X_YOLO/dairv2x.yaml \
    --max_det 100 \
    --device cuda
```

## 注意事项

1. **模型文件格式**：
   - 你的模型文件是 `.pth` 格式
   - YOLOv8通常使用 `.pt` 格式
   - 如果加载失败，可能需要重命名为 `.pt` 或检查文件格式

2. **数据配置文件路径**：
   - 默认路径：`/root/autodl-fs/datasets/DAIR-V2X_YOLO/dairv2x.yaml`
   - 如果路径不同，请修改脚本中的路径

3. **评估时间**：
   - 每个模型评估大约需要几分钟
   - 取决于验证集大小和GPU性能

4. **结果保存**：
   - 评估结果会打印到控制台
   - 批量评估时，结果也会保存到日志文件

## 预期结果

使用 `max_det=100` 后，mAP可能会下降2-5%，因为：
- 检测框数量从300减少到100
- 在目标密集场景中可能漏检部分目标
- 但这使得对比更公平（与DETR系列的100个查询对齐）

## 查看结果

评估完成后，查看结果：

```bash
# 查看所有评估结果
cat logs/reevaluation_maxdet100/*.log | grep -A 10 "评估结果"

# 或者查看特定模型的结果
cat logs/reevaluation_maxdet100/best_model_yolov8l_maxdet100.log
```

