import os
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
from pathlib import Path

def _default_project_dir() -> Path:
    """返回工程化的默认输出根目录: <当前脚本>/runs/detect"""
    return Path(__file__).resolve().parent / "runs" / "detect"


def _make_exp_name(model: str, data: str) -> str:
    """根据模型名与数据集名拼装更有可追踪性的实验名。"""
    model_stem = Path(model).stem
    data_stem = Path(data).stem
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{model_stem}_{data_stem}_{ts}"


def _resolve_data_path(data: str) -> str:
    p = Path(data)
    if p.is_file():
        return str(p)
    # 尝试从 experiments/datasets/ 解析
    cand = Path(__file__).resolve().parent.parent / "datasets" / data
    if cand.is_file():
        return str(cand)
    return data


def train_yolo(model="yolov8n.pt", data="coco8.yaml", epochs=1, imgsz=640, batch_size=8, project: Path | None = None, name: str | None = None):
    """训练 YOLO 模型，返回(best.pt 路径, 实验名)。"""
    print(f"[INFO] 开始训练 {model} ...")

    project_dir = Path(project) if project is not None else _default_project_dir()
    exp_name = name or _make_exp_name(model, data)

    # 加载模型
    yolo_model = YOLO(model)

    # 训练模型（将输出定向到工程化目录）
    results = yolo_model.train(
        data=_resolve_data_path(data),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        project=str(project_dir),
        name=exp_name,
        exist_ok=True,
        plots=False, # 训练阶段不生成图表
    )

    # 解析 best.pt 的真实路径
    save_dir = Path(getattr(results, "save_dir", project_dir / exp_name))
    best_pt = save_dir / "weights" / "best.pt"

    print("[INFO] 训练完成")
    return best_pt, exp_name

def validate_yolo(model="yolov8n.pt", data="coco8.yaml", imgsz=640, batch_size=8, project: Path | None = None, name: str | None = None):
    """评估 YOLO 模型，验证输出同样按工程化目录组织。"""
    print(f"[INFO] 开始验证 {model} ...")

    project_dir = Path(project) if project is not None else _default_project_dir()
    exp_name = name or _make_exp_name(str(model), data)

    # 加载模型
    yolo_model = YOLO(str(model))

    # 验证模型（输出到与训练同级的目录）
    results = yolo_model.val(
        data=_resolve_data_path(data),
        imgsz=imgsz,
        batch=batch_size,
        project=str(project_dir),
        name=f"{exp_name}_val",
        exist_ok=True,
    )

    # 输出评估指标（例如精度、mAP 等）
    try:
        print(f"[INFO] mAP@0.5: {results.metrics['mAP_0.5']}")
        print(f"[INFO] mAP@0.5:0.95: {results.metrics['mAP_0.5:0.95']}")
    except Exception:
        pass

    print("[INFO] 验证完成")

def main():
    # 统一的工程化输出根目录: <当前脚本>/runs/detect
    project_dir = _default_project_dir()
    project_dir.mkdir(parents=True, exist_ok=True)

    # 训练 YOLO 模型（返回 best.pt 路径与实验名）
    best_ckpt, exp_name = train_yolo(
        model="yolov8n.pt",
        data="coco8.yaml",
        epochs=1,
        imgsz=640,
        batch_size=8,
        project=project_dir,
    )

    # 验证 YOLO 模型（使用训练产出的 best.pt）
    validate_yolo(
        model=best_ckpt,
        data="coco8.yaml",
        imgsz=640,
        batch_size=8,
        project=project_dir,
        name=exp_name,
    )

if __name__ == "__main__":
    main()
