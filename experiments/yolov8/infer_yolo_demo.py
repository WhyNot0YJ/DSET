from ultralytics import YOLO
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", default="https://ultralytics.com/images/bus.jpg")
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--name", default="custom")
    args = ap.parse_args()

    print("[INFO] using ultralytics from editable source...")
    m = YOLO(args.model)
    # 将 YOLOv8 默认输出放到 runs/detect/yolov8
    m.predict(source=args.img, imgsz=640, save=True, project="runs/detect", name=f"yolov8_{args.name}")

if __name__ == "__main__":
    main()
