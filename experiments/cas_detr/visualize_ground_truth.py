#!/usr/bin/env python3
"""
Visualize Ground Truth Annotations
Useful for checking dataset annotations for errors or misalignment.
"""

import sys
import cv2
import numpy as np
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Dict

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from src.data.transforms.letterbox_geom import compute_letterbox_layout

# DAIR-V2X Classes
CLASS_NAMES = [
    "Car", "Truck", "Van", "Bus", "Pedestrian", 
    "Cyclist", "Motorcyclist", "Trafficcone"
]

# Colors (BGR format for OpenCV)
COLORS = [
    (255, 0, 0),      # Car - Blue
    (0, 255, 0),      # Truck - Green
    (255, 255, 0),    # Van - Cyan
    (0, 0, 255),      # Bus - Red
    (255, 0, 255),    # Pedestrian - Magenta
    (0, 255, 255),    # Cyclist - Yellow
    (128, 0, 128),   # Motorcyclist - Purple
    (255, 165, 0),   # Trafficcone - Orange
]

# Ignore Classes
IGNORE_CLASSES = [
    "PedestrianIgnore", "CarIgnore", "OtherIgnore", 
    "Unknown_movable", "Unknown_unmovable"
]


def load_annotations(annotation_path: Path) -> List[Dict]:
    """Load annotation file."""
    if not annotation_path.exists():
        return []
    
    with open(annotation_path, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    class_to_id = {name: i for i, name in enumerate(CLASS_NAMES)}
    processed_annotations = []
    
    for ann in annotations:
        class_name = ann["type"]
        
        # Skip ignore classes
        if class_name in IGNORE_CLASSES:
            continue
        
        # Get 2D BBox
        bbox_2d = ann["2d_box"]
        x1 = float(bbox_2d["xmin"])
        y1 = float(bbox_2d["ymin"])
        x2 = float(bbox_2d["xmax"])
        y2 = float(bbox_2d["ymax"])
        
        # Check if bbox is valid
        if x2 <= x1 or y2 <= y1:
            continue
        
        # Class mapping: Barrowlist -> Cyclist
        if class_name == "Barrowlist":
            class_id = 5  # Cyclist
        elif class_name in class_to_id:
            class_id = class_to_id[class_name]
        else:
            continue  # Skip unknown classes
        
        processed_annotations.append({
            'class_id': class_id,
            'class_name': CLASS_NAMES[class_id],
            'bbox': [x1, y1, x2, y2],  # [x1, y1, x2, y2] format
        })
    
    return processed_annotations


def draw_gt_boxes(image: np.ndarray, annotations: List[Dict], 
                  show_labels: bool = True, line_thickness: int = 2,
                  colors: List = None) -> np.ndarray:
    """Draw Ground Truth boxes on image.
    
    Args:
        image: BGR image
        annotations: List of annotations
        show_labels: Whether to show labels
        line_thickness: Thickness of bbox lines
        colors: BGR color list per class_id (default: COLORS)
    
    Returns:
        Image with GT boxes
    """
    image = image.copy()
    palette = colors if colors is not None else COLORS

    for ann in annotations:
        x1, y1, x2, y2 = map(int, ann['bbox'])
        class_id = ann['class_id']
        class_name = ann['class_name']
        
        # Clip to image bounds
        x1 = max(0, min(x1, image.shape[1] - 1))
        y1 = max(0, min(y1, image.shape[0] - 1))
        x2 = max(0, min(x2, image.shape[1] - 1))
        y2 = max(0, min(y2, image.shape[0] - 1))
        
        # Get Color
        color = palette[class_id] if class_id < len(palette) else (255, 255, 255)
        
        # Draw Rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), color, line_thickness)
        
        # Draw Label
        if show_labels:
            label_text = class_name
            (text_w, text_h), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            # Text Background
            cv2.rectangle(
                image,
                (x1, y1 - text_h - baseline - 4),
                (x1 + text_w, y1),
                color,
                -1
            )
            
            # Text
            cv2.putText(
                image,
                label_text,
                (x1, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
    
    return image


def build_letterbox_preview_bgr(
    image_bgr: np.ndarray,
    annotations: List[Dict],
    target_size: int = 640,
    fill: int = 0,
) -> np.ndarray:
    """Resize + center-pad like training; draw GT boxes in letterbox pixel space."""
    orig_h, orig_w = image_bgr.shape[:2]
    L = compute_letterbox_layout(orig_w, orig_h, target_size)
    resized = cv2.resize(
        image_bgr,
        (int(L["new_w"]), int(L["new_h"])),
        interpolation=cv2.INTER_LINEAR,
    )
    ch, cw = int(L["padded_h"]), int(L["padded_w"])
    canvas = np.full((ch, cw, 3), int(fill), dtype=np.uint8)
    pl, pt = int(L["pad_left"]), int(L["pad_top"])
    nh, nw = int(L["new_h"]), int(L["new_w"])
    canvas[pt : pt + nh, pl : pl + nw] = resized
    sc = float(L["scale"])
    ann_lb = []
    for ann in annotations:
        x1, y1, x2, y2 = ann["bbox"]
        ann_lb.append({
            **ann,
            "bbox": [
                x1 * sc + pl,
                y1 * sc + pt,
                x2 * sc + pl,
                y2 * sc + pt,
            ],
        })
    return draw_gt_boxes(canvas, ann_lb)


def visualize_single_image(
    data_root: Path,
    image_idx: int,
    output_path: Path = None,
    show: bool = True,
    letterbox_preview: bool = False,
    letterbox_target_size: int = 640,
    letterbox_fill: int = 0,
):
    """Visualize single image Ground Truth.
    
    Args:
        data_root: Dataset root
        image_idx: Image index
        output_path: Output path
        show: Whether to show image
    """
    # Build paths
    image_path = data_root / "image" / f"{image_idx:06d}.jpg"
    annotation_path = data_root / "annotations" / "camera" / f"{image_idx:06d}.json"
    
    if not image_path.exists():
        print(f"❌ Image not found: {image_path}")
        return
    
    # Load Image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"❌ Failed to read image: {image_path}")
        return
    
    # Load Annotations
    annotations = []
    if annotation_path.exists():
        annotations = load_annotations(annotation_path)
        print(f"✓ Loaded image: {image_path.name}")
        print(f"✓ Found {len(annotations)} boxes")
    else:
        print(f"⚠️  Annotation not found: {annotation_path}")
        print(f"✓ Loaded image: {image_path.name} (No annotations)")
    
    # Draw GT Boxes
    if annotations:
        image_with_boxes = draw_gt_boxes(image, annotations)
    else:
        image_with_boxes = image
    
    # Show Stats
    if annotations:
        print(f"\nAnnotation Stats:")
        class_counts = {}
        for ann in annotations:
            class_name = ann['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        for class_name, count in sorted(class_counts.items()):
            print(f"  {class_name}: {count}")
        
        # Y-coord Stats
        y_coords = []
        for ann in annotations:
            y1, y2 = ann['bbox'][1], ann['bbox'][3]
            y_coords.extend([y1, y2])
        if y_coords:
            print(f"\nY-coord Stats (H={image.shape[0]}):")
            print(f"  Min Y: {min(y_coords):.1f}")
            print(f"  Max Y: {max(y_coords):.1f}")
            print(f"  Mean Y: {np.mean(y_coords):.1f}")
            print(f"  Median Y: {np.median(y_coords):.1f}")
            print(f"  Boxes in upper half (0-{image.shape[0]//2}): {sum(1 for y in y_coords if y < image.shape[0]//2)}")
            print(f"  Boxes in lower half ({image.shape[0]//2}-{image.shape[0]}): {sum(1 for y in y_coords if y >= image.shape[0]//2)}")
    
    # Save or Show
    if output_path:
        cv2.imwrite(str(output_path), image_with_boxes)
        print(f"\n✓ Saved to: {output_path}")
        if letterbox_preview and annotations:
            lb_path = output_path.parent / f"{output_path.stem}_letterbox{output_path.suffix}"
            lb_img = build_letterbox_preview_bgr(
                image, annotations, letterbox_target_size, letterbox_fill
            )
            cv2.imwrite(str(lb_path), lb_img)
            print(f"✓ Letterbox preview saved to: {lb_path}")
    
    if show:
        # RGB for matplotlib
        image_rgb = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(16, 10))
        plt.imshow(image_rgb)
        plt.axis('off')
        plt.title(f"Ground Truth - {image_path.name} ({len(annotations) if annotations else 0} boxes)", 
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()


def visualize_multiple_images(data_root: Path, num_images: int = 10, 
                              start_idx: int = 0, output_dir: Path = None):
    """Visualize multiple images Ground Truth."""
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(num_images):
        image_idx = start_idx + i
        output_path = output_dir / f"gt_{image_idx:06d}.jpg" if output_dir else None
        print(f"\n{'='*60}")
        print(f"Image {i+1}/{num_images}: index={image_idx}")
        print(f"{'='*60}")
        visualize_single_image(data_root, image_idx, output_path, show=False)


def main():
    parser = argparse.ArgumentParser(description="Visualize Ground Truth Annotations")
    parser.add_argument("--data_root", type=str, required=True,
                       help="Dataset root directory")
    parser.add_argument("--image_idx", type=int, default=0,
                       help="Image index (Single mode)")
    parser.add_argument("--num_images", type=int, default=1,
                       help="Number of images (Multi mode)")
    parser.add_argument("--start_idx", type=int, default=0,
                       help="Start index (Multi mode)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output path (Single) or directory (Multi)")
    parser.add_argument("--no_show", action="store_true",
                       help="Do not show image")
    parser.add_argument(
        "--letterbox_preview",
        action="store_true",
        help="Also save training-style letterbox canvas with mapped GT boxes (needs --output)",
    )
    parser.add_argument("--letterbox_target_size", type=int, default=640,
                       help="Square size for letterbox preview (default 640)")
    parser.add_argument("--letterbox_fill", type=int, default=0,
                       help="Pad fill 0-255 (default 0 black)")

    args = parser.parse_args()
    
    data_root = Path(args.data_root)
    if not data_root.exists():
        print(f"❌ Dataset root not found: {data_root}")
        return
    
    if args.num_images == 1:
        # Single Mode
        output_path = Path(args.output) if args.output else None
        visualize_single_image(
            data_root,
            args.image_idx,
            output_path,
            show=not args.no_show,
            letterbox_preview=args.letterbox_preview,
            letterbox_target_size=args.letterbox_target_size,
            letterbox_fill=args.letterbox_fill,
        )
    else:
        # Multi Mode
        output_dir = Path(args.output) if args.output else None
        visualize_multiple_images(
            data_root,
            args.num_images,
            args.start_idx,
            output_dir
        )


if __name__ == "__main__":
    main()

