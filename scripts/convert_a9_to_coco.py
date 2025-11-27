#!/usr/bin/env python3
"""Convert A9 OpenLABEL annotations to COCO (pure python, no numpy)."""
import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

CATEGORY_MAPPING = {
    "CAR": 1,
    "TRUCK": 2,
    "PEDESTRIAN": 3,
    "TRAILER": 4,
    "VAN": 5,
    "BUS": 6,
    "MOTORCYCLE": 7,
    "BICYCLE": 8,
}

CATEGORY_NAMES = {
    1: "car",
    2: "truck",
    3: "pedestrian",
    4: "trailer",
    5: "van",
    6: "bus",
    7: "motorcycle",
    8: "bicycle",
}


def matrix4x4_to_list(values: List[float]) -> List[List[float]]:
    return [list(values[i:i+4]) for i in range(0, 16, 4)]


def apply_transform(point: List[float], T: List[List[float]]) -> List[float]:
    x, y, z = point
    xp = T[0][0]*x + T[0][1]*y + T[0][2]*z + T[0][3]
    yp = T[1][0]*x + T[1][1]*y + T[1][2]*z + T[1][3]
    zp = T[2][0]*x + T[2][1]*y + T[2][2]*z + T[2][3]
    return [xp, yp, zp]


def cuboid_to_corners(cuboid: List[float]) -> List[List[float]]:
    if len(cuboid) < 10:
        raise ValueError("Cuboid needs >=10 values")
    cx, cy, cz = cuboid[0:3]
    val5, val6 = cuboid[5], cuboid[6]
    if abs(val5*val5 + val6*val6 - 1.0) < 0.1:
        yaw = math.atan2(val5, val6)
        qw = math.cos(yaw / 2.0)
        qx = qy = 0.0
        qz = math.sin(yaw / 2.0)
        length, width, height = cuboid[7:10]
    else:
        qx, qy, qz, qw = cuboid[3:7]
        length, width, height = cuboid[7:10]
    norm = math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    qw, qx, qy, qz = qw/norm, qx/norm, qy/norm, qz/norm
    # quaternion to rotation
    R = [
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)],
    ]
    hl, hw, hh = length/2.0, width/2.0, height/2.0
    local = [
        [-hl, -hw, -hh],
        [ hl, -hw, -hh],
        [ hl,  hw, -hh],
        [-hl,  hw, -hh],
        [-hl, -hw,  hh],
        [ hl, -hw,  hh],
        [ hl,  hw,  hh],
        [-hl,  hw,  hh],
    ]
    corners = []
    for lx, ly, lz in local:
        rx = R[0][0]*lx + R[0][1]*ly + R[0][2]*lz + cx
        ry = R[1][0]*lx + R[1][1]*ly + R[1][2]*lz + cy
        rz = R[2][0]*lx + R[2][1]*ly + R[2][2]*lz + cz
        corners.append([rx, ry, rz])
    return corners


def project_point(point: List[float], camera_matrix: List[List[float]]) -> List[float]:
    x, y, z = point
    px = camera_matrix[0][0]*x + camera_matrix[0][1]*y + camera_matrix[0][2]*z + camera_matrix[0][3]
    py = camera_matrix[1][0]*x + camera_matrix[1][1]*y + camera_matrix[1][2]*z + camera_matrix[1][3]
    pz = camera_matrix[2][0]*x + camera_matrix[2][1]*y + camera_matrix[2][2]*z + camera_matrix[2][3]
    if abs(pz) < 1e-6:
        pz = 1e-6
    return [px/pz, py/pz, pz]


def compute_bbox(corners_2d: List[List[float]], width: int, height: int) -> Optional[List[float]]:
    xs, ys = [], []
    for x, y, depth in corners_2d:
        if depth <= 0:
            continue
        xs.append(min(max(x, 0.0), width - 1.0))
        ys.append(min(max(y, 0.0), height - 1.0))
    if not xs or not ys:
        return None
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_max <= x_min or y_max <= y_min:
        return None
    w = x_max - x_min
    h = y_max - y_min
    if w < 4 or h < 4:
        return None
    return [float(x_min), float(y_min), float(w), float(h)]


def find_image(frame_data: Dict, images_dir: Path, camera_name: str):
    image_files = frame_data.get("frame_properties", {}).get("image_file_names", [])
    for name in image_files:
        if camera_name in name:
            path = images_dir / camera_name / name
            if path.exists():
                return path, name
    return None


def get_camera_intrinsics(openlabel: Dict, camera_name: str):
    streams = openlabel["openlabel"].get("streams", {})
    stream = streams.get(camera_name)
    if not stream:
        return None
    intr = stream.get("stream_properties", {}).get("intrinsics_pinhole", {})
    matrix = intr.get("camera_matrix_3x4")
    width = intr.get("width_px")
    height = intr.get("height_px")
    if not matrix or width is None or height is None:
        return None
    return matrix, int(width), int(height)


def get_transform(openlabel: Dict, lidar_frame: str, camera_name: str) -> Optional[List[List[float]]]:
    coord = openlabel["openlabel"].get("coordinate_systems", {})
    entry = coord.get(camera_name)
    if not entry:
        return None
    if entry.get("parent") != lidar_frame:
        return None
    pose = entry.get("pose_wrt_parent", {}).get("matrix4x4")
    if not pose:
        return None
    # pose_wrt_parent already describes lidar -> camera transform
    return matrix4x4_to_list(pose)


def process_file(json_path: Path, images_dir: Path, camera_name: str) -> List[Dict]:
    data = json.loads(json_path.read_text())
    frames = data["openlabel"].get("frames", {})
    if not frames:
        return []
    frame_data = next(iter(frames.values()))
    image_info = find_image(frame_data, images_dir, camera_name)
    if not image_info:
        return []
    image_path, image_file = image_info
    intr = get_camera_intrinsics(data, camera_name)
    if not intr:
        return []
    camera_matrix, width, height = intr
    lidar_frame = None
    if "s110_lidar_ouster_north" in data["openlabel"]["coordinate_systems"]:
        lidar_frame = "s110_lidar_ouster_north"
    elif "s110_lidar_ouster_south" in data["openlabel"]["coordinate_systems"]:
        lidar_frame = "s110_lidar_ouster_south"
    if not lidar_frame:
        return []
    T = get_transform(data, lidar_frame, camera_name)
    if not T:
        return []
    annotations = []
    objects = frame_data.get("objects", {})
    for obj in objects.values():
        obj_data = obj.get("object_data", {})
        obj_type = obj_data.get("type")
        if obj_type not in CATEGORY_MAPPING:
            continue
        cuboid = obj_data.get("cuboid", {}).get("val")
        if not cuboid:
            continue
        try:
            corners = cuboid_to_corners(cuboid)
        except Exception:
            continue
        corners_cam = [apply_transform(pt, T) for pt in corners]
        proj = [project_point(pt, camera_matrix) for pt in corners_cam]
        bbox = compute_bbox(proj, width, height)
        if not bbox:
            continue
        annotations.append({
            "bbox": bbox,
            "category_id": CATEGORY_MAPPING[obj_type],
            "area": bbox[2] * bbox[3],
            "iscrowd": 0,
            "object_name": obj_data.get("name", ""),
        })
    if not annotations:
        return []
    return [{
        "file_name": image_file,
        "width": width,
        "height": height,
        "annotations": annotations,
    }]


def convert_split(split_dir: Path, output_dir: Path, camera_name: str):
    labels_dir = split_dir / "labels_point_clouds"
    images_dir = split_dir / "images"
    lidar_dirs = sorted(labels_dir.glob("s110_lidar_*"))

    def bbox_iou(b1, b2):
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[0] + b1[2], b2[0] + b2[2])
        y2 = min(b1[1] + b1[3], b2[1] + b2[3])
        if x2 <= x1 or y2 <= y1:
            return 0.0
        inter = (x2 - x1) * (y2 - y1)
        area1 = b1[2] * b1[3]
        area2 = b2[2] * b2[3]
        return inter / (area1 + area2 - inter)
    coco = {
        "info": {
            "description": f"A9 Dataset - {split_dir.name} - {camera_name}",
            "version": "1.0",
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {"id": cid, "name": name, "supercategory": "vehicle" if cid <= 6 else "person"}
            for cid, name in CATEGORY_NAMES.items()
        ],
    }
    image_map = {}
    image_box_map = defaultdict(list)
    image_object_map = defaultdict(dict)
    next_image_id = 1
    next_ann_id = 1
    for lidar_dir in lidar_dirs:
        for json_file in sorted(lidar_dir.glob("*.json")):
            samples = process_file(json_file, images_dir, camera_name)
            for sample in samples:
                file_name = sample["file_name"]
                if file_name not in image_map:
                    image_map[file_name] = next_image_id
                    coco["images"].append({
                        "id": next_image_id,
                        "file_name": file_name,
                        "width": sample["width"],
                        "height": sample["height"],
                    })
                    next_image_id += 1
                image_id = image_map[file_name]
                for ann in sample["annotations"]:
                    bbox = ann["bbox"]
                    cat = ann["category_id"]
                    obj_name = ann.get("object_name", "")
                    add_entry = True
                    if obj_name:
                        key = (obj_name, cat)
                        if key in image_object_map[image_id]:
                            add_entry = False
                        else:
                            image_object_map[image_id][key] = bbox
                    if add_entry:
                        for existing in image_box_map[image_id]:
                            if existing["category_id"] == cat and bbox_iou(bbox, existing["bbox"]) > 0.7:
                                add_entry = False
                                break
                    if not add_entry:
                        continue
                    entry = ann.copy()
                    entry.pop("object_name", None)
                    entry["id"] = next_ann_id
                    entry["image_id"] = image_id
                    coco["annotations"].append(entry)
                    image_box_map[image_id].append({"bbox": bbox, "category_id": cat})
                    next_ann_id += 1
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{split_dir.name}_{camera_name}_coco.json"
    with open(output_path, "w") as f:
        json.dump(coco, f, indent=2)
    print(f"Saved {output_path}: images={len(coco['images'])}, annotations={len(coco['annotations'])}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a9_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--splits", nargs="*", default=None)
    parser.add_argument("--cameras", nargs="*", default=None)
    args = parser.parse_args()
    a9_root = Path(args.a9_root)
    output_dir = Path(args.output_dir)
    splits = args.splits or [f"a9_dataset_r02_s{i:02d}" for i in range(1, 5)]
    cameras = args.cameras or ["s110_camera_basler_south1_8mm", "s110_camera_basler_south2_8mm"]
    for split in splits:
        split_dir = a9_root / split
        if not split_dir.exists():
            print(f"Skip missing {split_dir}")
            continue
        for camera in cameras:
            print(f"Processing {split}/{camera}")
            convert_split(split_dir, output_dir, camera)


if __name__ == "__main__":
    main()
