#!/usr/bin/env python3
"""Merge multiple COCO json files into one."""
import argparse
import json
from pathlib import Path


def merge_coco(files, output):
    merged = {
        "info": {"description": "Merged A9 COCO", "version": "1.0"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [],
    }
    image_map = {}  # file_name -> new id
    next_image_id = 1
    next_ann_id = 1
    for idx, file in enumerate(files):
        data = json.loads(Path(file).read_text())
        if not merged["categories"]:
            merged["categories"] = data.get("categories", [])
        id_to_file = {img["id"]: img for img in data.get("images", [])}
        for ann in data.get("annotations", []):
            img_info = id_to_file.get(ann["image_id"])
            if not img_info:
                continue
            fname = img_info["file_name"]
            if fname not in image_map:
                image_map[fname] = next_image_id
                merged["images"].append({
                    "id": next_image_id,
                    "file_name": fname,
                    "width": img_info["width"],
                    "height": img_info["height"],
                })
                next_image_id += 1
            merged["annotations"].append({
                "id": next_ann_id,
                "image_id": image_map[fname],
                "category_id": ann["category_id"],
                "bbox": ann["bbox"],
                "area": ann["area"],
                "iscrowd": ann.get("iscrowd", 0),
            })
            next_ann_id += 1
    Path(output).write_text(json.dumps(merged, indent=2))
    print(f"Saved {output}: images={len(merged['images'])}, annotations={len(merged['annotations'])}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    files = sorted(input_dir.glob("*_coco.json"))
    if not files:
        raise SystemExit(f"No COCO files in {input_dir}")
    merge_coco(files, args.output)


if __name__ == "__main__":
    main()
