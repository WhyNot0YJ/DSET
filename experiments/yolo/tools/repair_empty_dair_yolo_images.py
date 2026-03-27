#!/usr/bin/env python3
"""
若 DAIR-V2X_YOLO/images/{train,val,test} 下存在 0 字节 .jpg，而从 DAIR-V2X/image/
存在同名非空文件，则用后者覆盖前者。

与 materialize_dair_yolo_images.py 区别：本脚本处理「已是普通文件但为空」的坏数据，
不依赖符号链接。

用法:
  python3 tools/repair_empty_dair_yolo_images.py \\
    --yolo-root /root/autodl-fs/datasets/DAIR-V2X_YOLO \\
    --image-dir /root/autodl-fs/datasets/DAIR-V2X/image
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--yolo-root",
        type=Path,
        default=Path("/root/autodl-fs/datasets/DAIR-V2X_YOLO"),
    )
    p.add_argument(
        "--image-dir",
        type=Path,
        default=Path("/root/autodl-fs/datasets/DAIR-V2X/image"),
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    src_dir = args.image_dir
    if not src_dir.is_dir():
        raise SystemExit(f"源图目录不存在: {src_dir}")

    fixed = skipped_no_src = skipped_src_empty = already_ok = 0
    for split in ("train", "val", "test"):
        d = args.yolo_root / "images" / split
        if not d.is_dir():
            continue
        for dest in sorted(d.glob("*.jpg")):
            if not dest.is_file():
                continue
            if dest.stat().st_size > 0:
                already_ok += 1
                continue
            base = dest.name
            src = src_dir / base
            if not src.is_file():
                skipped_no_src += 1
                print(f"[no-src] {dest} <- missing {src}")
                continue
            if src.stat().st_size == 0:
                skipped_src_empty += 1
                print(f"[src-empty] {src}")
                continue
            if args.dry_run:
                print(f"[would copy] {src} -> {dest}")
                fixed += 1
                continue
            shutil.copy2(src, dest)
            fixed += 1

    print("--- 统计 ---")
    print(f"  copied_empty_fixed: {fixed}")
    print(f"  already_non_empty: {already_ok}")
    print(f"  skipped_no_source: {skipped_no_src}")
    print(f"  skipped_source_empty: {skipped_src_empty}")


if __name__ == "__main__":
    main()
