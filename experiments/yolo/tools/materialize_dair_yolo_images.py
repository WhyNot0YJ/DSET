#!/usr/bin/env python3
"""
将 DAIR-V2X_YOLO/images/{train,val,test} 下的符号链接替换为真实图片文件副本。

用法:
  python3 tools/materialize_dair_yolo_images.py [--root /path/to/DAIR-V2X_YOLO] [--dry-run]

真实文件通常来自链接目标（常见为 DAIR-V2X-COCO 的 image/ 目录或其它解压路径）。
"""

from __future__ import annotations

import argparse
import shutil
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Dereference symlinks in DAIR-V2X_YOLO image folders.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/root/autodl-fs/datasets/DAIR-V2X_YOLO"),
        help="DAIR-V2X_YOLO 数据集根目录（内含 images/train 等）",
    )
    parser.add_argument("--dry-run", action="store_true", help="只统计与打印，不写盘")
    args = parser.parse_args()

    images = args.root / "images"
    if not images.is_dir():
        raise SystemExit(f"不存在目录: {images}")

    target_roots: Counter[str] = Counter()
    stats = {"symlink_ok": 0, "symlink_broken": 0, "already_file": 0, "other": 0}

    for split in ("train", "val", "test"):
        d = images / split
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            if not p.name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                continue
            if p.is_symlink():
                try:
                    resolved = p.resolve()
                except OSError as e:
                    print(f"[broken-resolve] {p}: {e}")
                    stats["symlink_broken"] += 1
                    continue
                if not resolved.is_file():
                    print(f"[broken] {p} -> {p.readlink()} (resolved missing: {resolved})")
                    stats["symlink_broken"] += 1
                    continue
                target_roots[str(resolved.parent)] += 1
                if args.dry_run:
                    print(f"[would copy] {p} <- {resolved}")
                    stats["symlink_ok"] += 1
                    continue
                tmp = p.with_name(p.name + ".materialize.tmp")
                try:
                    shutil.copy2(resolved, tmp)
                    p.unlink(missing_ok=False)
                    tmp.rename(p)
                except OSError as e:
                    tmp.unlink(missing_ok=True)
                    print(f"[fail] {p}: {e}")
                    stats["symlink_broken"] += 1
                    continue
                stats["symlink_ok"] += 1
            elif p.is_file():
                stats["already_file"] += 1
            else:
                stats["other"] += 1

    print("--- 真实文件所在目录（按链接解析结果统计，前 20）---")
    for parent, cnt in target_roots.most_common(20):
        print(f"  {cnt:6d}  <- {parent}")
    print("--- 统计 ---")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    if args.dry_run and stats["symlink_ok"]:
        print("\n去掉 --dry-run 后执行实际拷贝。")


if __name__ == "__main__":
    main()
