#!/usr/bin/env python3
"""扫描数据集中无法打开或损坏的图片（0 字节、截断 JPEG 等），便于重新复制。

用法:
  python scripts/verify_dataset_images.py "C:/Users/yujie/Downloads/datasets/UA-DETRAC_YOLO/images"
  python scripts/verify_dataset_images.py /path/to/UA-DETRAC_COCO --splits train val test
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def check_one(path: Path) -> str | None:
    """返回错误说明；正常则返回 None。"""
    try:
        if not path.is_file():
            return "not a file"
        if path.stat().st_size == 0:
            return "empty file (0 bytes)"
    except OSError as e:
        return f"os stat: {e}"

    try:
        from PIL import Image

        with Image.open(path) as im:
            im.verify()
    except Exception as e:
        return f"PIL: {e}"

    # verify() 后需再 open 一次做完整解码（部分截断图 verify 仍过）
    try:
        from PIL import Image

        with Image.open(path) as im:
            im.load()
    except Exception as e:
        return f"load: {e}"

    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify JPEG/PNG in dataset folders")
    parser.add_argument(
        "root",
        type=Path,
        help="根目录，例如 .../UA-DETRAC_YOLO/images 或 .../UA-DETRAC_COCO",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=[],
        help="若设置，则只扫描 root/<split> 子目录（如 train val test）；不设置则递归扫描整个 root",
    )
    parser.add_argument(
        "--ext",
        nargs="*",
        default=[".jpg", ".jpeg", ".png", ".bmp", ".webp"],
        help="扩展名（小写）",
    )
    args = parser.parse_args()

    root: Path = args.root
    exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in args.ext}

    files: list[Path] = []
    if args.splits:
        for sp in args.splits:
            d = root / sp
            if not d.is_dir():
                print(f"[WARN] 跳过不存在的目录: {d}", file=sys.stderr)
                continue
            for p in d.rglob("*"):
                if p.suffix.lower() in exts:
                    files.append(p)
    else:
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                files.append(p)

    files.sort(key=lambda p: str(p))
    bad: list[tuple[Path, str]] = []
    for i, p in enumerate(files):
        err = check_one(p)
        if err:
            bad.append((p, err))
        if (i + 1) % 500 == 0:
            print(f"  已检查 {i + 1}/{len(files)} ...", file=sys.stderr)

    print(f"共扫描 {len(files)} 个文件，异常 {len(bad)} 个。\n")
    for p, msg in bad:
        print(f"{p}\n  -> {msg}\n")

    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
