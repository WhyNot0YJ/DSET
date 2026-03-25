"""DETR / YOLO 共用：数据路径相对 ``/root/autodl-fs`` 解析（无其它根目录回退）。"""

from pathlib import Path

AUTODL_FS_ROOT = Path("/root/autodl-fs")


def resolve_autodl_fs_path(path_str: str) -> str:
    """
    解析存在的文件或目录路径（绝对路径字符串）。

    - 若已存在（绝对或相对当前工作目录），返回 ``resolve()`` 结果。
    - 若为绝对路径但不存在，抛出 ``FileNotFoundError``。
    - 若为相对路径，**仅**尝试 ``/root/autodl-fs / path_str``。
    """
    if not path_str or not str(path_str).strip():
        raise ValueError("路径为空")
    p = Path(path_str)
    if p.exists():
        return str(p.resolve())
    if p.is_absolute():
        raise FileNotFoundError(f"路径不存在: {p}")
    resolved = AUTODL_FS_ROOT / path_str
    if resolved.exists():
        return str(resolved.resolve())
    raise FileNotFoundError(f"路径不存在: {path_str}（已解析为 {resolved}）")


def resolve_detr_data_root(data_root: str) -> str:
    """数据集根目录；语义同 :func:`resolve_autodl_fs_path`。"""
    return resolve_autodl_fs_path(data_root)
