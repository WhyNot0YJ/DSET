"""可配置的验证 / 评估频率（按 epoch 0-based，与 train 循环一致）。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# 与原 RT-DETR / CaS-DETR 硬编码策略一致：
# - epoch < 50：每 10 轮验证一次（(epoch+1) % 10 == 0）
# - 50–70：每 5 轮
# - 70–90：每 2 轮
# - 90 以后：每轮验证
DEFAULT_RTDETR_EVAL_SCHEDULE: List[Dict[str, Any]] = [
    {"until_epoch": 50, "every": 10},
    {"until_epoch": 70, "every": 5},
    {"until_epoch": 90, "every": 2},
    {"until_epoch": None, "always": True},
]


def _phase_since_epoch(phase: Dict[str, Any]) -> Optional[int]:
    """从第几个 epoch（0-based）起本阶段才参与匹配；可与 until 同用。"""
    if "since_epoch" in phase and phase["since_epoch"] is not None:
        return int(phase["since_epoch"])
    if "from_epoch" in phase and phase["from_epoch"] is not None:
        return int(phase["from_epoch"])
    return None


def should_run_validation(
    epoch: int,
    schedule: Optional[List[Dict[str, Any]]] = None,
) -> bool:
    """是否在完成 ``epoch``（0-based）这一轮训练后运行验证。

    **YAML 阶段列表** ``training.eval_schedule``：自上而下扫描，**第一个适用**的阶段决定本 epoch 是否验证。

    阶段字段：

    - ``until_epoch`` (int | null): 仅当 ``epoch < until_epoch`` 时本阶段才可能生效；
      ``null`` 表示无上限（直到训练结束）。
    - ``since_epoch`` / ``from_epoch`` (int | null): 若设置，仅当 ``epoch >= since_epoch`` 时本阶段才可能生效。
      与 ``every`` 连用时：验证当 ``(epoch - since_epoch) % every == 0``（用于「从第 N 轮起每 K 轮评一次」，
      例如 ``since_epoch: 20, every: 2`` → epoch 20,22,24,…）。
    - ``every`` (int): 若未设置 ``since_epoch`` / ``from_epoch``，则 ``(epoch + 1) % every == 0`` 时验证（与历史脚本一致）。
    - ``always`` (bool): 若为 True，本阶段每个 epoch 都验证（可代替 ``every: 1``）。
    - ``enabled`` (bool): 若为 False，本阶段内**永不**验证；缺省为 True。

    ``schedule`` 为 ``None`` 或空列表时，使用 ``DEFAULT_RTDETR_EVAL_SCHEDULE``。
    """
    phases = schedule if schedule else DEFAULT_RTDETR_EVAL_SCHEDULE
    if not phases:
        phases = DEFAULT_RTDETR_EVAL_SCHEDULE

    for phase in phases:
        until = phase.get("until_epoch")
        if until is not None and epoch >= int(until):
            continue

        since = _phase_since_epoch(phase)
        if since is not None and epoch < since:
            continue

        if phase.get("enabled") is False:
            return False

        if phase.get("always"):
            return True

        every = int(phase.get("every", 1))
        if every <= 0:
            return False
        if since is not None:
            return (epoch - since) % every == 0
        return (epoch + 1) % every == 0

    # 未匹配任何阶段（配置错误）；保守起见与旧逻辑末尾一致：每轮验证
    return True


def describe_eval_schedule(schedule: Optional[List[Dict[str, Any]]]) -> str:
    """便于日志打印的简短描述。"""
    if not schedule:
        return "default (50/70/90 + always)"
    parts = []
    for ph in schedule:
        until = ph.get("until_epoch")
        since = _phase_since_epoch(ph)
        ub = f"epoch<{until}" if until is not None else "rest"
        if since is not None:
            ub = f"epoch>={since}" + (f" & {ub}" if until is not None else "")
        if ph.get("enabled") is False:
            parts.append(f"{ub}: off")
        elif ph.get("always"):
            parts.append(f"{ub}: every epoch")
        elif since is not None:
            parts.append(f"{ub}: every {ph.get('every', 1)} from since_epoch")
        else:
            parts.append(f"{ub}: every {ph.get('every', 1)} epochs (1-based stride)")
    return "; ".join(parts)
