"""Early stopping：只负责「连续多少次未改善就停」，不维护最佳指标。

最佳值由训练器在验证后更新；此处仅根据训练器传入的 improved 计数，
避免与 best_map / best_loss 两套状态漂移。
"""

import logging
from typing import Optional


class EarlyStopping:
    """连续 patience 次验证未改善则返回 True（应停止训练）。"""

    __slots__ = ("patience", "metric_name", "logger", "counter")

    def __init__(
        self,
        patience: int = 15,
        mode: str = "max",
        min_delta: float = 0.0001,
        metric_name: str = "mAP_0.5_0.95",
        logger: Optional[logging.Logger] = None,
    ):
        # mode / min_delta 保留仅为与旧 checkpoint、旧调用兼容，不再使用
        self.patience = patience
        self.metric_name = metric_name
        self.logger = logger
        self.counter = 0

    def step(
        self,
        improved: bool,
        *,
        best_value: float,
        best_epoch: int,
    ) -> bool:
        """一次验证结束后调用。improved 由训练器用与保存 best 相同的规则算出。"""
        if improved:
            self.counter = 0
            return False

        self.counter += 1
        if self.logger:
            self.logger.info(
                f"  ⏳ {self.metric_name} 无改善 ({self.counter}/{self.patience}), "
                f"最佳值: {best_value:.4f} (epoch {best_epoch})"
            )

        if self.counter >= self.patience:
            if self.logger:
                self.logger.info(
                    f"\n{'=' * 60}\n"
                    f"⛔ Early Stopping触发！\n"
                    f"   已连续 {self.patience} 个epoch无改善\n"
                    f"   最佳 {self.metric_name}: {best_value:.4f} (epoch {best_epoch})\n"
                    f"{'=' * 60}\n"
                )
            return True
        return False

    def state_dict(self) -> dict:
        return {"counter": self.counter, "patience": self.patience, "metric_name": self.metric_name}

    def load_state_dict(self, state_dict: dict) -> None:
        self.counter = state_dict.get("counter", 0)
        self.patience = state_dict.get("patience", self.patience)
        self.metric_name = state_dict.get("metric_name", self.metric_name)


__all__ = ["EarlyStopping"]
