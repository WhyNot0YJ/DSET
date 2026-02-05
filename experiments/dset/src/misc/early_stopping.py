"""Early Stopping工具

自动监控训练指标，在性能不再提升时提前停止训练。

用法示例：
    from src.misc.early_stopping import EarlyStopping
    
    early_stopping = EarlyStopping(
        patience=15,
        mode='max',
        metric_name='mAP_0.5_0.95'
    )
    
    for epoch in range(num_epochs):
        val_metric = validate()
        
        if early_stopping(val_metric, epoch):
            print(f"Early stopping triggered at epoch {epoch}")
            break
"""

import logging
from typing import Optional


class EarlyStopping:
    """Early Stopping类，用于提前停止训练。
    
    监控指定指标，当连续多个epoch没有改善时触发停止。
    
    Attributes:
        patience: 等待多少个epoch没有改善后停止
        mode: 'min' 或 'max'，指标优化方向
        min_delta: 最小改善阈值
        metric_name: 监控的指标名称
        best_value: 当前最佳值
        counter: 没有改善的epoch计数
        best_epoch: 最佳epoch编号
    """
    
    def __init__(
        self,
        patience: int = 15,
        mode: str = 'max',
        min_delta: float = 0.0001,
        metric_name: str = 'mAP_0.5_0.95',
        logger: Optional[logging.Logger] = None
    ):
        """初始化Early Stopping。
        
        Args:
            patience: 容忍多少个epoch没有改善
            mode: 'min' (loss) 或 'max' (mAP等)
            min_delta: 认为有改善的最小变化量
            metric_name: 监控指标的名称（用于日志）
            logger: 日志记录器
        """
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.metric_name = metric_name
        self.logger = logger
        
        self.counter = 0
        self.best_value = float('-inf') if mode == 'max' else float('inf')
        self.best_epoch = 0
        self.should_stop = False
    
    def __call__(self, current_value: float, epoch: int) -> bool:
        """检查是否应该停止训练。
        
        Args:
            current_value: 当前epoch的指标值
            epoch: 当前epoch编号
            
        Returns:
            bool: 是否应该停止训练
        """
        is_improvement = self._is_improvement(current_value)
        
        if is_improvement:
            # 有改善
            self.best_value = current_value
            self.best_epoch = epoch
            self.counter = 0
            if self.logger:
                self.logger.info(
                    f"  {self.metric_name} 提升到 {current_value:.4f} "
                    f"(最佳epoch: {epoch})"
                )
        else:
            # 没有改善
            self.counter += 1
            if self.logger and self.counter > 0:
                self.logger.info(
                    f"  ⏳ {self.metric_name} 无改善 ({self.counter}/{self.patience}), "
                    f"最佳值: {self.best_value:.4f} (epoch {self.best_epoch})"
                )
            
            # 检查是否达到patience
            if self.counter >= self.patience:
                self.should_stop = True
                if self.logger:
                    self.logger.info(
                        f"\n{'='*60}\n"
                        f"⛔ Early Stopping触发！\n"
                        f"   已连续 {self.patience} 个epoch无改善\n"
                        f"   最佳 {self.metric_name}: {self.best_value:.4f} (epoch {self.best_epoch})\n"
                        f"{'='*60}\n"
                    )
                return True
        
        return False
    
    def _is_improvement(self, current_value: float) -> bool:
        """判断当前值是否比最佳值有改善。
        
        Args:
            current_value: 当前指标值
            
        Returns:
            bool: 是否有改善
        """
        if self.mode == 'max':
            return current_value > self.best_value + self.min_delta
        else:
            return current_value < self.best_value - self.min_delta
    
    def state_dict(self) -> dict:
        """返回状态字典，用于保存checkpoint。
        
        Returns:
            dict: 包含所有状态的字典
        """
        return {
            'patience': self.patience,
            'mode': self.mode,
            'min_delta': self.min_delta,
            'metric_name': self.metric_name,
            'counter': self.counter,
            'best_value': self.best_value,
            'best_epoch': self.best_epoch,
            'should_stop': self.should_stop
        }
    
    def load_state_dict(self, state_dict: dict) -> None:
        """从状态字典恢复状态。
        
        Args:
            state_dict: 之前保存的状态字典
        """
        self.patience = state_dict.get('patience', self.patience)
        self.mode = state_dict.get('mode', self.mode)
        self.min_delta = state_dict.get('min_delta', self.min_delta)
        self.metric_name = state_dict.get('metric_name', self.metric_name)
        self.counter = state_dict.get('counter', 0)
        self.best_value = state_dict.get('best_value', self.best_value)
        self.best_epoch = state_dict.get('best_epoch', 0)
        self.should_stop = state_dict.get('should_stop', False)


__all__ = ['EarlyStopping']

