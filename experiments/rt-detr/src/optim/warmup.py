"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

from ..core import register


@register()
class WarmupLR:
    """简化的学习率预热调度器"""
    def __init__(self, optimizer, warmup_epochs=5, warmup_start_lr=1e-7, warmup_end_lr=1e-4):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        self.warmup_end_lr = warmup_end_lr
        self.current_epoch = 0
        
    def step(self):
        """更新学习率"""
        if self.current_epoch < self.warmup_epochs:
            # 线性预热
            lr = self.warmup_start_lr + (self.warmup_end_lr - self.warmup_start_lr) * (self.current_epoch + 1) / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        self.current_epoch += 1
        
    def finished(self):
        """检查预热是否完成"""
        return self.current_epoch >= self.warmup_epochs
        
    def get_last_lr(self):
        """获取当前学习率"""
        return [param_group['lr'] for param_group in self.optimizer.param_groups]
    
    def state_dict(self):
        """返回调度器状态字典"""
        return {
            'warmup_epochs': self.warmup_epochs,
            'warmup_start_lr': self.warmup_start_lr,
            'warmup_end_lr': self.warmup_end_lr,
            'current_epoch': self.current_epoch
        }
    
    def load_state_dict(self, state_dict):
        """加载调度器状态"""
        self.warmup_epochs = state_dict.get('warmup_epochs', self.warmup_epochs)
        self.warmup_start_lr = state_dict.get('warmup_start_lr', self.warmup_start_lr)
        self.warmup_end_lr = state_dict.get('warmup_end_lr', self.warmup_end_lr)
        self.current_epoch = state_dict.get('current_epoch', self.current_epoch)

