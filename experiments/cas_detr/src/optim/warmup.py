"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

from torch.optim.lr_scheduler import LRScheduler

from ..core import register


class Warmup(object):
    def __init__(self, lr_scheduler: LRScheduler, warmup_duration: int, last_step: int=-1) -> None:
        self.lr_scheduler = lr_scheduler
        self.warmup_end_values = [pg['lr'] for pg in lr_scheduler.optimizer.param_groups]
        self.last_step = last_step
        self.warmup_duration = warmup_duration
        self.step()

    def state_dict(self):
        return {k: v for k, v in self.__dict__.items() if k != 'lr_scheduler'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def get_warmup_factor(self, step, **kwargs):
        raise NotImplementedError

    def step(self, ):
        self.last_step += 1
        if self.last_step >= self.warmup_duration:
            return
        factor = self.get_warmup_factor(self.last_step)
        for i, pg in enumerate(self.lr_scheduler.optimizer.param_groups):
            pg['lr'] = factor * self.warmup_end_values[i]
    
    def finished(self, ):
        if self.last_step >= self.warmup_duration:
            return True 
        return False


@register()
class LinearWarmup(Warmup):
    def __init__(self, lr_scheduler: LRScheduler, warmup_duration: int, last_step: int = -1) -> None:
        super().__init__(lr_scheduler, warmup_duration, last_step)

    def get_warmup_factor(self, step):
        return min(1.0, (step + 1) / self.warmup_duration)


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

