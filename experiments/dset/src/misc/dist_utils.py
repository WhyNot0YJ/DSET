"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import random
import numpy as np
import torch
import os
import sys

def setup_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_print():
    """设置打印选项"""
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if force:
            builtin_print(*args, **kwargs)
        else:
            # 可以在这里添加条件来控制打印
            builtin_print(*args, **kwargs)

    __builtin__.print = print


# 简化的分布式函数 - 单机版本
def is_dist_available_and_initialized():
    """检查分布式是否可用且已初始化 - 单机版本总是返回False"""
    return False


def get_world_size():
    """获取分布式训练的世界大小 - 单机版本总是返回1"""
    return 1


def get_rank():
    """获取当前进程的rank - 单机版本总是返回0"""
    return 0


def is_main_process():
    """检查是否为主进程 - 单机版本总是返回True"""
    return True


def save_on_master(*args, **kwargs):
    """只在主进程上保存 - 单机版本直接保存"""
    torch.save(*args, **kwargs)


def de_parallel(model):
    """从分布式并行模型中提取原始模型 - 单机版本直接返回"""
    return model
