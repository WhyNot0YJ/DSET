"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

# for register purpose
from . import optim
from . import nn
from . import zoo

# 延迟导入 data 模块，避免循环导入
# 使用 __getattr__ 实现延迟导入，只有在真正访问时才导入
# 这样可以避免在模块初始化时就导入 data，从而避免循环导入
def __getattr__(name):
    if name == 'data':
        from . import data
        return data
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")