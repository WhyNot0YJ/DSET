"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

# for register purpose
# 使用延迟导入避免循环导入问题
from . import optim
try:
    from . import data 
except (ImportError, AttributeError):
    # 如果出现循环导入，延迟到后续导入时再注册
    pass
from . import nn
from . import zoo