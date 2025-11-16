"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

# for register purpose
from . import optim
from . import nn
from . import zoo
# 注意：data 模块不在这里导入，避免循环导入
# train.py 中直接使用 from src.data.dataset.dairv2x_detection import ... 即可
# @register() 装饰器会在模块导入时自动执行，不需要在 __init__.py 中预先导入
