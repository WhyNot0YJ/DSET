"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

# for register purpose
from . import optim
from . import nn
from . import zoo

# data 模块延迟导入：放在最后，确保其他模块已初始化
# 使用延迟导入避免循环导入问题
import sys
if 'src.data' not in sys.modules:
    try:
        from . import data
    except (ImportError, AttributeError) as e:
        # 如果导入失败（循环导入），不阻止其他模块的导入
        # 后续通过 from src.data.dataset.dairv2x_detection import ... 时会自动导入
        pass
