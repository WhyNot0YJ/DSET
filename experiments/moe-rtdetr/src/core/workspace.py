""""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import functools
from collections import defaultdict
from typing import Any


GLOBAL_CONFIG = defaultdict(dict)


def register(dct: Any = GLOBAL_CONFIG, name=None, force=False):
    """
    简化的注册装饰器 - 只支持类注册
    """
    def decorator(foo):
        register_name = foo.__name__ if name is None else name
        
        if not force and register_name in dct:
            raise ValueError(f'{register_name} has been already registered')

        # 简化版本：直接存储类
        dct[register_name] = foo
        return foo

    return decorator



def create(type_or_name, global_cfg=GLOBAL_CONFIG, **kwargs):
    """
    简化的创建函数 - 直接实例化注册的类
    """
    name = type_or_name if isinstance(type_or_name, str) else type_or_name.__name__

    if name not in global_cfg:
        raise ValueError(f'The module {name} is not registered')

    cls = global_cfg[name]
    return cls(**kwargs)
