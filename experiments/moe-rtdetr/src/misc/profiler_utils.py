"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import time
import torch

class stats:
    """简单的性能统计工具"""
    
    def __init__(self):
        self.times = {}
        self.counts = {}
    
    def start(self, name):
        """开始计时"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.times[name] = time.time()
    
    def end(self, name):
        """结束计时"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if name in self.times:
            elapsed = time.time() - self.times[name]
            if name not in self.counts:
                self.counts[name] = []
            self.counts[name].append(elapsed)
            return elapsed
        return 0
    
    def get_stats(self, name):
        """获取统计信息"""
        if name in self.counts:
            times = self.counts[name]
            return {
                'count': len(times),
                'total': sum(times),
                'avg': sum(times) / len(times),
                'min': min(times),
                'max': max(times)
            }
        return None
