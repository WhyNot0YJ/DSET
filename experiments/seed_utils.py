"""随机种子设置工具

确保实验的可重复性，固定所有随机性来源：
- Python random
- NumPy random
- PyTorch random
- CUDA random
"""

import random
import numpy as np
import torch
import os


def set_seed(seed: int = 42, deterministic: bool = True):
    """设置所有随机种子以确保实验可重复性
    
    Args:
        seed: 随机种子值（默认42）
        deterministic: 是否使用确定性算法（会降低性能但保证完全可重复）
    
    示例：
        >>> set_seed(42)  # 设置种子为42
        >>> set_seed(3407, deterministic=True)  # 使用确定性模式
    """
    print(f"🌱 设置随机种子: {seed}")
    print(f"{'✅' if deterministic else '⚠️ '} 确定性模式: {deterministic}")
    
    # 1. Python内置random模块
    random.seed(seed)
    
    # 2. NumPy随机数生成器
    np.random.seed(seed)
    
    # 3. PyTorch CPU随机数生成器
    torch.manual_seed(seed)
    
    # 4. PyTorch GPU随机数生成器（如果使用CUDA）
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 多GPU情况
    
    # 5. 确定性算法设置
    if deterministic:
        # PyTorch 1.7+
        torch.use_deterministic_algorithms(True, warn_only=True)
        
        # cuDNN确定性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # 环境变量（用于一些操作）
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        os.environ['PYTHONHASHSEED'] = str(seed)
        
        print("   ✓ PyTorch确定性算法已启用")
        print("   ✓ cuDNN确定性模式已启用")
        print("   ⚠️  警告：确定性模式可能降低训练速度10-30%")
    else:
        # 非确定性但更快（允许cuDNN自动调优）
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        print("   ℹ️  使用非确定性模式（更快但结果可能略有差异）")
    
    print(f"   ✓ 所有随机源已设置为种子 {seed}\n")


def seed_worker(worker_id: int):
    """DataLoader worker初始化函数，确保每个worker有独立但可重复的随机种子
    
    Args:
        worker_id: worker的ID
    
    使用方法：
        >>> dataloader = DataLoader(
        ...     dataset, 
        ...     worker_init_fn=seed_worker,
        ...     generator=torch.Generator().manual_seed(42)
        ... )
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_recommended_seeds():
    """返回一些推荐的随机种子
    
    这些种子在多个深度学习任务中表现良好
    """
    return {
        'default': 42,       # 经典种子（《银河系漫游指南》）
        'pytorch': 3407,     # PyTorch社区推荐
        'lucky': 7,          # 幸运数字
        'prime': 97,         # 质数
        'year': 2025,        # 当前年份
        'test1': 12345,      # 简单易记
        'test2': 54321,      # 简单易记
        'test3': 99999,      # 简单易记
    }


def print_random_states():
    """打印当前随机状态（用于调试）"""
    print("📊 当前随机状态：")
    print(f"   Python random: {random.getstate()[1][0]}")
    print(f"   NumPy random: {np.random.get_state()[1][0]}")
    print(f"   PyTorch CPU: {torch.initial_seed()}")
    if torch.cuda.is_available():
        print(f"   PyTorch CUDA: {torch.cuda.initial_seed()}")
    print()


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("随机种子工具测试")
    print("=" * 60)
    
    # 测试1：基本种子设置
    print("\n【测试1】基本种子设置")
    set_seed(42, deterministic=False)
    
    # 测试2：确定性模式
    print("\n【测试2】确定性模式")
    set_seed(42, deterministic=True)
    
    # 测试3：打印状态
    print("\n【测试3】打印随机状态")
    print_random_states()
    
    # 测试4：推荐种子
    print("\n【测试4】推荐的随机种子")
    seeds = get_recommended_seeds()
    for name, seed in seeds.items():
        print(f"   {name:10s}: {seed}")
    
    # 测试5：验证可重复性
    print("\n【测试5】验证可重复性")
    set_seed(42)
    tensor1 = torch.randn(3, 3)
    
    set_seed(42)
    tensor2 = torch.randn(3, 3)
    
    print(f"   两次生成是否相同: {torch.allclose(tensor1, tensor2)}")
    print(f"   第一次: {tensor1[0, :3]}")
    print(f"   第二次: {tensor2[0, :3]}")
    
    print("\n" + "=" * 60)
    print("✅ 测试完成！")
    print("=" * 60)

