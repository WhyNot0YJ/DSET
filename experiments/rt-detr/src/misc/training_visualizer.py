"""训练过程可视化工具

提供训练曲线绘制、指标记录和状态保存功能，支持多种模型类型。

主要功能：
- 自动记录训练/验证指标
- 实时绘制训练曲线
- 支持checkpoint保存和恢复
- 专为MOE模型设计的专家使用率可视化
- 灵活的接口，易于集成到不同训练脚本

用法示例：
    from src.misc.training_visualizer import TrainingVisualizer
    
    # 初始化
    visualizer = TrainingVisualizer(log_dir='logs/exp1', model_type='moe')
    
    # 记录指标
    visualizer.record(
        epoch=0,
        train_loss=1234.5,
        val_loss=567.8,
        mAP_0_5=0.123,
        mAP_0_75=0.089,
        mAP_0_5_0_95=0.056,
        learning_rate=1e-5,
        expert_usage=[0.15, 0.18, 0.16, 0.17, 0.19, 0.15],
        router_loss=0.05
    )
    
    # 绘制曲线
    visualizer.plot()
    
    # 保存/加载状态
    state = visualizer.state_dict()
    visualizer.load_state_dict(state)
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt


class TrainingVisualizer:
    """训练可视化工具类。
    
    自动记录和绘制训练过程中的各种指标，支持保存和恢复状态。
    
    Attributes:
        log_dir: 日志保存目录
        model_type: 模型类型 ('moe', 'standard' 等)
        history: 历史指标字典
    """
    
    def __init__(
        self, 
        log_dir: Union[str, Path], 
        model_type: str = 'standard',
        experiment_name: Optional[str] = None
    ):
        """初始化可视化工具。
        
        Args:
            log_dir: 日志保存目录路径
            model_type: 模型类型，'moe' 会启用专家使用率可视化，'standard' 为普通模型
            experiment_name: 实验名称，用于在曲线标题中显示（如 'rt_detr_r34'）
        """
        self.log_dir = Path(log_dir)
        self.model_type = model_type
        self.experiment_name = experiment_name
        
        # 初始化历史记录
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'mAP_0_5': [],
            'mAP_0_75': [],
            'mAP_0_5_0_95': [],
            'learning_rate': [],
            'expert_usage': [],  # 仅MOE模型使用
            'router_loss': []     # 仅MOE模型使用
        }
    
    def record(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        mAP_0_5: float = 0.0,
        mAP_0_75: float = 0.0,
        mAP_0_5_0_95: float = 0.0,
        learning_rate: float = 0.0,
        expert_usage: Optional[List[float]] = None,
        router_loss: Optional[float] = None
    ) -> None:
        """记录一个epoch的训练指标。
        
        Args:
            epoch: 当前epoch编号
            train_loss: 训练损失
            val_loss: 验证损失
            mAP_0_5: mAP@0.5指标
            mAP_0_75: mAP@0.75指标
            mAP_0_5_0_95: mAP@[0.5:0.95]指标
            learning_rate: 当前学习率
            expert_usage: 专家使用率列表（仅MOE模型）
            router_loss: 路由器损失（仅MOE模型）
        """
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['mAP_0_5'].append(mAP_0_5)
        self.history['mAP_0_75'].append(mAP_0_75)
        self.history['mAP_0_5_0_95'].append(mAP_0_5_0_95)
        self.history['learning_rate'].append(learning_rate)
        
        if expert_usage is not None:
            self.history['expert_usage'].append(expert_usage)
        if router_loss is not None:
            self.history['router_loss'].append(router_loss)
    
    def plot(self) -> None:
        """绘制并保存训练曲线。
        
        根据model_type自动选择合适的绘图布局：
        - MOE模型: 4个子图（损失、mAP、学习率、专家使用率）
        - 标准模型: 3个子图（损失、mAP、学习率）
        
        生成的图像保存在log_dir下。
        """
        if len(self.history['train_loss']) == 0:
            return
        
        if self.model_type == 'moe' and len(self.history['expert_usage']) > 0:
            self._plot_with_experts()
        else:
            self._plot_standard()
    
    def _plot_standard(self) -> None:
        """绘制标准训练曲线。"""
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # 创建1x3的子图
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        title = f'{self.experiment_name.upper()} Training Curves' if self.experiment_name else 'RTDETR Training Curves'
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # 1. 损失曲线
        ax = axes[0]
        ax.plot(epochs, self.history['train_loss'], 'b-o', 
                label='Train Loss', linewidth=2, markersize=4)
        ax.plot(epochs, self.history['val_loss'], 'r-s', 
                label='Val Loss', linewidth=2, markersize=4)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Loss Curves', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 2. mAP曲线
        ax = axes[1]
        if max(self.history['mAP_0_5']) > 0:
            ax.plot(epochs, self.history['mAP_0_5'], 'g-^', 
                    label='mAP@0.5', linewidth=2, markersize=4)
            ax.plot(epochs, self.history['mAP_0_75'], 'c-v', 
                    label='mAP@0.75', linewidth=2, markersize=4)
            ax.plot(epochs, self.history['mAP_0_5_0_95'], 'm-d', 
                    label='mAP@[0.5:0.95]', linewidth=2, markersize=4)
            ax.legend(fontsize=10)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('mAP', fontsize=12)
        ax.set_title('mAP Metrics', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 3. 学习率曲线
        ax = axes[2]
        if max(self.history['learning_rate']) > 0:
            ax.plot(epochs, self.history['learning_rate'], 
                    'orange', linewidth=2)
            ax.set_yscale('log')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Learning Rate', fontsize=12)
        ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.log_dir / 'training_curves.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_with_experts(self) -> None:
        """绘制包含MOE专家信息的训练曲线。"""
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # 创建2x2的子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        title = f'{self.experiment_name.upper()} Training Curves' if self.experiment_name else 'MOE RT-DETR Training Curves'
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # 1. 损失曲线
        ax = axes[0, 0]
        ax.plot(epochs, self.history['train_loss'], 'b-o', 
                label='Train Loss', linewidth=2, markersize=4)
        ax.plot(epochs, self.history['val_loss'], 'r-s', 
                label='Val Loss', linewidth=2, markersize=4)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Loss Curves', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 2. mAP曲线
        ax = axes[0, 1]
        if max(self.history['mAP_0_5']) > 0:
            ax.plot(epochs, self.history['mAP_0_5'], 'g-^', 
                    label='mAP@0.5', linewidth=2, markersize=4)
            ax.plot(epochs, self.history['mAP_0_75'], 'c-v', 
                    label='mAP@0.75', linewidth=2, markersize=4)
            ax.plot(epochs, self.history['mAP_0_5_0_95'], 'm-d', 
                    label='mAP@[0.5:0.95]', linewidth=2, markersize=4)
            ax.legend(fontsize=10)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('mAP', fontsize=12)
        ax.set_title('mAP Metrics', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 3. 学习率曲线
        ax = axes[1, 0]
        if max(self.history['learning_rate']) > 0:
            ax.plot(epochs, self.history['learning_rate'], 
                    'orange', linewidth=2)
            ax.set_yscale('log')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Learning Rate', fontsize=12)
        ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 4. 专家使用率（最新epoch）
        ax = axes[1, 1]
        if self.history['expert_usage']:
            latest_usage = self.history['expert_usage'][-1]
            expert_ids = [f'Expert{i}' for i in range(len(latest_usage))]
            colors = plt.cm.viridis(np.linspace(0, 1, len(latest_usage)))
            bars = ax.bar(expert_ids, latest_usage, color=colors, 
                         alpha=0.7, edgecolor='black')
            
            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2%}',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            ax.set_xlabel('Expert ID', fontsize=12)
            ax.set_ylabel('Usage Rate', fontsize=12)
            ax.set_title(f'Expert Usage Distribution (Epoch {len(epochs)})', 
                        fontsize=14, fontweight='bold')
            ax.set_ylim(0, max(latest_usage) * 1.2)
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        save_path = self.log_dir / 'training_curves.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # 绘制专家使用率历史（单独的图）
        self._plot_expert_usage_history()
    
    def _plot_expert_usage_history(self) -> None:
        """绘制专家使用率历史变化。"""
        if not self.history['expert_usage']:
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        epochs = range(1, len(self.history['expert_usage']) + 1)
        expert_usage_array = np.array(self.history['expert_usage'])
        num_experts = expert_usage_array.shape[1]
        
        # 为每个专家绘制一条线
        colors = plt.cm.tab10(np.linspace(0, 1, num_experts))
        for i in range(num_experts):
            ax.plot(epochs, expert_usage_array[:, i], 
                   marker='o', label=f'Expert{i}', 
                   color=colors[i], linewidth=2, markersize=4)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Usage Rate', fontsize=12)
        ax.set_title('Expert Usage History', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, ncol=2)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.log_dir / 'expert_usage_history.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def state_dict(self) -> Dict:
        """返回可序列化的状态字典。
        
        Returns:
            包含所有历史记录的字典，可用于checkpoint保存
        """
        return {
            'model_type': self.model_type,
            'history': self.history
        }
    
    def load_state_dict(self, state_dict: Dict) -> None:
        """从状态字典恢复历史记录。
        
        Args:
            state_dict: 之前保存的状态字典
        """
        if 'model_type' in state_dict:
            self.model_type = state_dict['model_type']
        if 'history' in state_dict:
            self.history = state_dict['history']
    
    def export_to_csv(self, filename: str = 'training_history.csv') -> None:
        """导出训练历史到CSV文件。
        
        Args:
            filename: CSV文件名
        """
        import csv
        
        csv_path = self.log_dir / filename
        
        # 准备数据
        rows = []
        num_epochs = len(self.history['train_loss'])
        
        for i in range(num_epochs):
            row = {
                'epoch': i + 1,
                'train_loss': self.history['train_loss'][i],
                'val_loss': self.history['val_loss'][i],
                'mAP_0.5': self.history['mAP_0_5'][i],
                'mAP_0.75': self.history['mAP_0_75'][i],
                'mAP_0.5_0.95': self.history['mAP_0_5_0_95'][i],
                'learning_rate': self.history['learning_rate'][i]
            }
            
            if i < len(self.history['router_loss']):
                row['router_loss'] = self.history['router_loss'][i]
            
            if i < len(self.history['expert_usage']):
                for j, usage in enumerate(self.history['expert_usage'][i]):
                    row[f'expert_{j}_usage'] = usage
            
            rows.append(row)
        
        # 写入CSV
        if rows:
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
    
    def get_best_epoch(self, metric: str = 'val_loss', mode: str = 'min') -> int:
        """获取最佳epoch编号。
        
        Args:
            metric: 指标名称 ('val_loss', 'mAP_0_5_0_95' 等)
            mode: 'min' 或 'max'
        
        Returns:
            最佳epoch编号（从1开始）
        """
        if metric not in self.history:
            raise ValueError(f"指标 '{metric}' 不存在")
        
        values = self.history[metric]
        if not values:
            return 0
        
        if mode == 'min':
            best_idx = values.index(min(values))
        else:
            best_idx = values.index(max(values))
        
        return best_idx + 1


__all__ = ['TrainingVisualizer']

