"""训练过程可视化（CaS_DETR 精简版）

仅维护主曲线所需标量（损失 / mAP / 学习率）与轻量 CSV；与 RT-DETR 共用 checkpoint 中的
`visualizer_state_dict`。细粒度指标（MoE、CASS、专家占比等）请在训练日志与最终 test 输出中查看。
"""

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt


class TrainingVisualizer:
    """训练曲线与 checkpoint 中的 `history`；CaS_DETR 仅保留 loss/mAP/lr 等主标量。"""
    
    def __init__(
        self, 
        log_dir: Union[str, Path], 
        model_type: str = 'standard',
        experiment_name: Optional[str] = None
    ):
        """初始化可视化工具。
        
        Args:
            log_dir: 日志保存目录路径
            model_type: 模型类型
                - 'standard': 标准模型
                - 'moe': MOE模型（单MoE）
                - 'cas_detr': CaS_DETR模型（Token Pruning + 双MoE）
            experiment_name: 实验名称，用于在曲线标题中显示（如 'cas_detr2_r18'）
        """
        self.log_dir = Path(log_dir)
        self.model_type = model_type
        self.experiment_name = experiment_name
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'mAP_0_5': [],
            'mAP_0_75': [],
            'mAP_0_5_0_95': [],
            'learning_rate': [],
            'expert_usage': [],
            'router_loss': [],
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
        router_loss: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """每个 epoch 只记录主曲线标量；其余关键字参数忽略（兼容旧调用）。"""
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['mAP_0_5'].append(mAP_0_5)
        self.history['mAP_0_75'].append(mAP_0_75)
        self.history['mAP_0_5_0_95'].append(mAP_0_5_0_95)
        self.history['learning_rate'].append(learning_rate)

        if self.model_type == 'moe':
            if expert_usage is not None:
                self.history['expert_usage'].append(expert_usage)
            if router_loss is not None:
                self.history['router_loss'].append(router_loss)

    def state_dict(self) -> Dict[str, Any]:
        """与 experiments/rt-detr/src/misc/training_visualizer.py 一致，供 checkpoint 保存。"""
        return {
            'model_type': self.model_type,
            'history': self.history,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """与 RT-DETR 一致；从 checkpoint 恢复 history（当前 log_dir / experiment_name 由 Trainer 决定）。"""
        if not state_dict:
            return
        if 'model_type' in state_dict:
            self.model_type = state_dict['model_type']
        if 'history' in state_dict:
            old = state_dict['history']
            for k in self.history:
                if k in old:
                    self.history[k] = old[k]

    def export_to_csv(self, filename: str = 'training_history.csv') -> None:
        """仅导出主曲线标量（与精简策略一致）；细粒度指标见训练日志与 test 输出。"""
        csv_path = self.log_dir / filename
        self.log_dir.mkdir(parents=True, exist_ok=True)
        num_epochs = len(self.history['train_loss'])
        if num_epochs == 0:
            return

        fieldnames = [
            'epoch',
            'train_loss',
            'val_loss',
            'mAP_0.5',
            'mAP_0.75',
            'mAP_0.5_0.95',
            'learning_rate',
        ]
        rows: List[Dict[str, Any]] = []
        for i in range(num_epochs):
            rows.append({
                'epoch': i + 1,
                'train_loss': self.history['train_loss'][i],
                'val_loss': self.history['val_loss'][i],
                'mAP_0.5': self.history['mAP_0_5'][i],
                'mAP_0.75': self.history['mAP_0_75'][i],
                'mAP_0.5_0.95': self.history['mAP_0_5_0_95'][i],
                'learning_rate': self.history['learning_rate'][i],
            })

        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def get_best_epoch(self, metric: str = 'val_loss', mode: str = 'min') -> int:
        """与 RT-DETR 一致：返回最佳 epoch（1-based）。"""
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
    
    def plot(self) -> None:
        """CaS_DETR 仅保存一张主曲线图；MOE 仍可用 2×2 + 专家历史（若需）。"""
        if len(self.history['train_loss']) == 0:
            return

        if self.model_type == 'moe' and len(self.history['expert_usage']) > 0:
            self._plot_with_experts()
        else:
            self._plot_standard()
    
    def _plot_standard(self) -> None:
        """绘制标准训练曲线（无MOE专家信息）。"""
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
        
        # 3. 学习率（与 RT-DETR 一致：对数坐标）
        ax = axes[2]
        if max(self.history['learning_rate']) > 0:
            ax.plot(epochs, self.history['learning_rate'], 'orange', linewidth=2)
            ax.set_yscale('log')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Learning Rate', fontsize=12)
        ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        self.log_dir.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(self.log_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_with_experts(self) -> None:
        """与 experiments/rt-detr：MOE 2×2 主图 + expert_usage_history.png。"""
        epochs = range(1, len(self.history['train_loss']) + 1)

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        title = f'{self.experiment_name.upper()} Training Curves' if self.experiment_name else 'MOE RT-DETR Training Curves'
        fig.suptitle(title, fontsize=16, fontweight='bold')

        ax = axes[0, 0]
        ax.plot(epochs, self.history['train_loss'], 'b-o', label='Train Loss', linewidth=2, markersize=4)
        ax.plot(epochs, self.history['val_loss'], 'r-s', label='Val Loss', linewidth=2, markersize=4)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Loss Curves', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        if max(self.history['mAP_0_5']) > 0:
            ax.plot(epochs, self.history['mAP_0_5'], 'g-^', label='mAP@0.5', linewidth=2, markersize=4)
            ax.plot(epochs, self.history['mAP_0_75'], 'c-v', label='mAP@0.75', linewidth=2, markersize=4)
            ax.plot(epochs, self.history['mAP_0_5_0_95'], 'm-d', label='mAP@[0.5:0.95]', linewidth=2, markersize=4)
            ax.legend(fontsize=10)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('mAP', fontsize=12)
        ax.set_title('mAP Metrics', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        if max(self.history['learning_rate']) > 0:
            ax.plot(epochs, self.history['learning_rate'], 'orange', linewidth=2)
            ax.set_yscale('log')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Learning Rate', fontsize=12)
        ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        if self.history['expert_usage']:
            latest_usage = self.history['expert_usage'][-1]
            expert_ids = [f'Expert{i}' for i in range(len(latest_usage))]
            colors = plt.cm.viridis(np.linspace(0, 1, len(latest_usage)))
            bars = ax.bar(expert_ids, latest_usage, color=colors, alpha=0.7, edgecolor='black')
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f'{height:.2%}',
                    ha='center',
                    va='bottom',
                    fontsize=10,
                    fontweight='bold',
                )
            ax.set_xlabel('Expert ID', fontsize=12)
            ax.set_ylabel('Usage Rate', fontsize=12)
            ax.set_title(f'Expert Usage Distribution (Epoch {len(epochs)})', fontsize=14, fontweight='bold')
            ax.set_ylim(0, max(latest_usage) * 1.2)
            ax.grid(True, alpha=0.3, axis='y')

        self.log_dir.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(self.log_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
        plt.close()

        self._plot_expert_usage_history()

    def _plot_expert_usage_history(self) -> None:
        """与 RT-DETR 一致：专家使用率随 epoch 变化。"""
        if not self.history['expert_usage']:
            return
        fig, ax = plt.subplots(figsize=(12, 6))
        epochs = range(1, len(self.history['expert_usage']) + 1)
        expert_usage_array = np.array(self.history['expert_usage'])
        num_experts = expert_usage_array.shape[1]
        colors = plt.cm.tab10(np.linspace(0, 1, num_experts))
        for i in range(num_experts):
            ax.plot(
                epochs,
                expert_usage_array[:, i],
                marker='o',
                label=f'Expert{i}',
                color=colors[i],
                linewidth=2,
                markersize=4,
            )
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Usage Rate', fontsize=12)
        ax.set_title('Expert Usage History', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, ncol=2)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.log_dir / 'expert_usage_history.png', dpi=150, bbox_inches='tight')
        plt.close()


__all__ = ['TrainingVisualizer']
