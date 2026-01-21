"""训练过程可视化工具

提供训练曲线绘制、指标记录和状态保存功能，支持多种模型类型。

主要功能：
- 自动记录训练/验证指标
- 实时绘制训练曲线
- 支持checkpoint保存和恢复
- 专为MOE模型设计的专家使用率可视化
- 专为DSET模型设计的双稀疏机制可视化（Token Pruning + 双MoE）
- 灵活的接口，易于集成到不同训练脚本

支持的模型类型：
- 'standard': 标准模型
- 'moe': MOE模型（单MoE）
- 'dset': DSET模型（Token Pruning + Encoder MoE + Decoder MoE）

用法示例：
    from src.misc.training_visualizer import TrainingVisualizer
    
    # MOE模型
    visualizer = TrainingVisualizer(log_dir='logs/exp1', model_type='moe')
    visualizer.record(
        epoch=0,
        train_loss=1234.5,
        val_loss=567.8,
        mAP_0_5=0.123,
        learning_rate=1e-5,
        expert_usage=[0.15, 0.18, 0.16, 0.17, 0.19, 0.15],
        router_loss=0.05
    )
    
    # DSET模型
    visualizer = TrainingVisualizer(log_dir='logs/dset_exp', model_type='dset')
    visualizer.record(
        epoch=0,
        train_loss=1234.5,
        detection_loss=1200.0,
        encoder_moe_loss=15.0,
        decoder_moe_loss=18.5,
        token_pruning_loss=1.0,
        token_pruning_ratio=0.15,
        encoder_expert_usage=[0.24, 0.26, 0.25, 0.25],
        decoder_expert_usage=[0.16, 0.17, 0.18, 0.17, 0.16, 0.16]
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
            model_type: 模型类型
                - 'standard': 标准模型
                - 'moe': MOE模型（单MoE）
                - 'dset': DSET模型（Token Pruning + 双MoE）
            experiment_name: 实验名称，用于在曲线标题中显示（如 'dset2_r18'）
        """
        self.log_dir = Path(log_dir)
        self.model_type = model_type
        self.experiment_name = experiment_name
        
        # 初始化历史记录
        self.history = {
            # 基础指标
            'train_loss': [],
            'val_loss': [],
            'mAP_0_5': [],
            'mAP_0_75': [],
            'mAP_0_5_0_95': [],
            'learning_rate': [],
            
            # MOE模型专用
            'expert_usage': [],  # MOE/DSET的decoder专家使用率
            'router_loss': [],   # MOE balance loss
            
            # DSET模型专用
            'detection_loss': [],
            'encoder_moe_loss': [],
            'decoder_moe_loss': [],
            'token_pruning_loss': [],
            'token_pruning_ratio': [],
            'encoder_expert_usage': [],  # DSET的encoder (Patch-MoE) 专家使用率
            'decoder_expert_usage': [],  # DSET的decoder MoE 专家使用率
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
        # DSET专用参数
        detection_loss: float = 0.0,
        encoder_moe_loss: float = 0.0,
        decoder_moe_loss: float = 0.0,
        token_pruning_loss: float = 0.0,
        token_pruning_ratio: float = 0.0,
        encoder_expert_usage: Optional[List[float]] = None,
        decoder_expert_usage: Optional[List[float]] = None
    ) -> None:
        """记录一个epoch的训练指标。
        
        Args:
            epoch: 当前epoch编号
            train_loss: 训练总损失
            val_loss: 验证总损失
            mAP_0_5: mAP@0.5指标
            mAP_0_75: mAP@0.75指标
            mAP_0_5_0_95: mAP@[0.5:0.95]指标
            learning_rate: 当前学习率
            expert_usage: 专家使用率列表（MOE模型的decoder专家）
            router_loss: 路由器损失（MOE模型）
            detection_loss: 检测损失（DSET模型）
            encoder_moe_loss: Encoder MoE balance loss（DSET模型）
            decoder_moe_loss: Decoder MoE balance loss（DSET模型）
            token_pruning_loss: Token pruning辅助损失（DSET模型）
            token_pruning_ratio: 实际token pruning比例（DSET模型）
            encoder_expert_usage: Encoder (Patch-MoE) 专家使用率（DSET模型）
            decoder_expert_usage: Decoder MoE 专家使用率（DSET模型）
        """
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['mAP_0_5'].append(mAP_0_5)
        self.history['mAP_0_75'].append(mAP_0_75)
        self.history['mAP_0_5_0_95'].append(mAP_0_5_0_95)
        self.history['learning_rate'].append(learning_rate)
        
        # MOE模型参数
        if expert_usage is not None:
            self.history['expert_usage'].append(expert_usage)
        if router_loss is not None:
            self.history['router_loss'].append(router_loss)
        
        # DSET模型参数
        if self.model_type == 'dset':
            self.history['detection_loss'].append(detection_loss)
            self.history['encoder_moe_loss'].append(encoder_moe_loss)
            self.history['decoder_moe_loss'].append(decoder_moe_loss)
            self.history['token_pruning_loss'].append(token_pruning_loss)
            self.history['token_pruning_ratio'].append(token_pruning_ratio)
            if encoder_expert_usage is not None:
                self.history['encoder_expert_usage'].append(encoder_expert_usage)
            if decoder_expert_usage is not None:
                self.history['decoder_expert_usage'].append(decoder_expert_usage)
    
    def plot(self) -> None:
        """绘制并保存训练曲线。
        
        根据model_type自动选择合适的绘图布局：
        - 标准模型: 3个子图（损失、mAP、学习率）
        - MOE模型: 4个子图（损失、mAP、学习率、专家使用率）
        - DSET模型: 多图（主曲线、Token Pruning分析、双MoE分析、Loss分解）
        
        生成的图像保存在log_dir下。
        """
        if len(self.history['train_loss']) == 0:
            return
        
        if self.model_type == 'dset':
            self._plot_dset()
        elif self.model_type == 'moe' and len(self.history['expert_usage']) > 0:
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
        
        # 创建1x3的子图（只显示三张图）
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        title = f'{self.experiment_name.upper()} Training Curves' if self.experiment_name else 'MOE RT-DETR Training Curves'
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
        if len(self.history['learning_rate']) > 0 and max(self.history['learning_rate']) > 0:
            ax.plot(epochs, self.history['learning_rate'], 
                    'orange', linewidth=2, marker='o', markersize=3)
            ax.set_yscale('log')
        else:
            ax.text(0.5, 0.5, 'Learning Rate data not available', 
                   ha='center', va='center', fontsize=12, color='gray',
                   transform=ax.transAxes)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Learning Rate', fontsize=12)
        ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
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
    
    def _plot_dset(self) -> None:
        """绘制DSET模型的专用可视化（双稀疏架构）。"""
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # 1. 主训练曲线 (2x2布局)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        title = f'DSET Training: {self.experiment_name}' if self.experiment_name else 'DSET Training Curves'
        fig.suptitle(title, fontsize=18, fontweight='bold')
        
        # 1.1 Total Loss
        ax = axes[0, 0]
        ax.plot(epochs, self.history['train_loss'], 'b-o', label='Train Loss', linewidth=2, markersize=3)
        if max(self.history['val_loss']) > 0:
            ax.plot(epochs, self.history['val_loss'], 'r-s', label='Val Loss', linewidth=2, markersize=3)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Total Loss', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 1.2 mAP Metrics
        ax = axes[0, 1]
        if max(self.history['mAP_0_5']) > 0:
            ax.plot(epochs, self.history['mAP_0_5'], 'g-^', label='mAP@0.5', linewidth=2, markersize=3)
            ax.plot(epochs, self.history['mAP_0_75'], 'c-v', label='mAP@0.75', linewidth=2, markersize=3)
            ax.plot(epochs, self.history['mAP_0_5_0_95'], 'm-d', label='mAP@[0.5:0.95]', linewidth=2, markersize=3)
            ax.legend(fontsize=10)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('mAP', fontsize=12)
        ax.set_title('Detection Performance', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 1.3 Token Pruning Ratio & Loss (双y轴)
        ax = axes[1, 0]
        has_ratio_data = len(self.history['token_pruning_ratio']) > 0
        has_loss_data = len(self.history['token_pruning_loss']) > 0
        
        if has_ratio_data or has_loss_data:
            pruning_epochs = epochs[:max(len(self.history['token_pruning_ratio']) if has_ratio_data else 0,
                                         len(self.history['token_pruning_loss']) if has_loss_data else 0)]
            
            # 左y轴：Pruning Ratio
            if has_ratio_data:
                ax.plot(pruning_epochs, self.history['token_pruning_ratio'], 'purple', 
                       linewidth=2.5, label='Actual Pruning Ratio', marker='o', markersize=3)
                ax.set_ylabel('Pruning Ratio', fontsize=12, color='purple')
                ax.tick_params(axis='y', labelcolor='purple')
            else:
                # 如果没有ratio数据，左y轴显示loss
                loss_epochs = epochs[:len(self.history['token_pruning_loss'])]
                ax.plot(loss_epochs, self.history['token_pruning_loss'], 'orange', 
                       linewidth=2, label='Token Pruning Loss', marker='s', markersize=3, linestyle='--')
                ax.set_ylabel('Pruning Loss', fontsize=12, color='orange')
                ax.tick_params(axis='y', labelcolor='orange')
            
            # 右y轴：Pruning Loss（仅当同时有ratio和loss数据时）
            if has_loss_data and has_ratio_data:
                ax2 = ax.twinx()
                loss_epochs = epochs[:len(self.history['token_pruning_loss'])]
                ax2.plot(loss_epochs, self.history['token_pruning_loss'], 'orange', 
                        linewidth=2, label='Token Pruning Loss', marker='s', markersize=3, linestyle='--')
                ax2.set_ylabel('Pruning Loss', fontsize=12, color='orange')
                ax2.tick_params(axis='y', labelcolor='orange')
                
                # 合并图例
                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc='best')
            elif has_ratio_data:
                # 只有ratio数据，没有loss数据
                ax.legend(fontsize=10)
            else:
                # 只有loss数据，没有ratio数据
                ax.legend(fontsize=10)
            
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_title('Token Pruning Progress', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # 如果ratio都是0但loss存在，添加说明文本
            if has_ratio_data and max(self.history['token_pruning_ratio']) == 0 and has_loss_data:
                ax.text(0.5, 0.95, 'CASS Loss training (pruning ratio = 0, keep_ratio >= 1.0)', 
                       ha='center', va='top', fontsize=10, color='gray',
                       transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            elif has_ratio_data and max(self.history['token_pruning_ratio']) == 0:
                ax.text(0.5, 0.95, 'Pruning disabled (keep_ratio >= 1.0, ratio = 0)', 
                       ha='center', va='top', fontsize=10, color='gray',
                       transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            # 如果没有token pruning数据，显示说明
            ax.text(0.5, 0.5, 'Token Pruning data not available\n(pruning may be disabled or keep_ratio >= 1.0)', 
                   ha='center', va='center', fontsize=12, color='gray',
                   transform=ax.transAxes)
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Pruning Ratio', fontsize=12)
            ax.set_title('Token Pruning Progress', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        # 1.4 Learning Rate
        ax = axes[1, 1]
        if len(self.history['learning_rate']) > 0 and max(self.history['learning_rate']) > 0:
            ax.plot(epochs, self.history['learning_rate'], 'orange', linewidth=2, marker='o', markersize=3)
            ax.set_yscale('log')
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Learning Rate (log)', fontsize=12)
            ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
        else:
            # 如果学习率数据不可用，显示说明
            ax.text(0.5, 0.5, 'Learning Rate data not available', 
                   ha='center', va='center', fontsize=12, color='gray',
                   transform=ax.transAxes)
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Learning Rate (log)', fontsize=12)
            ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.log_dir / 'dset_main_curves.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. Loss分解 (2x2布局)
        if len(self.history['detection_loss']) > 0 and max(self.history['detection_loss']) > 0:
            fig, axes = plt.subplots(2, 2, figsize=(16, 10))
            fig.suptitle('Loss Breakdown Analysis', fontsize=16, fontweight='bold')
            
            loss_epochs = range(1, len(self.history['detection_loss']) + 1)
            
            # 2.1 总loss对比
            ax = axes[0, 0]
            ax.plot(epochs, self.history['train_loss'], 'b-o', label='Total Train Loss', linewidth=2, markersize=3)
            if max(self.history['val_loss']) > 0:
                ax.plot(epochs, self.history['val_loss'], 'r-s', label='Total Val Loss', linewidth=2, markersize=3)
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Loss', fontsize=12)
            ax.set_title('Total Loss', fontsize=13, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # 2.2 Loss组成（堆叠图）
            ax = axes[0, 1]
            if len(self.history['detection_loss']) > 0:
                detection = np.array(self.history['detection_loss'])
                encoder_moe = np.array(self.history['encoder_moe_loss'])
                decoder_moe = np.array(self.history['decoder_moe_loss'])
                token_pruning = np.array(self.history['token_pruning_loss'])
                
                ax.fill_between(loss_epochs, 0, detection, label='Detection Loss', alpha=0.7, color='steelblue')
                ax.fill_between(loss_epochs, detection, detection + decoder_moe, 
                              label='+ Decoder MoE Loss', alpha=0.7, color='lightgreen')
                ax.fill_between(loss_epochs, detection + decoder_moe, detection + decoder_moe + encoder_moe, 
                              label='+ Encoder MoE Loss', alpha=0.7, color='lightcoral')
                ax.fill_between(loss_epochs, detection + decoder_moe + encoder_moe, 
                              detection + decoder_moe + encoder_moe + token_pruning, 
                              label='+ Token Pruning Loss', alpha=0.7, color='plum')
                
                ax.set_xlabel('Epoch', fontsize=12)
                ax.set_ylabel('Loss', fontsize=12)
                ax.set_title('Loss Components (Stacked)', fontsize=13, fontweight='bold')
                ax.legend(fontsize=9, loc='upper right')
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'Loss breakdown data not available', 
                       ha='center', va='center', fontsize=12, color='gray',
                       transform=ax.transAxes)
                ax.set_xlabel('Epoch', fontsize=12)
                ax.set_ylabel('Loss', fontsize=12)
                ax.set_title('Loss Components (Stacked)', fontsize=13, fontweight='bold')
            
            # 2.3 MoE Losses对比
            ax = axes[1, 0]
            if max(self.history['encoder_moe_loss']) > 0:
                ax.plot(loss_epochs, self.history['encoder_moe_loss'], 'b-o', 
                       label='Encoder MoE Loss', linewidth=2, markersize=3)
            if max(self.history['decoder_moe_loss']) > 0:
                ax.plot(loss_epochs, self.history['decoder_moe_loss'], 'g-s', 
                       label='Decoder MoE Loss', linewidth=2, markersize=3)
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Loss', fontsize=12)
            ax.set_title('MoE Balance Losses', fontsize=13, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # 2.4 Token Pruning Loss
            ax = axes[1, 1]
            if max(self.history['token_pruning_loss']) > 0:
                ax.plot(loss_epochs, self.history['token_pruning_loss'], 'purple', 
                       label='Token Pruning Loss', linewidth=2, marker='s', markersize=3)
                ax.legend(fontsize=10)
            else:
                # 如果没有数据，显示说明
                ax.text(0.5, 0.5, 'Token Pruning Loss = 0\n(CASS Loss may be disabled)', 
                       ha='center', va='center', fontsize=11, color='gray',
                       transform=ax.transAxes)
            # 无论是否有数据，都设置标签和标题
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Loss', fontsize=12)
            ax.set_title('Token Pruning Auxiliary Loss', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.log_dir / 'dset_loss_breakdown.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        # 3. 双MoE专家使用率分析
        # 只有当至少有decoder数据时才绘制MoE图表
        if len(self.history['decoder_expert_usage']) > 0:
            fig, axes = plt.subplots(2, 2, figsize=(16, 10))
            fig.suptitle('Dual-MoE Expert Usage Analysis', fontsize=16, fontweight='bold')
            
            # 3.1 Encoder专家使用率 - 当前
            if len(self.history['encoder_expert_usage']) > 0 and len(self.history['encoder_expert_usage'][-1]) > 0:
                ax = axes[0, 0]
                latest_encoder = self.history['encoder_expert_usage'][-1]
                num_encoder_experts = len(latest_encoder)
                colors_enc = plt.cm.Blues(np.linspace(0.5, 0.9, num_encoder_experts))
                bars = ax.bar(range(num_encoder_experts), latest_encoder, color=colors_enc, 
                             edgecolor='black', linewidth=1.5)
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2%}',
                           ha='center', va='bottom', fontsize=10, fontweight='bold')
                ax.axhline(y=1.0/num_encoder_experts, color='red', linestyle='--', linewidth=2, label='Uniform')
                ax.set_xlabel('Encoder Expert ID', fontsize=11)
                ax.set_ylabel('Usage Rate', fontsize=11)
                ax.set_title(f'Encoder (Patch-MoE) - Epoch {len(epochs)}', fontsize=12, fontweight='bold')
                ax.set_xticks(range(num_encoder_experts))
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3, axis='y')
                
                # 3.2 Encoder专家使用率历史
                ax = axes[0, 1]
                encoder_array = np.array(self.history['encoder_expert_usage'])
                expert_epochs = range(1, len(self.history['encoder_expert_usage']) + 1)
                colors_hist = plt.cm.tab10(np.linspace(0, 1, num_encoder_experts))
                for i in range(num_encoder_experts):
                    ax.plot(expert_epochs, encoder_array[:, i], marker='o', label=f'Enc-E{i}',
                           color=colors_hist[i], linewidth=2, markersize=3)
                ax.set_xlabel('Epoch', fontsize=11)
                ax.set_ylabel('Usage Rate', fontsize=11)
                ax.set_title('Encoder Expert Usage History', fontsize=12, fontweight='bold')
                ax.legend(fontsize=9, ncol=2)
                ax.grid(True, alpha=0.3)
            else:
                # 如果没有encoder数据，隐藏前两个子图并添加说明
                for ax in [axes[0, 0], axes[0, 1]]:
                    ax.text(0.5, 0.5, 'Encoder MoE data not available\n(requires encoder_expert_usage)', 
                           ha='center', va='center', fontsize=12, color='gray',
                           transform=ax.transAxes)
                    ax.set_xticks([])
                    ax.set_yticks([])
            
            # 3.3 Decoder专家使用率 - 当前
            if len(self.history['decoder_expert_usage']) > 0 and len(self.history['decoder_expert_usage'][-1]) > 0:
                ax = axes[1, 0]
                latest_decoder = self.history['decoder_expert_usage'][-1]
                num_decoder_experts = len(latest_decoder)
                colors_dec = plt.cm.Greens(np.linspace(0.5, 0.9, num_decoder_experts))
                bars = ax.bar(range(num_decoder_experts), latest_decoder, color=colors_dec, 
                             edgecolor='black', linewidth=1.5)
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2%}',
                           ha='center', va='bottom', fontsize=10, fontweight='bold')
                ax.axhline(y=1.0/num_decoder_experts, color='red', linestyle='--', linewidth=2, label='Uniform')
                ax.set_xlabel('Decoder Expert ID', fontsize=11)
                ax.set_ylabel('Usage Rate', fontsize=11)
                ax.set_title(f'Decoder MoE - Epoch {len(epochs)}', fontsize=12, fontweight='bold')
                ax.set_xticks(range(num_decoder_experts))
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3, axis='y')
                
                # 3.4 Decoder专家使用率历史
                ax = axes[1, 1]
                decoder_array = np.array(self.history['decoder_expert_usage'])
                expert_epochs = range(1, len(self.history['decoder_expert_usage']) + 1)
                colors_hist = plt.cm.tab20(np.linspace(0, 1, num_decoder_experts))
                for i in range(num_decoder_experts):
                    ax.plot(expert_epochs, decoder_array[:, i], marker='o', label=f'Dec-E{i}',
                           color=colors_hist[i], linewidth=2, markersize=3)
                ax.set_xlabel('Epoch', fontsize=11)
                ax.set_ylabel('Usage Rate', fontsize=11)
                ax.set_title('Decoder Expert Usage History', fontsize=12, fontweight='bold')
                ax.legend(fontsize=9, ncol=3)
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.log_dir / 'dset_moe_experts.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        # 仍然生成标准的training_curves.png（向后兼容）
        self._plot_with_experts() if len(self.history['decoder_expert_usage']) > 0 else self._plot_standard()


__all__ = ['TrainingVisualizer']

