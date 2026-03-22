"""训练过程可视化工具

提供训练曲线绘制、指标记录和状态保存功能，支持多种模型类型。

主要功能：
- 自动记录训练/验证指标
- 实时绘制训练曲线
- 支持checkpoint保存和恢复
- 专为MOE模型设计的专家使用率可视化
- 专为CaS_DETR模型设计的双稀疏机制可视化（Token Pruning + 双MoE）
- 灵活的接口，易于集成到不同训练脚本

支持的模型类型：
- 'standard': 标准模型
- 'moe': MOE模型（单MoE）

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
    
    # CaS_DETR模型
    visualizer = TrainingVisualizer(log_dir='logs/cas_detr_exp', model_type='cas_detr')
    visualizer.record(
        epoch=0,
        train_loss=1234.5,
        detection_loss=1200.0,
        decoder_moe_loss=18.5,
        token_pruning_loss=1.0,
        token_pruning_ratio=0.15,
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
                - 'cas_detr': CaS_DETR模型（Token Pruning + 双MoE）
            experiment_name: 实验名称，用于在曲线标题中显示（如 'cas_detr2_r18'）
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
            'expert_usage': [],  # MOE/CaS_DETR的decoder专家使用率
            'router_loss': [],   # MOE balance loss
            
            # CaS_DETR模型专用
            'detection_loss': [],
            'decoder_moe_loss': [],
            'token_pruning_loss': [],
            'token_pruning_ratio': [],
            'decoder_expert_usage': [],  # CaS_DETR的decoder MoE 专家使用率
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
        # CaS_DETR专用参数
        detection_loss: float = 0.0,
        decoder_moe_loss: float = 0.0,
        token_pruning_loss: float = 0.0,
        token_pruning_ratio: float = 0.0,
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
            detection_loss: 检测损失（CaS_DETR模型）
            decoder_moe_loss: Decoder MoE balance loss（CaS_DETR模型）
            token_pruning_loss: Token pruning辅助损失（CaS_DETR模型）
            token_pruning_ratio: 实际token pruning比例（CaS_DETR模型）
            decoder_expert_usage: Decoder MoE 专家使用率（CaS_DETR模型）
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
        
        # CaS_DETR模型参数
        if self.model_type == 'cas_detr':
            self.history['detection_loss'].append(detection_loss)
            self.history['decoder_moe_loss'].append(decoder_moe_loss)
            self.history['token_pruning_loss'].append(token_pruning_loss)
            self.history['token_pruning_ratio'].append(token_pruning_ratio)
            if decoder_expert_usage is not None:
                self.history['decoder_expert_usage'].append(decoder_expert_usage)
    
    def plot(self) -> None:
        """绘制并保存训练曲线。
        
        根据model_type自动选择合适的绘图布局：
        - 标准模型: 3个子图（损失、mAP、学习率）
        - MOE模型: 4个子图（损失、mAP、学习率、专家使用率）
        - CaS_DETR模型: 多图（主曲线、Token Pruning分析、双MoE分析、Loss分解）
        
        生成的图像保存在log_dir下。
        """
        if len(self.history['train_loss']) == 0:
            return
        
        if self.model_type == 'cas_detr':
            self._plot_cas_detr()
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
        
        # 3. MoE专家使用率分析 (Decoder)
        if len(self.history['decoder_expert_usage']) > 0:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            fig.suptitle('MoE Expert Usage Analysis', fontsize=16, fontweight='bold')
            
            # 3.1 Decoder专家使用率 - 当前
            if len(self.history['decoder_expert_usage']) > 0 and len(self.history['decoder_expert_usage'][-1]) > 0:
                ax = axes[0]
                latest_decoder = self.history['decoder_expert_usage'][-1]
                num_decoder_experts = len(latest_decoder)
                colors_dec = plt.cm.Greens(np.linspace(0.5, 0.9, num_decoder_experts))
                bars = ax.bar(range(num_decoder_experts), latest_decoder, color=colors_dec, edgecolor='black', linewidth=1.5)
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2%}', ha='center', va='bottom', fontsize=10, fontweight='bold')
                
                ax.axhline(y=1.0/num_decoder_experts, color='red', linestyle='--', linewidth=2, label='Uniform')
                ax.set_xlabel('Decoder Expert ID', fontsize=11)
                ax.set_ylabel('Usage Rate', fontsize=11)
                ax.set_title(f'Decoder MoE - Epoch {len(epochs)}', fontsize=12, fontweight='bold')
                ax.set_xticks(range(num_decoder_experts))
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3, axis='y')
                
                # 3.2 Decoder专家使用率历史
                ax = axes[1]
                decoder_array = np.array(self.history['decoder_expert_usage'])
                expert_epochs = range(1, len(self.history['decoder_expert_usage']) + 1)
                colors_hist = plt.cm.tab10(np.linspace(0, 1, num_decoder_experts))
                for i in range(num_decoder_experts):
                    ax.plot(expert_epochs, decoder_array[:, i], label=f'Expert {i}', linewidth=2, color=colors_hist[i])
                
                ax.set_xlabel('Epoch', fontsize=11)
                ax.set_ylabel('Usage Rate', fontsize=11)
                ax.set_title('Decoder Expert Usage History', fontsize=12, fontweight='bold')
                ax.legend(fontsize=9, ncol=3)
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.log_dir / 'cas_detr_moe_experts.png', dpi=150, bbox_inches='tight')
            plt.close()

        # 仍然生成标准的training_curves.png（向后兼容）
        self._plot_with_experts() if len(self.history['decoder_expert_usage']) > 0 else self._plot_standard()

__all__ = ['TrainingVisualizer']
