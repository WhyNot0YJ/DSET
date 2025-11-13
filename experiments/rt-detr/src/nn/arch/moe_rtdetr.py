"""
MOE-based RT-DETR for Task-Selective Detection
支持三种不同的专家配置方案
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math


class MOEConfig:
    """MOE配置类"""
    
    # 方案A：按类别分专家（6个专家）
    CONFIG_A = {
        "num_experts": 6,
        "expert_mapping": {
            0: "car",
            1: "truck", 
            2: "bus",
            3: "person",
            4: "bicycle",
            5: "motorcycle"
        },
        "class_to_expert": {
            "car": 0,
            "truck": 1,
            "bus": 2, 
            "person": 3,
            "bicycle": 4,
            "motorcycle": 5
        }
    }
    
    # 方案B：按任务复杂度分专家（3个专家）
    CONFIG_B = {
        "num_experts": 3,
        "expert_mapping": {
            0: "vehicles",    # car, truck, bus
            1: "people",      # person
            2: "two_wheelers" # bicycle, motorcycle
        },
        "class_to_expert": {
            "car": 0, "truck": 0, "bus": 0,
            "person": 1,
            "bicycle": 2, "motorcycle": 2
        }
    }
    
    # 方案C：按尺寸分专家（3个专家）
    CONFIG_C = {
        "num_experts": 3,
        "expert_mapping": {
            0: "large_objects",  # truck, bus
            1: "medium_objects", # car
            2: "small_objects"   # person, bicycle, motorcycle
        },
        "class_to_expert": {
            "truck": 0, "bus": 0,
            "car": 1,
            "person": 2, "bicycle": 2, "motorcycle": 2
        }
    }


class RTDETRExpert(nn.Module):
    """单个RT-DETR专家网络"""
    
    def __init__(self, hidden_dim: int = 256, num_queries: int = 300):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        
        # 专家特定的decoder
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=1024,
                dropout=0.1,
                activation='relu',
                batch_first=True
            ),
            num_layers=6
        )
        
        # 检测头
        self.bbox_head = nn.Linear(hidden_dim, 4)  # bbox regression
        self.class_head = nn.Linear(hidden_dim, 1)  # confidence score
        
        # 查询嵌入
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
    def forward(self, encoder_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            encoder_features: [batch_size, seq_len, hidden_dim]
        Returns:
            dict with 'bboxes' and 'scores'
        """
        batch_size = encoder_features.size(0)
        
        # 查询嵌入
        query_embed = self.query_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 解码
        decoder_output = self.decoder(query_embed, encoder_features)
        
        # 预测
        bboxes = self.bbox_head(decoder_output)  # [batch_size, num_queries, 4]
        scores = self.class_head(decoder_output).squeeze(-1)  # [batch_size, num_queries]
        
        return {
            'bboxes': bboxes,
            'scores': scores
        }


class MOERTDETR(nn.Module):
    """MOE-based RT-DETR模型"""
    
    def __init__(self, config_name: str = "A", hidden_dim: int = 256, num_queries: int = 300):
        super().__init__()
        
        # 选择配置
        if config_name == "A":
            self.config = MOEConfig.CONFIG_A
        elif config_name == "B":
            self.config = MOEConfig.CONFIG_B
        elif config_name == "C":
            self.config = MOEConfig.CONFIG_C
        else:
            raise ValueError(f"Unknown config: {config_name}")
        
        self.num_experts = self.config["num_experts"]
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        
        # 共享的backbone和encoder（这里简化，实际应该使用RT-DETR的组件）
        self.backbone = self._build_backbone()
        self.encoder = self._build_encoder()
        
        # 多个专家网络
        self.experts = nn.ModuleList([
            RTDETRExpert(hidden_dim, num_queries) 
            for _ in range(self.num_experts)
        ])
        
        # 专家权重（可学习参数）
        self.expert_weights = nn.Parameter(torch.ones(self.num_experts) / self.num_experts)
        
        # 门控网络（用于第二阶段）
        self.gating_network = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_experts),
            nn.Softmax(dim=-1)
        )
        
    def _build_backbone(self):
        """构建backbone（简化版本）"""
        return nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1),
            nn.Conv2d(64, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
    
    def _build_encoder(self):
        """构建encoder（简化版本）"""
        return nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=8,
                dim_feedforward=1024,
                dropout=0.1,
                activation='relu',
                batch_first=True
            ),
            num_layers=6
        )
    
    def forward(self, images: torch.Tensor, targets: Optional[List[Dict]] = None, 
                language_instructions: Optional[List[str]] = None) -> Dict:
        """
        Args:
            images: [batch_size, 3, H, W]
            targets: 训练时的目标
            language_instructions: 语言指令（第二阶段使用）
        """
        batch_size = images.size(0)
        
        # 特征提取
        features = self.backbone(images)  # [batch_size, 256, 1, 1]
        features = features.flatten(2).transpose(1, 2)  # [batch_size, 1, 256]
        
        # 编码
        encoder_features = self.encoder(features)  # [batch_size, 1, 256]
        
        # 所有专家前向传播
        expert_outputs = []
        for expert in self.experts:
            output = expert(encoder_features)
            expert_outputs.append(output)
        
        if self.training and targets is not None:
            # 训练模式：计算每个专家的损失
            expert_losses = []
            for i, expert_output in enumerate(expert_outputs):
                expert_loss = self._compute_expert_loss(expert_output, targets, i)
                expert_losses.append(expert_loss)
            
            return {
                'expert_outputs': expert_outputs,
                'expert_losses': expert_losses,
                'total_loss': sum(expert_losses)
            }
        else:
            # 推理模式
            if language_instructions is not None:
                # 第二阶段：使用语言门控选择专家
                return self._language_gated_inference(expert_outputs, language_instructions)
            else:
                # 第一阶段：返回所有专家输出
                return {
                    'expert_outputs': expert_outputs
                }
    
    def _compute_expert_loss(self, expert_output: Dict, targets: List[Dict], expert_id: int) -> torch.Tensor:
        """计算单个专家的损失"""
        # 过滤该专家应该处理的目标
        expert_targets = self._filter_targets_for_expert(targets, expert_id)
        
        if len(expert_targets) == 0:
            return torch.tensor(0.0, device=expert_output['bboxes'].device)
        
        # 简化的损失计算
        bbox_loss = F.mse_loss(expert_output['bboxes'], expert_targets[0]['bboxes'])
        score_loss = F.binary_cross_entropy_with_logits(
            expert_output['scores'], 
            expert_targets[0]['scores']
        )
        
        return bbox_loss + score_loss
    
    def _filter_targets_for_expert(self, targets: List[Dict], expert_id: int) -> List[Dict]:
        """为特定专家过滤目标"""
        expert_targets = []
        for target in targets:
            # 根据专家ID过滤类别
            if self._should_expert_handle_target(target, expert_id):
                expert_targets.append(target)
        return expert_targets
    
    def _should_expert_handle_target(self, target: Dict, expert_id: int) -> bool:
        """判断专家是否应该处理该目标"""
        # 这里需要根据实际的target格式来实现
        # 简化版本：假设target包含class信息
        if 'class' in target:
            class_name = target['class']
            if class_name in self.config['class_to_expert']:
                return self.config['class_to_expert'][class_name] == expert_id
        return False
    
    def _language_gated_inference(self, expert_outputs: List[Dict], 
                                 language_instructions: List[str]) -> Dict:
        """使用语言门控进行推理"""
        # 这里需要实现语言编码和门控逻辑
        # 简化版本：随机选择专家
        batch_size = len(language_instructions)
        selected_experts = torch.randint(0, self.num_experts, (batch_size,))
        
        final_outputs = []
        for i, expert_id in enumerate(selected_experts):
            final_outputs.append(expert_outputs[expert_id])
        
        return {
            'final_outputs': final_outputs,
            'selected_experts': selected_experts
        }


def create_moe_model(config_name: str = "A", **kwargs) -> MOERTDETR:
    """创建MOE模型的工厂函数"""
    return MOERTDETR(config_name=config_name, **kwargs)


# 测试代码
if __name__ == "__main__":
    # 测试三种配置
    for config in ["A", "B", "C"]:
        print(f"\n=== 测试配置 {config} ===")
        model = create_moe_model(config)
        print(f"专家数量: {model.num_experts}")
        print(f"专家映射: {model.config['expert_mapping']}")
        
        # 测试前向传播
        images = torch.randn(2, 3, 224, 224)
        outputs = model(images)
        print(f"输出专家数量: {len(outputs['expert_outputs'])}")
        print(f"每个专家输出形状: {outputs['expert_outputs'][0]['bboxes'].shape}")
