"""
MOE损失函数
支持专家特定损失计算和专家平衡损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
import math


class MOECriterion(nn.Module):
    """MOE损失函数"""
    
    def __init__(self, num_experts: int, loss_weights: Dict[str, float] = None):
        super().__init__()
        self.num_experts = num_experts
        
        # 损失权重
        self.loss_weights = loss_weights or {
            'bbox_loss': 5.0,
            'cls_loss': 2.0,
            'expert_balance_loss': 0.1
        }
        
        # 专家平衡损失的权重
        self.balance_weight = 0.1
        
    def forward(self, outputs: Dict, targets: List[Dict], **kwargs) -> Dict[str, torch.Tensor]:
        """
        Args:
            outputs: 模型输出，包含expert_outputs和expert_losses
            targets: 目标列表
        Returns:
            损失字典
        """
        expert_outputs = outputs['expert_outputs']
        expert_losses = outputs['expert_losses']
        
        # 计算总损失
        total_loss = sum(expert_losses)
        
        # 计算专家平衡损失
        balance_loss = self._compute_expert_balance_loss(expert_losses)
        
        # 计算专家利用率损失
        utilization_loss = self._compute_utilization_loss(expert_outputs, targets)
        
        # 组合损失
        final_loss = (
            total_loss + 
            self.loss_weights['expert_balance_loss'] * balance_loss +
            self.loss_weights['expert_balance_loss'] * utilization_loss
        )
        
        return {
            'total_loss': final_loss,
            'expert_losses': expert_losses,
            'balance_loss': balance_loss,
            'utilization_loss': utilization_loss
        }
    
    def _compute_expert_balance_loss(self, expert_losses: List[torch.Tensor]) -> torch.Tensor:
        """计算专家平衡损失，鼓励所有专家都有相似的贡献"""
        if len(expert_losses) <= 1:
            return torch.tensor(0.0, device=expert_losses[0].device)
        
        # 计算损失的标准差
        losses = torch.stack(expert_losses)
        loss_std = torch.std(losses)
        
        return loss_std
    
    def _compute_utilization_loss(self, expert_outputs: List[Dict], 
                                 targets: List[Dict]) -> torch.Tensor:
        """计算专家利用率损失，鼓励专家被均匀使用"""
        # 统计每个专家处理的目标数量
        expert_counts = torch.zeros(self.num_experts)
        
        for target in targets:
            # 这里需要根据实际的target格式来实现
            # 简化版本：假设target包含expert_id信息
            if 'expert_id' in target:
                expert_id = target['expert_id']
                if 0 <= expert_id < self.num_experts:
                    expert_counts[expert_id] += 1
        
        # 计算利用率的标准差
        if expert_counts.sum() > 0:
            expert_counts = expert_counts / expert_counts.sum()
            utilization_std = torch.std(expert_counts)
            return utilization_std
        else:
            return torch.tensor(0.0)
    
    def compute_expert_specific_loss(self, expert_output: Dict, targets: List[Dict], 
                                   expert_id: int) -> torch.Tensor:
        """计算专家特定损失"""
        # 过滤该专家应该处理的目标
        expert_targets = self._filter_targets_for_expert(targets, expert_id)
        
        if len(expert_targets) == 0:
            return torch.tensor(0.0, device=expert_output['bboxes'].device)
        
        # 计算bbox损失
        bbox_loss = self._compute_bbox_loss(expert_output['bboxes'], expert_targets)
        
        # 计算分类损失
        cls_loss = self._compute_cls_loss(expert_output['scores'], expert_targets)
        
        total_loss = (
            self.loss_weights['bbox_loss'] * bbox_loss +
            self.loss_weights['cls_loss'] * cls_loss
        )
        
        return total_loss
    
    def _filter_targets_for_expert(self, targets: List[Dict], expert_id: int) -> List[Dict]:
        """为特定专家过滤目标"""
        expert_targets = []
        for target in targets:
            if self._should_expert_handle_target(target, expert_id):
                expert_targets.append(target)
        return expert_targets
    
    def _should_expert_handle_target(self, target: Dict, expert_id: int) -> bool:
        """判断专家是否应该处理该目标"""
        # 这里需要根据实际的target格式来实现
        if 'expert_id' in target:
            return target['expert_id'] == expert_id
        return False
    
    def _compute_bbox_loss(self, pred_bboxes: torch.Tensor, targets: List[Dict]) -> torch.Tensor:
        """计算bbox损失"""
        if len(targets) == 0:
            return torch.tensor(0.0, device=pred_bboxes.device)
        
        # 简化的bbox损失计算
        target_bboxes = torch.stack([t['bboxes'] for t in targets])
        return F.mse_loss(pred_bboxes, target_bboxes)
    
    def _compute_cls_loss(self, pred_scores: torch.Tensor, targets: List[Dict]) -> torch.Tensor:
        """计算分类损失"""
        if len(targets) == 0:
            return torch.tensor(0.0, device=pred_scores.device)
        
        # 简化的分类损失计算
        target_scores = torch.stack([t['scores'] for t in targets])
        return F.binary_cross_entropy_with_logits(pred_scores, target_scores)


class ExpertBalanceLoss(nn.Module):
    """专家平衡损失"""
    
    def __init__(self, num_experts: int, balance_weight: float = 0.1):
        super().__init__()
        self.num_experts = num_experts
        self.balance_weight = balance_weight
    
    def forward(self, expert_losses: List[torch.Tensor]) -> torch.Tensor:
        """计算专家平衡损失"""
        if len(expert_losses) <= 1:
            return torch.tensor(0.0, device=expert_losses[0].device)
        
        losses = torch.stack(expert_losses)
        
        # 计算损失的标准差
        loss_std = torch.std(losses)
        
        # 计算损失的方差
        loss_var = torch.var(losses)
        
        return self.balance_weight * (loss_std + loss_var)


class ExpertUtilizationLoss(nn.Module):
    """专家利用率损失"""
    
    def __init__(self, num_experts: int, utilization_weight: float = 0.1):
        super().__init__()
        self.num_experts = num_experts
        self.utilization_weight = utilization_weight
    
    def forward(self, expert_utilization: torch.Tensor) -> torch.Tensor:
        """计算专家利用率损失"""
        # expert_utilization: [num_experts] 每个专家的利用率
        
        # 计算利用率的标准差
        utilization_std = torch.std(expert_utilization)
        
        # 计算利用率与均匀分布的KL散度
        uniform_dist = torch.ones_like(expert_utilization) / self.num_experts
        kl_div = F.kl_div(
            F.log_softmax(expert_utilization, dim=0),
            uniform_dist,
            reduction='sum'
        )
        
        return self.utilization_weight * (utilization_std + kl_div)


# 测试代码
if __name__ == "__main__":
    # 测试MOE损失函数
    num_experts = 6
    criterion = MOECriterion(num_experts)
    
    # 模拟输出
    expert_outputs = [
        {
            'bboxes': torch.randn(2, 300, 4),
            'scores': torch.randn(2, 300)
        }
        for _ in range(num_experts)
    ]
    
    expert_losses = [torch.tensor(0.5 + i * 0.1) for i in range(num_experts)]
    
    outputs = {
        'expert_outputs': expert_outputs,
        'expert_losses': expert_losses
    }
    
    targets = []  # 简化的目标
    
    loss_dict = criterion(outputs, targets)
    print("损失字典:", loss_dict)
