"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import random 
import numpy as np 
from typing import List, Dict, Optional, Tuple

from ...core import register


__all__ = ['RTDETR', 'MOERTDETR', 'Router', 'ExpertNetwork']


@register()
class RTDETR(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(self, \
        backbone: nn.Module, 
        encoder: nn.Module, 
        decoder: nn.Module, 
    ):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        
    def forward(self, x, targets=None):
        x = self.backbone(x)
        x = self.encoder(x)        
        x = self.decoder(x, targets)

        return x
    
    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self 


class Router(nn.Module):
    """Router - Decides which experts should be activated"""
    
    def __init__(self, input_dim: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Router Network
        self.router_net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, num_experts),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            features: [batch_size, seq_len, input_dim] Input features
        Returns:
            expert_weights: [batch_size, seq_len, num_experts] Expert weights
            expert_indices: [batch_size, seq_len, top_k] Selected expert indices
            routing_weights: [batch_size, seq_len, top_k] Routing weights
        """
        batch_size, seq_len, _ = features.shape
        
        # Compute expert weights
        expert_logits = self.router_net(features)  # [batch_size, seq_len, num_experts]
        
        # Top-K Selection
        top_k_weights, top_k_indices = torch.topk(expert_logits, self.top_k, dim=-1)
        
        # Renormalize weights
        routing_weights = torch.softmax(top_k_weights, dim=-1)
        
        return expert_logits, top_k_indices, routing_weights


class ExpertNetwork(nn.Module):
    """Single Expert Network"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Expert-specific processing network
        self.expert_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Expert forward pass"""
        return self.expert_net(x)


@register()
class MOERTDETR(nn.Module):
    """MOE RT-DETR Model"""
    __inject__ = ['backbone', 'encoder', 'decoder']
    
    def __init__(self, 
                 backbone: nn.Module, 
                 encoder: nn.Module, 
                 decoder: nn.Module,
                 num_experts: int = 6,
                 top_k: int = 2,
                 config_name: str = "A"):
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder
        self.num_experts = num_experts
        self.top_k = top_k
        self.config_name = config_name
        
        # MOE Config
        self.config = self._get_moe_config(config_name)
        
        # Router
        self.router = Router(256, num_experts, top_k)  # Assume hidden_dim=256
        
        # Expert Networks
        self.experts = nn.ModuleList([
            ExpertNetwork(256, 256, 256) for _ in range(num_experts)
        ])
        
        # Expert Weights (Learnable parameters)
        self.expert_weights = nn.Parameter(torch.ones(num_experts) / num_experts)
        
    def _get_moe_config(self, config_name: str) -> Dict:
        """Get MOE Config"""
        configs = {
            "A": {
                "num_experts": 6,
                "expert_mapping": {
                    0: "car", 1: "truck", 2: "bus",
                    3: "person", 4: "bicycle", 5: "motorcycle"
                },
                "class_to_expert": {
                    "car": 0, "truck": 1, "bus": 2,
                    "person": 3, "bicycle": 4, "motorcycle": 5
                }
            },
            "B": {
                "num_experts": 3,
                "expert_mapping": {
                    0: "vehicles", 1: "people", 2: "two_wheelers"
                },
                "class_to_expert": {
                    "car": 0, "truck": 0, "bus": 0,
                    "person": 1, "bicycle": 2, "motorcycle": 2
                }
            },
            "C": {
                "num_experts": 3,
                "expert_mapping": {
                    0: "large_objects", 1: "medium_objects", 2: "small_objects"
                },
                "class_to_expert": {
                    "truck": 0, "bus": 0, "car": 1,
                    "person": 2, "bicycle": 2, "motorcycle": 2
                }
            }
        }
        return configs.get(config_name, configs["A"])
    
    def forward(self, x, targets=None):
        """Forward pass"""
        # Shared feature extraction
        x = self.backbone(x)
        x = self.encoder(x)
        
        # Get encoded features for router
        if isinstance(x, (list, tuple)):
            encoder_features = x[0] if len(x) > 0 else x
        else:
            encoder_features = x
        
        # Ensure encoder_features is 3D tensor [batch_size, seq_len, hidden_dim]
        if len(encoder_features.shape) == 4:  # [B, C, H, W]
            B, C, H, W = encoder_features.shape
            encoder_features = encoder_features.flatten(2).transpose(1, 2)  # [B, H*W, C]
        
        # Router decides expert selection
        expert_logits, expert_indices, routing_weights = self.router(encoder_features)
        
        # Expert processing
        expert_outputs = []
        for expert in self.experts:
            expert_output = expert(encoder_features)
            expert_outputs.append(expert_output)
        
        # Weighted fusion of expert outputs
        combined_features = self._combine_expert_outputs(
            expert_outputs, expert_indices, routing_weights
        )
        
        # Use fused features for decoding
        if isinstance(x, (list, tuple)):
            # If encoder returns a list, replace the first element
            x[0] = combined_features
        else:
            x = combined_features
        
        # Decoder processing
        x = self.decoder(x, targets)
        
        if self.training and targets is not None:
            # Training mode: Add MOE related info
            if isinstance(x, dict):
                x['expert_logits'] = expert_logits
                x['expert_indices'] = expert_indices
                x['routing_weights'] = routing_weights
                x['router_loss'] = self._compute_router_loss(expert_logits)
            else:
                # If decoder does not return a dict, wrap it
                x = {
                    'outputs': x,
                    'expert_logits': expert_logits,
                    'expert_indices': expert_indices,
                    'routing_weights': routing_weights,
                    'router_loss': self._compute_router_loss(expert_logits)
                }
        
        return x
    
    def _combine_expert_outputs(self, expert_outputs: List[torch.Tensor], 
                               expert_indices: torch.Tensor, 
                               routing_weights: torch.Tensor) -> torch.Tensor:
        """Fuse expert outputs"""
        batch_size, seq_len, top_k = expert_indices.shape
        
        # Initialize output
        combined_output = torch.zeros_like(expert_outputs[0])
        
        # Weighted fusion
        for b in range(batch_size):
            for s in range(seq_len):
                for k in range(top_k):
                    expert_idx = expert_indices[b, s, k].item()
                    weight = routing_weights[b, s, k]
                    combined_output[b, s] += weight * expert_outputs[expert_idx][b, s]
        
        return combined_output
    
    def _compute_router_loss(self, expert_logits: torch.Tensor) -> torch.Tensor:
        """Compute router loss (Load Balance)"""
        # Compute usage frequency for each expert
        expert_usage = torch.mean(expert_logits, dim=[0, 1])  # [num_experts]
        
        # Compute standard deviation of usage (encourage uniform usage)
        usage_std = torch.std(expert_usage)
        
        # Compute load balance loss
        target_usage = 1.0 / self.num_experts
        load_balance_loss = torch.sum((expert_usage - target_usage) ** 2)
        
        return usage_std + load_balance_loss
    
    def deploy(self):
        """Deploy mode"""
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self
