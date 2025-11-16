"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.

DSET (Dual-Sparse Expert Transformer) HybridEncoder
集成Token Pruning和Patch-MoE
"""

import copy
from collections import OrderedDict
from typing import Dict

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from .utils import get_activation
from .token_pruning import TokenPruner, SpatialTokenPruner
from .patch_level_pruning import PatchLevelPruner
from .moe_components import PatchMoELayer

from ...core import register


__all__ = ['HybridEncoder']



class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in, 
            ch_out, 
            kernel_size, 
            stride, 
            padding=(kernel_size-1)//2 if padding is None else padding, 
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class RepVggBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act='relu'):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)
        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        if hasattr(self, 'conv'):
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.conv2(x)

        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, 'conv'):
            self.conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, padding=1)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias 

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: ConvNormLayer):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class CSPRepLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 expansion=1.0,
                 bias=None,
                 act="silu"):
        super(CSPRepLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.conv2 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(*[
            RepVggBlock(hidden_channels, hidden_channels, act=act) for _ in range(num_blocks)
        ])
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)


# transformer
class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 normalize_before=False,
                 use_moe=False,
                 num_experts=4,
                 moe_top_k=2,
                 patch_size=8):
        super().__init__()
        self.normalize_before = normalize_before
        self.use_moe = use_moe

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)

        # FFN层：支持MoE
        if use_moe:
            # 使用Patch-MoE层
            self.patch_moe_layer = PatchMoELayer(
                d_model=d_model,
                dim_feedforward=dim_feedforward,
                num_experts=num_experts,
                top_k=moe_top_k,
                dropout=dropout,
                activation=activation,
                patch_size=patch_size
            )
        else:
            # 标准FFN
            self.linear1 = nn.Linear(d_model, dim_feedforward)
            self.dropout = nn.Dropout(dropout)
            self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation) 

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward_ffn(self, src, spatial_shape=None):
        """FFN前向传播（支持MoE）"""
        if self.use_moe:
            return self.patch_moe_layer(src, spatial_shape=spatial_shape)
        else:
            return self.linear2(self.dropout(self.activation(self.linear1(src))))

    def forward(self, src, src_mask=None, pos_embed=None, spatial_shape=None) -> torch.Tensor:
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.forward_ffn(src, spatial_shape=spatial_shape)
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        # 共享MoE设计：所有层共享同一个encoder_layer
        # 这样可以大幅减少参数量，提升推理速度
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, pos_embed=None, spatial_shape=None) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed, spatial_shape=spatial_shape)

        if self.norm is not None:
            output = self.norm(output)

        return output


@register()
class HybridEncoder(nn.Module):
    __share__ = ['eval_spatial_size', ]

    def __init__(self,
                 in_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 hidden_dim=256,
                 nhead=8,
                 dim_feedforward = 1024,
                 dropout=0.0,
                 enc_act='gelu',
                 use_encoder_idx=[2],
                 num_encoder_layers=1,
                 pe_temperature=10000,
                 expansion=1.0,
                 depth_mult=1.0,
                 act='silu',
                 eval_spatial_size=None, 
                 version='v2',
                 # DSET 双稀疏参数
                 token_keep_ratio=0.7,
                 token_pruning_warmup_epochs=10,
                 patch_moe_num_experts=4,
                 patch_moe_top_k=2,
                 patch_moe_patch_size=8):
        """初始化HybridEncoder（支持DSET双稀疏）
        
        新增参数：
            token_keep_ratio: Patch保留比例（0.5-0.7，用于Patch-level Pruning）
            token_pruning_warmup_epochs: Token Pruning warmup epochs
            patch_moe_num_experts: Patch-MoE专家数量
            patch_moe_top_k: Patch-MoE top-k
            patch_moe_patch_size: Patch-MoE patch大小（默认8x8，必须与Patch-level Pruning一致）
        
        注意：
            - Patch-MoE 和 Patch-level Pruning 必然启用（DSET核心特性）
            - 无需配置 use_patch_moe 和 use_token_pruning
        """
        super().__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size        
        self.out_channels = [hidden_dim for _ in range(len(in_channels))]
        self.out_strides = feat_strides
        
        # DSET 双稀疏配置
        # ⚠️ 重要：Patch-MoE 和 Patch-level Pruning 必然启用（DSET核心特性）
        self.use_patch_moe = True  # 固定启用
        self.use_token_pruning = True  # 自动启用（使用 Patch-level Pruning）
        self.use_patch_level_pruning = True  # 使用 Patch-level Pruning（与 Patch-MoE 兼容）
        
        # channel projection
        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            if version == 'v1':
                proj = nn.Sequential(
                    nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(hidden_dim))
            elif version == 'v2':
                proj = nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False)),
                    ('norm', nn.BatchNorm2d(hidden_dim))
                ]))
            else:
                raise AttributeError()
                
            self.input_proj.append(proj)

        # Token/Patch Pruning（Patch-level Pruning 必然启用，与 Patch-MoE 兼容）
        self.token_pruners = nn.ModuleList([
            PatchLevelPruner(
                input_dim=hidden_dim,
                patch_size=patch_moe_patch_size,  # 必须与 Patch-MoE 的 patch_size 一致
                keep_ratio=token_keep_ratio,
                adaptive=True,
                min_patches=10,
                warmup_epochs=token_pruning_warmup_epochs,
                prune_in_eval=True
            ) for _ in range(len(use_encoder_idx))
        ])
        
        # encoder transformer（支持Patch-MoE，必然启用）
        # 共享MoE设计：所有encoder层共享同一个layer
        encoder_layer = TransformerEncoderLayer(
            hidden_dim, 
            nhead=nhead,
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            activation=enc_act,
            use_moe=True,  # Patch-MoE 必然启用
            num_experts=patch_moe_num_experts,
            moe_top_k=patch_moe_top_k,
            patch_size=patch_moe_patch_size)

        self.encoder = nn.ModuleList([
            TransformerEncoder(encoder_layer, num_encoder_layers) for _ in range(len(use_encoder_idx))
        ])

        # top-down fpn
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1, 0, -1):
            self.lateral_convs.append(ConvNormLayer(hidden_dim, hidden_dim, 1, 1, act=act))
            self.fpn_blocks.append(
                CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
            )

        # bottom-up pan
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            self.downsample_convs.append(
                ConvNormLayer(hidden_dim, hidden_dim, 3, 2, act=act)
            )
            self.pan_blocks.append(
                CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
            )

        self._reset_parameters()

    def _reset_parameters(self):
        if self.eval_spatial_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_spatial_size[1] // stride, self.eval_spatial_size[0] // stride,
                    self.hidden_dim, self.pe_temperature)
                setattr(self, f'pos_embed{idx}', pos_embed)
                # self.register_buffer(f'pos_embed{idx}', pos_embed)

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.):
        """
        """
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]
    
    def set_epoch(self, epoch: int):
        """设置当前epoch（用于Token Pruning的渐进式启用）"""
        if self.token_pruners is not None:
            for pruner in self.token_pruners:
                pruner.set_epoch(epoch)
    
    def get_encoder_moe_loss(self, encoder_info: dict) -> Dict[str, torch.Tensor]:
        """计算Encoder Patch-MoE的损失（包含负载均衡损失和熵正则项）
        
        返回：
            Dict包含：
            - 'balance_loss': Patch-level负载均衡损失
            - 'entropy_loss': 熵正则项损失
        """
        # Patch-MoE 必然启用，无需检查
        from .moe_components import compute_patch_moe_balance_loss, compute_patch_moe_entropy_loss
        
        router_logits_list = encoder_info.get('moe_router_logits', [])
        expert_indices_list = encoder_info.get('moe_expert_indices', [])
        
        if len(router_logits_list) == 0:
            # 尝试从其他信息推断device
            device = None
            if 'moe_expert_indices' in encoder_info and len(encoder_info['moe_expert_indices']) > 0:
                device = encoder_info['moe_expert_indices'][0].device
            zero_tensor = torch.tensor(0.0, device=device) if device is not None else torch.tensor(0.0)
            return {'balance_loss': zero_tensor, 'entropy_loss': zero_tensor}
        
        # 假设所有Patch-MoE层使用相同数量的专家
        num_experts = router_logits_list[0].shape[-1] if len(router_logits_list) > 0 else 4
        
        # 计算Patch-MoE专用损失
        balance_loss = compute_patch_moe_balance_loss(router_logits_list, num_experts, expert_indices_list)
        entropy_loss = compute_patch_moe_entropy_loss(router_logits_list)
        
        return {'balance_loss': balance_loss, 'entropy_loss': entropy_loss}

    def forward(self, feats, return_encoder_info=False):
        assert len(feats) == len(self.in_channels)
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        
        # 用于收集Patch-MoE和Token Pruning的统计信息
        encoder_info = {
            'token_pruning_ratios': [],
            'importance_scores_list': [],
            'moe_router_logits': [],
            'moe_expert_indices': []
        }
        
        # encoder
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                # flatten [B, C, H, W] to [B, HxW, C]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                
                # Patch-level Pruning（必然启用，与 Patch-MoE 兼容）
                src_flatten, kept_indices, prune_info = self.token_pruners[i](
                    src_flatten, 
                    spatial_shape=(h, w),
                    return_indices=True
                )
                encoder_info['token_pruning_ratios'].append(prune_info.get('pruning_ratio', 0.0))
                
                # 保存重要性分数用于计算损失（Patch-level Pruning）
                if 'patch_importance_scores' in prune_info:
                    encoder_info['importance_scores_list'].append(prune_info['patch_importance_scores'])
                
                # 更新h, w（Patch-level Pruning 保持规则2D结构）
                new_spatial_shape = prune_info.get('new_spatial_shape', (h, w))
                h, w = new_spatial_shape
                kept_indices = None  # Patch-level Pruning 保持规则结构，不需要kept_indices
                
                # Position embedding
                if self.training or self.eval_spatial_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature).to(src_flatten.device)
                else:
                    pos_embed = getattr(self, f'pos_embed{enc_ind}', None).to(src_flatten.device)
                
                # Patch-level Pruning 保持规则结构，pos_embed不需要选择

                # Encoder forward（支持Patch-MoE）
                memory :torch.Tensor = self.encoder[i](src_flatten, pos_embed=pos_embed, spatial_shape=(h, w))
                
                # 收集Patch-MoE的路由信息（必然启用）
                for layer in self.encoder[i].layers:
                    if hasattr(layer, 'patch_moe_layer'):
                        moe_layer = layer.patch_moe_layer
                        if hasattr(moe_layer, 'router_logits_cache') and moe_layer.router_logits_cache is not None:
                            encoder_info['moe_router_logits'].append(moe_layer.router_logits_cache)
                        if hasattr(moe_layer, 'expert_indices_cache') and moe_layer.expert_indices_cache is not None:
                            encoder_info['moe_expert_indices'].append(moe_layer.expert_indices_cache)
                
                # 恢复特征图（Patch-level Pruning 保持规则2D结构，直接reshape）
                proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()

        # broadcasting and fusion
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_heigh = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_heigh)
            inner_outs[0] = feat_heigh
            upsample_feat = F.interpolate(feat_heigh, scale_factor=2., mode='nearest')
            inner_out = self.fpn_blocks[len(self.in_channels)-1-idx](torch.concat([upsample_feat, feat_low], dim=1))
            inner_outs.insert(0, inner_out)

        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            out = self.pan_blocks[idx](torch.concat([downsample_feat, feat_height], dim=1))
            outs.append(out)

        if return_encoder_info:
            return outs, encoder_info
        return outs
