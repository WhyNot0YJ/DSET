"""DSET HybridEncoder - 集成Token Pruning和Patch-MoE"""

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
                 patch_size=4):
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
                 version='v2',
                 # DSET 双稀疏参数
                 token_keep_ratio=0.7,
                 token_pruning_warmup_epochs=10,
                 patch_moe_num_experts=4,
                 patch_moe_top_k=2,
                 patch_moe_patch_size=4):
        """
        Args:
            token_keep_ratio: Patch保留比例（0.5-0.7）
            token_pruning_warmup_epochs: Pruning warmup epochs
            patch_moe_num_experts: Patch-MoE专家数量
            patch_moe_top_k: Patch-MoE top-k
            patch_moe_patch_size: Patch大小（需与Pruning一致）
        """
        super().__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.out_channels = [hidden_dim for _ in range(len(in_channels))]
        self.out_strides = feat_strides
        
        self.use_patch_moe = True
        self.use_token_pruning = True
        self.use_patch_level_pruning = True
        
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

        # 根据模型结构自动计算每一层的 num_patches 和 min_patches
        # 使用默认 640×640
        image_h = image_w = 640  # 默认值
        
        self.token_pruners = nn.ModuleList([
            PatchLevelPruner(
                input_dim=hidden_dim,
                patch_size=patch_moe_patch_size,
                keep_ratio=token_keep_ratio,
                adaptive=True,
                min_patches=self._calculate_min_patches_for_layer(
                    enc_ind, feat_strides, image_h, image_w, patch_moe_patch_size
                ),
                warmup_epochs=token_pruning_warmup_epochs,
                prune_in_eval=True
            ) for enc_ind in use_encoder_idx
        ])
        
        encoder_layer = TransformerEncoderLayer(
            hidden_dim, 
            nhead=nhead,
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            activation=enc_act,
            use_moe=True,
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
    
    def _calculate_min_patches_for_layer(self, enc_ind: int, feat_strides: list, 
                                        image_h: int, image_w: int, patch_size: int) -> int:
        """
        根据模型结构计算该层的 num_patches 和合适的 min_patches
        
        计算逻辑与 PatchLevelPruner.forward 中的逻辑完全一致：
        - 特征图尺寸 = 输入图像尺寸 / stride
        - patch_h = min(patch_size, feat_h)
        - num_patches_h = (feat_h + patch_h - 1) // patch_h
        - num_patches = num_patches_h * num_patches_w
        
        Args:
            enc_ind: encoder 层索引（对应 feat_strides 的索引）
            feat_strides: 特征图 stride 列表，如 [8, 16, 32]
            image_h: 输入图像高度（如 640）
            image_w: 输入图像宽度（如 640）
            patch_size: patch 大小（如 4）
        
        Returns:
            min_patches: 该层的最小保留 patch 数（确保可以剪枝）
        """
        # 获取该层对应的 stride
        stride = feat_strides[enc_ind] if enc_ind < len(feat_strides) else feat_strides[-1]
        
        # 计算特征图尺寸
        feat_h = image_h // stride
        feat_w = image_w // stride
        
        # 计算 num_patches（与 PatchLevelPruner.forward 中的逻辑完全一致）
        patch_h = min(patch_size, feat_h)
        patch_w = min(patch_size, feat_w)
        num_patches_h = (feat_h + patch_h - 1) // patch_h
        num_patches_w = (feat_w + patch_w - 1) // patch_w
        num_patches = num_patches_h * num_patches_w
        
        # 至少保留 1 个，确保可以剪枝
        min_patches = max(1, int(num_patches * 0.75))
        
        return min_patches

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
        """设置当前epoch"""
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
            device = None
            if 'moe_expert_indices' in encoder_info and len(encoder_info['moe_expert_indices']) > 0:
                device = encoder_info['moe_expert_indices'][0].device
            zero_tensor = torch.tensor(0.0, device=device) if device is not None else torch.tensor(0.0)
            return {'balance_loss': zero_tensor, 'entropy_loss': zero_tensor}
        
        num_experts = router_logits_list[0].shape[-1] if len(router_logits_list) > 0 else 4
        balance_loss = compute_patch_moe_balance_loss(router_logits_list, num_experts, expert_indices_list)
        entropy_loss = compute_patch_moe_entropy_loss(router_logits_list)
        
        return {'balance_loss': balance_loss, 'entropy_loss': entropy_loss}

    def forward(self, feats, return_encoder_info=False):
        assert len(feats) == len(self.in_channels)
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        
        encoder_info = {
            'token_pruning_ratios': [],
            'importance_scores_list': [],
            'moe_router_logits': [],
            'moe_expert_indices': []
        }
        
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                
                src_flatten, kept_indices, prune_info = self.token_pruners[i](
                    src_flatten, 
                    spatial_shape=(h, w),
                    return_indices=True
                )
                encoder_info['token_pruning_ratios'].append(prune_info.get('pruning_ratio', 0.0))
                
                if 'patch_importance_scores' in prune_info:
                    encoder_info['importance_scores_list'].append(prune_info['patch_importance_scores'])
                
                new_spatial_shape = prune_info.get('new_spatial_shape', (h, w))
                h_pruned, w_pruned = new_spatial_shape
                original_spatial_shape = prune_info.get('original_spatial_shape', (h, w))
                h_original, w_original = original_spatial_shape
                kept_indices = None
                
                pos_embed = self.build_2d_sincos_position_embedding(
                    w_pruned, h_pruned, self.hidden_dim, self.pe_temperature).to(src_flatten.device)

                memory :torch.Tensor = self.encoder[i](src_flatten, pos_embed=pos_embed, spatial_shape=(h_pruned, w_pruned))
                
                for layer in self.encoder[i].layers:
                    if hasattr(layer, 'patch_moe_layer'):
                        moe_layer = layer.patch_moe_layer
                        if hasattr(moe_layer, 'router_logits_cache') and moe_layer.router_logits_cache is not None:
                            encoder_info['moe_router_logits'].append(moe_layer.router_logits_cache)
                        if hasattr(moe_layer, 'expert_indices_cache') and moe_layer.expert_indices_cache is not None:
                            encoder_info['moe_expert_indices'].append(moe_layer.expert_indices_cache)
                
                memory_2d = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h_pruned, w_pruned).contiguous()
                if h_pruned != h_original or w_pruned != w_original:
                    memory_2d = F.interpolate(memory_2d, size=(h_original, w_original), mode='bilinear', align_corners=False)
                proj_feats[enc_ind] = memory_2d

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
