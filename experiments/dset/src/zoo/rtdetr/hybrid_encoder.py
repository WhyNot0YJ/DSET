"""DSET HybridEncoder - 集成Token Pruning和Patch-MoE"""

import copy
from collections import OrderedDict
from typing import Dict

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from .utils import get_activation
from .token_level_pruning import TokenLevelPruner
from .moe_components import MoELayer

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
                 moe_noise_std=0.1,
                 router_init_std=0.02): 
        super().__init__()
        self.normalize_before = normalize_before
        self.use_moe = use_moe

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)

        # FFN层：支持MoE
        if use_moe:
            # 使用统一的Token-Level MoE层
            self.moe_layer = MoELayer(
                d_model=d_model,
                dim_feedforward=dim_feedforward,
                num_experts=num_experts,
                top_k=moe_top_k,
                dropout=dropout,
                activation=activation,
                noise_std=moe_noise_std,
                router_init_std=router_init_std # [新增]
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
            # spatial_shape 仅作为元数据传递，不再用于计算
            return self.moe_layer(src, spatial_shape=spatial_shape)
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
                 encoder_moe_num_experts=4,
                 encoder_moe_top_k=2,
                 # Predictor type selection
                 predictor_type='cnn',
                 # CASS (Context-Aware Soft Supervision) 参数
                 use_cass=False,
                 cass_expansion_ratio=0.3,
                 cass_min_size=1.0,
                 cass_decay_type='gaussian',
                 # CASS Varifocal Loss 参数
                 use_focal_loss=True,
                 cass_focal_alpha=0.75,
                 cass_focal_beta=2.0,
                 # MoE noise_std parameter
                 moe_noise_std=0.1,
                 router_init_std=0.02): # [修复] 恢复为冠军版本值
        """
        Args:
            token_keep_ratio: Patch retention ratio (0.5-0.7)
            token_pruning_warmup_epochs: Warmup epochs for pruning
            encoder_moe_num_experts: Number of experts for Encoder-MoE
            encoder_moe_top_k: Top-K experts for Encoder-MoE
            predictor_type: Type of importance predictor ('cnn' or 'linear')
            use_cass: Whether to use Context-Aware Soft Supervision
            cass_expansion_ratio: Context band expansion ratio (0.2-0.3)
            cass_min_size: Minimum box size on feature map (protects small objects)
            cass_decay_type: Decay type for context band ('gaussian' or 'linear')
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
        
        # DSET dual-sparse parameters - 保存参数以便后续使用
        self.token_keep_ratio = token_keep_ratio
        self.token_pruning_warmup_epochs = token_pruning_warmup_epochs
        self.encoder_moe_num_experts = encoder_moe_num_experts
        self.encoder_moe_top_k = encoder_moe_top_k
        
        # Predictor type
        self.predictor_type = predictor_type
        
        # CASS parameters - 保存参数以便后续使用
        self.use_cass = use_cass
        self.cass_expansion_ratio = cass_expansion_ratio
        self.cass_min_size = cass_min_size
        self.cass_decay_type = cass_decay_type
        # CASS Varifocal Loss (VFL) parameters
        self.use_focal_loss = use_focal_loss
        self.cass_focal_alpha = cass_focal_alpha
        self.cass_focal_beta = cass_focal_beta
        
        # MoE parameters
        self.moe_noise_std = moe_noise_std
        self.router_init_std = router_init_std # [新增]
        
        self.use_encoder_moe = True
        self.use_token_pruning = True
        self.use_token_level_pruning = True
        
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

        # [HSP 核心修改] Support hierarchical scale-aware pruning
        self.token_pruners = nn.ModuleList()
        for enc_ind in use_encoder_idx:
            # 1. Determine keep_ratio for current layer
            if isinstance(token_keep_ratio, dict):
                # If dict, get by layer index (0=P3, 1=P4, 2=P5), default to 0.7
                current_keep_ratio = token_keep_ratio.get(enc_ind, 0.7)
            else:
                # If single float, keep old behavior
                current_keep_ratio = token_keep_ratio

            # 2. Log HSP strategy (optional)
            
            # 3. Create pruner with CASS support
            pruner = TokenLevelPruner(
                input_dim=hidden_dim,
                keep_ratio=current_keep_ratio,  # Use layer-specific ratio
                adaptive=True,
                min_tokens=self._calculate_min_tokens_for_layer(),
                warmup_epochs=token_pruning_warmup_epochs,
                prune_in_eval=True,
                # Predictor type selection
                predictor_type=predictor_type,
                # CASS parameters
                use_cass=use_cass,
                cass_expansion_ratio=cass_expansion_ratio,
                cass_min_size=cass_min_size,
                cass_decay_type=cass_decay_type,
                # CASS Varifocal Loss (VFL) parameters
                use_focal_loss=use_focal_loss,
                cass_focal_alpha=cass_focal_alpha,
                cass_focal_beta=cass_focal_beta
            )
            self.token_pruners.append(pruner)
        
        encoder_layer = TransformerEncoderLayer(
            hidden_dim, 
            nhead=nhead,
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            activation=enc_act,
            use_moe=True,
            num_experts=encoder_moe_num_experts,
            moe_top_k=encoder_moe_top_k,
            moe_noise_std=moe_noise_std,
            router_init_std=router_init_std) # [新增]

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
    
    def _calculate_min_tokens_for_layer(self) -> int:
        """
        Calculate min_tokens.
        Returns a fixed safe minimum (1) to ensure code stability,
        leaving the actual pruning ratio control entirely to `keep_ratio`.
        """
        # A small constant to prevent empty tensors/crashes
        # Changed from 16 to 1 to allow extreme pruning (e.g. 98% pruned)
        return 1

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
        """Compute Encoder Patch-MoE balance loss."""
        # Patch-MoE is always enabled
        from .moe_components import compute_moe_balance_loss
        
        router_logits_list = encoder_info.get('moe_router_logits', [])
        expert_indices_list = encoder_info.get('moe_expert_indices', [])
        
        if len(router_logits_list) == 0:
            device = None
            if 'moe_expert_indices' in encoder_info and len(encoder_info['moe_expert_indices']) > 0:
                device = encoder_info['moe_expert_indices'][0].device
            zero_tensor = torch.tensor(0.0, device=device) if device is not None else torch.tensor(0.0)
            return {'balance_loss': zero_tensor}
        
        num_experts = router_logits_list[0].shape[-1] if len(router_logits_list) > 0 else 4
        top_k = self.encoder_moe_top_k if hasattr(self, 'encoder_moe_top_k') else 2
        balance_loss = compute_moe_balance_loss(router_logits_list, num_experts, expert_indices_list, top_k=top_k)
        
        return {'balance_loss': balance_loss}

    def forward(self, feats, return_encoder_info=False):
        assert len(feats) == len(self.in_channels)
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        
        # [修复] 共享层模式下，每个 Batch 开始前清空 MoE 记录
        for encoder in self.encoder:
            for layer in encoder.layers:
                if hasattr(layer, 'moe_layer') and hasattr(layer.moe_layer, 'reset_cache'):
                    layer.moe_layer.reset_cache()
        
        encoder_info = {
            'token_pruning_ratios': [],
            'importance_scores_list': [],
            'feat_shapes_list': [],  # Store feature map shapes for CASS
            'moe_router_logits': [],
            'moe_expert_indices': []
        }
        
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                h_original, w_original = h, w
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)  # [B, H*W, C]
                
                # 1. 生成完整原始分辨率的 PosEmbed
                pos_embed_full = self.build_2d_sincos_position_embedding(
                    w_original, h_original, self.hidden_dim, self.pe_temperature
                ).to(src_flatten.device)  # [1, H*W, C]
                pos_embed_full = pos_embed_full.squeeze(0)  # [H*W, C]
                
                # 2. 执行 Token Pruning
                src_flatten, kept_indices, prune_info = self.token_pruners[i](
                    src_flatten, 
                    spatial_shape=(h, w),
                    return_indices=True
                )
                encoder_info['token_pruning_ratios'].append(prune_info.get('pruning_ratio', 0.0))
                
                # 关键修复：只要 info 里有分数，就收集，不管当前是否在进行物理剪枝
                if 'token_importance_scores' in prune_info and prune_info['token_importance_scores'] is not None:
                    encoder_info['importance_scores_list'].append(prune_info['token_importance_scores'])
                    encoder_info['feat_shapes_list'].append((h, w))  # Store corresponding feature shape
                
                # 3. 使用 Gather 提取对应的 PosEmbed
                # kept_indices: [B, N_kept] (可能包含 -1 作为填充值)
                B = src_flatten.shape[0]
                N_kept = src_flatten.shape[1]
                
                if kept_indices is not None:
                    # 确保 kept_indices 是 2D: [B, N_kept]
                    if kept_indices.dim() == 1:
                        kept_indices = kept_indices.unsqueeze(0).expand(B, -1)
                    elif kept_indices.dim() == 2 and kept_indices.shape[0] == 1:
                        kept_indices = kept_indices.expand(B, -1)
                    
                    # 处理 -1 填充值：将 -1 替换为 0（安全值，实际不会使用）
                    valid_mask = (kept_indices >= 0) & (kept_indices < h_original * w_original)
                    kept_indices_clean = kept_indices.clamp(0, h_original * w_original - 1)
                    
                    # Gather: 从完整PosEmbed中提取对应位置
                    pos_embed_full_batch = pos_embed_full.unsqueeze(0).expand(B, -1, -1)  # [B, H*W, C]
                    batch_indices = torch.arange(B, device=kept_indices.device).unsqueeze(1).expand(-1, N_kept)  # [B, N_kept]
                    pos_embed = pos_embed_full_batch[batch_indices, kept_indices_clean]  # [B, N_kept, C]
                    
                    # 将无效位置的位置编码设为0
                    pos_embed = pos_embed * valid_mask.unsqueeze(-1)
                else:
                    # 如果没有 kept_indices（warmup期间），使用所有位置
                    pos_embed = pos_embed_full.unsqueeze(0).expand(B, -1, -1)  # [B, H*W, C]
                
                # 4. Encoder 处理（spatial_shape 仅作为元数据，不再用于计算）
                memory :torch.Tensor = self.encoder[i](
                    src_flatten, 
                    pos_embed=pos_embed, 
                    spatial_shape=None  # 不再用于reshape
                )
                
                # 5. 收集 MoE 信息（支持列表化的缓存）
                for layer in self.encoder[i].layers:
                    if hasattr(layer, 'moe_layer'):
                        moe_layer = layer.moe_layer
                        # 使用 extend 将列表内容合并
                        if hasattr(moe_layer, 'router_logits_cache') and moe_layer.router_logits_cache:
                            encoder_info['moe_router_logits'].extend(moe_layer.router_logits_cache)
                        if hasattr(moe_layer, 'expert_indices_cache') and moe_layer.expert_indices_cache:
                            encoder_info['moe_expert_indices'].extend(moe_layer.expert_indices_cache)
                
                # 6. 特征还原：极致向量化 Scatter 模式
                # memory: [B, N_kept, C]
                memory_2d_flat = torch.zeros(
                    B, h_original * w_original, self.hidden_dim,
                    device=memory.device, dtype=memory.dtype
                )
                
                if kept_indices is not None:
                    # 彻底消除 Python 循环，并针对单 Batch 进一步简化
                    valid_mask = (kept_indices >= 0)
                    
                    if B == 1:
                        # 针对推理（BS=1）的极简路径
                        valid_indices = kept_indices.view(-1)[valid_mask.view(-1)]
                        valid_features = memory.view(-1, self.hidden_dim)[valid_mask.view(-1)]
                    else:
                        # 针对训练（BS>1）的向量化路径
                        batch_offsets = torch.arange(B, device=memory.device).view(B, 1) * (h_original * w_original)
                        global_indices = (kept_indices + batch_offsets).view(-1)
                        
                        valid_mask_flat = valid_mask.view(-1)
                        valid_indices = global_indices[valid_mask_flat]
                        valid_features = memory.view(-1, self.hidden_dim)[valid_mask_flat]
                    
                    # 使用 index_copy_ 或 index_put_ 进行高速填充
                    memory_2d_flat.view(-1, self.hidden_dim).index_copy_(0, valid_indices, valid_features)
                else:
                    # 如果没有 kept_indices（warmup期间），直接填充
                    if memory.shape[1] == h_original * w_original:
                        memory_2d_flat = memory
                    else:
                        memory_2d_flat[:, :N_kept] = memory
                
                # Reshape 回 [B, C, H_original, W_original]
                memory_2d = memory_2d_flat.permute(0, 2, 1).reshape(B, self.hidden_dim, h_original, w_original).contiguous()
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
        
        # 即使 return_encoder_info=False，为了可视化也通过 hack 方式挂载
        # 这是一个简单的 hack，用于在 inference mode 下让外部 hook 能够访问 encoder_info
        if not self.training:
            # 将 encoder_info 附加到输出张量列表的第一个元素上作为属性
            if hasattr(outs, '__len__') and len(outs) > 0 and isinstance(outs[0], torch.Tensor):
                setattr(outs[0], 'encoder_info', encoder_info)
        
        return outs
