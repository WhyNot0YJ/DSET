"""CaS_DETR HybridEncoder - 集成Token Pruning"""

import copy
import logging
import os
from collections import OrderedDict
from typing import Dict, List, Optional

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from .utils import get_activation
from .token_level_pruning import TokenLevelPruner

from ...core import register


__all__ = ['HybridEncoder']

_LOGGER = logging.getLogger(__name__)


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


class CAIPPredictor(nn.Module):
    """Context-Aware Importance Predictor (CAIP).

    Enhances token pruning robustness by fusing a lightweight global context
    branch with the existing local Linear scorer and exposes a
    ``scene_complexity`` scalar for dynamic token keep-ratio in the encoder;
    decoder MoE is not driven by this signal.

    Paths
    -----
    * **Local Path** – identical to ``LinearImportancePredictor``
      (fc1 → GELU → Dropout → ``local_feat [B, N, hidden]``).
      Supervised by CASS loss through the final ``importance_scores``.
    * **Global Path** – GAP over the token dimension →
      1×1 Conv → GELU → 1×1 Conv → Sigmoid → ``global_weights [B, hidden]``.
      The pre-sigmoid activation mean is returned as ``scene_complexity``.
    * **Interaction** – element-wise product of ``local_feat`` and
      ``global_weights``, projected to a scalar per token.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 reduction_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        reduced_dim = max(input_dim // reduction_ratio, 16)

        # --- Local Path (mirrors LinearImportancePredictor) ---
        self.local_fc1 = nn.Linear(input_dim, hidden_dim)
        self.local_act = nn.GELU()
        self.local_dropout = nn.Dropout(dropout)
        self.local_fc2 = nn.Linear(hidden_dim, 1)

        # --- Global Path (GAP → 1×1 Conv → GELU → 1×1 Conv → Sigmoid) ---
        self.global_fc1 = nn.Conv1d(input_dim, reduced_dim, kernel_size=1)
        self.global_act = nn.GELU()
        self.global_fc2 = nn.Conv1d(reduced_dim, hidden_dim, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in (self.local_fc1, self.local_fc2):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
        for m in (self.global_fc1, self.global_fc2):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, tokens: torch.Tensor, H: int = 0, W: int = 0):
        """
        Args:
            tokens: [B, N, C] input token sequence (possibly multi-scale concatenated).
            H, W: kept for API compatibility with ``LinearImportancePredictor``.

        Returns:
            importance_scores: [B, N] logits (no sigmoid), CASS-supervised.
            scene_complexity:  scalar – mean pre-sigmoid activation of the
                               global path; higher ⇒ more complex scene.
        """
        # Local Path
        local_feat = self.local_fc1(tokens)       # [B, N, hidden]
        local_feat = self.local_act(local_feat)
        local_feat = self.local_dropout(local_feat)

        # Global Path: GAP → 1×1 Conv → GELU → 1×1 Conv → Sigmoid
        gap = tokens.mean(dim=1, keepdim=True)     # [B, 1, C]
        gap = gap.permute(0, 2, 1)                 # [B, C, 1]
        g = self.global_fc1(gap)                   # [B, reduced, 1]
        g = self.global_act(g)
        pre_sigmoid = self.global_fc2(g)           # [B, hidden, 1]
        global_weights = torch.sigmoid(pre_sigmoid)  # [B, hidden, 1]

        scene_complexity = pre_sigmoid.mean()      # scalar

        # Interaction: modulate local features with global channel weights
        global_weights = global_weights.squeeze(-1).unsqueeze(1)  # [B, 1, hidden]
        modulated = local_feat * global_weights    # [B, N, hidden]

        importance_scores = self.local_fc2(modulated).squeeze(-1)  # [B, N]
        return importance_scores, scene_complexity


# transformer
class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 normalize_before=False):
        super().__init__()
        self.normalize_before = normalize_before

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)

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

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
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
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed)

        if self.norm is not None:
            output = self.norm(output)

        return output


class DetailBranch(nn.Module):
    """Lightweight detail enhancement for high-resolution features.

    Depthwise-separable conv + channel attention (SE), applied to the
    finest FPN level to preserve small-object cues that token pruning may lose.
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.dw_conv = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False)
        self.dw_bn = nn.BatchNorm2d(channels)
        self.pw_conv = nn.Conv2d(channels, channels, 1, bias=False)
        self.pw_bn = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True)

        mid = max(channels // reduction, 16)
        self.se_fc1 = nn.Conv2d(channels, mid, 1)
        self.se_fc2 = nn.Conv2d(mid, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act(self.dw_bn(self.dw_conv(x)))
        out = self.act(self.pw_bn(self.pw_conv(out)))
        # SE channel attention
        w = F.adaptive_avg_pool2d(out, 1)
        w = self.se_fc2(F.relu(self.se_fc1(w), inplace=True)).sigmoid()
        out = out * w
        return out + residual


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
                 use_encoder_idx=[1, 2],
                 num_encoder_layers=1,
                 pe_temperature=10000,
                 expansion=1.0,
                 depth_mult=1.0,
                 act='silu',
                 version='v2',
                 # CaS_DETR 双稀疏参数
                 token_keep_ratio=0.7,
                 enable_cas_predictor=True,
                 # CASS (Context-Aware Soft Supervision) 参数
                 use_cass=False,
                 cass_expansion_ratio=0.3,
                 cass_min_size=1.0,
                 cass_decay_type='gaussian',
                 use_subpixel_offset=True,
                 # CASS Loss 参数
                 cass_loss_type='vfl',  # 'focal' or 'vfl'
                 cass_focal_alpha=0.75,
                 cass_focal_beta=2.0,
                 # CAIP (Context-Aware Importance Predictor) 参数
                 use_caip=False,
                 caip_reduction_ratio=4,
                 caip_complexity_alpha=0.3,
                 # High-resolution detail branch
                 use_detail_branch=False,
                 **kwargs):  # token_pruning_warmup_epochs, caip_complexity_beta 等旧键忽略
        """
        Args:
            token_keep_ratio: Patch retention ratio (0.5-0.7)
            enable_cas_predictor: Whether to enable token pruning and importance predictor
            use_cass: Whether to use Context-Aware Soft Supervision
            cass_expansion_ratio: Context band expansion ratio (0.2-0.8)
            cass_min_size: Minimum box size on feature map (protects small objects)
            cass_decay_type: Decay type for context band ('gaussian' or 'linear')
            use_subpixel_offset: Whether to use sub-pixel offset compensation
            cass_loss_type: Loss type ('focal' for Focal Loss, 'vfl' for Varifocal Loss)
            cass_focal_alpha: Focal/VFL alpha parameter (positive sample weight)
            cass_focal_beta: Focal/VFL beta/gamma parameter (hard example mining strength)
            use_caip: Whether to use CAIP (global context branch for importance scoring)
            caip_reduction_ratio: Channel reduction ratio in the CAIP global path
            caip_complexity_alpha: Sensitivity for dynamic keep-ratio adjustment (0–1)
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

        # Multi-scale level embedding [L, C] — only when L = len(use_encoder_idx) > 1 (single-level RT-DETR skips)
        _n_enc_levels = len(self.use_encoder_idx) if self.use_encoder_idx else 0
        if _n_enc_levels > 1:
            self.level_embed = nn.Parameter(torch.empty(_n_enc_levels, hidden_dim))
            nn.init.normal_(self.level_embed, mean=0.0, std=0.02)
        else:
            self.level_embed = None
        
        # CaS_DETR dual-sparse parameters - 保存参数以便后续使用
        self.token_keep_ratio = token_keep_ratio
        self.enable_cas_predictor = enable_cas_predictor
        
        # CASS parameters - 保存参数以便后续使用
        self.use_cass = use_cass and enable_cas_predictor
        self.cass_expansion_ratio = cass_expansion_ratio
        self.cass_min_size = cass_min_size
        self.cass_decay_type = cass_decay_type
        self.use_subpixel_offset = use_subpixel_offset
        # CASS Loss parameters
        self.cass_loss_type = cass_loss_type
        self.cass_focal_alpha = cass_focal_alpha
        self.cass_focal_beta = cass_focal_beta
        
        self.use_token_pruning = enable_cas_predictor
        self.use_token_level_pruning = enable_cas_predictor
        
        # CAIP parameters
        self.use_caip = use_caip and enable_cas_predictor
        self.caip_complexity_alpha = caip_complexity_alpha
        
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

        # [HSP 核心修改] Shared global multi-scale pruning
        if isinstance(token_keep_ratio, dict):
            ratios = [token_keep_ratio.get(enc_ind, 0.7) for enc_ind in use_encoder_idx]
            global_keep_ratio = sum(ratios) / max(1, len(ratios))
        else:
            global_keep_ratio = token_keep_ratio

        if self.enable_cas_predictor:
            self.shared_token_pruner = TokenLevelPruner(
                input_dim=hidden_dim,
                keep_ratio=global_keep_ratio,
                adaptive=True,
                min_tokens=self._calculate_min_tokens_for_layer(),
                prune_in_eval=True,
                # CASS parameters
                use_cass=self.use_cass,
                cass_expansion_ratio=cass_expansion_ratio,
                cass_min_size=cass_min_size,
                cass_decay_type=cass_decay_type,
                use_subpixel_offset=use_subpixel_offset,
                # CASS Loss parameters
                cass_loss_type=cass_loss_type,
                cass_focal_alpha=cass_focal_alpha,
                cass_focal_beta=cass_focal_beta,
            )
        else:
            self.shared_token_pruner = None
        
        # CAIP predictor (replaces internal Linear scorer when enabled)
        if self.use_caip:
            self.caip_predictor = CAIPPredictor(
                input_dim=hidden_dim,
                hidden_dim=128,
                reduction_ratio=caip_reduction_ratio,
                dropout=0.1,
            )
        else:
            self.caip_predictor = None

        # 训练时按 epoch 汇总各层保留 token 数，由 train.py 在 epoch 结束时调用 finalize
        self._prune_agg_sum_kept: Optional[torch.Tensor] = None
        self._prune_agg_n_images: int = 0
        self._prune_agg_level_sizes: Optional[List[int]] = None
        self._prune_agg_level_names: Optional[List[str]] = None
        
        encoder_layer = TransformerEncoderLayer(
            hidden_dim, 
            nhead=nhead,
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            activation=enc_act)

        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)
        
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
        
        # Detail branch: lightweight enhancement on the finest FPN level (P3)
        self.use_detail_branch = use_detail_branch
        self.detail_branch = DetailBranch(hidden_dim) if use_detail_branch else None
    
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
        """生成2D sincos位置编码。

        必须与特征图展平顺序严格一致：
        feat = proj_feats[enc_ind]                    # [B, C, H, W]
        tokens = feat.flatten(2).permute(0, 2, 1)     # [B, H*W, C]
        即：H 是慢维度（行），W 是快维度（列）。
        """
        grid_h = torch.arange(int(h), dtype=torch.float32)  # 行 (height)
        grid_w = torch.arange(int(w), dtype=torch.float32)  # 列 (width)

        # indexing='ij' → 输出 (H, W)
        grid_h, grid_w = torch.meshgrid(grid_h, grid_w, indexing='ij')

        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'

        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        # 正确对应：grid_w → x (横向), grid_h → y (纵向)
        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        pe = torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)
        return pe[None, :, :]  # [1, H*W, embed_dim]

    @staticmethod
    def _normalize_kept_indices(kept_indices: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Ensure kept_indices shape is [B, N_kept] for gather/scatter."""
        if kept_indices is None:
            return None
        if kept_indices.dim() == 1:
            return kept_indices.unsqueeze(0).expand(batch_size, -1)
        if kept_indices.dim() == 2 and kept_indices.shape[0] == 1:
            return kept_indices.expand(batch_size, -1)
        return kept_indices

    @staticmethod
    def _gather_pos_embed(pos_embed_full: torch.Tensor,
                          kept_indices: torch.Tensor,
                          batch_size: int,
                          total_tokens: int) -> torch.Tensor:
        """Gather positional embeddings for kept tokens from full grid."""
        if kept_indices is None:
            return pos_embed_full.unsqueeze(0).expand(batch_size, -1, -1)

        kept_indices = HybridEncoder._normalize_kept_indices(kept_indices, batch_size)
        valid_mask = (kept_indices >= 0) & (kept_indices < total_tokens)
        kept_indices_clean = kept_indices.clamp(0, total_tokens - 1)

        pos_embed_full_batch = pos_embed_full.unsqueeze(0).expand(batch_size, -1, -1)
        batch_indices = torch.arange(batch_size, device=kept_indices.device).unsqueeze(1).expand_as(kept_indices)
        pos_embed = pos_embed_full_batch[batch_indices, kept_indices_clean]

        return pos_embed * valid_mask.unsqueeze(-1)

    @staticmethod
    def _scatter_tokens_to_feature_map(memory: torch.Tensor,
                                       kept_indices: torch.Tensor,
                                       h: int,
                                       w: int,
                                       hidden_dim: int) -> torch.Tensor:
        """Scatter pruned tokens back to full spatial grid."""
        B = memory.shape[0]
        total_tokens = h * w
        memory_2d_flat = torch.zeros(
            B, total_tokens, hidden_dim,
            device=memory.device, dtype=memory.dtype
        )

        if kept_indices is not None:
            kept_indices = HybridEncoder._normalize_kept_indices(kept_indices, B)
            valid_mask = (kept_indices >= 0) & (kept_indices < total_tokens)
            kept_indices_clean = kept_indices.clamp(0, total_tokens - 1)

            if B == 1:
                valid_indices = kept_indices_clean.view(-1)[valid_mask.view(-1)]
                valid_features = memory.view(-1, hidden_dim)[valid_mask.view(-1)]
            else:
                batch_offsets = torch.arange(B, device=memory.device).view(B, 1) * total_tokens
                global_indices = (kept_indices_clean + batch_offsets).view(-1)
                valid_mask_flat = valid_mask.view(-1)
                valid_indices = global_indices[valid_mask_flat]
                valid_features = memory.view(-1, hidden_dim)[valid_mask_flat]

            memory_2d_flat.view(-1, hidden_dim).index_copy_(0, valid_indices, valid_features)
        else:
            if memory.shape[1] == total_tokens:
                memory_2d_flat = memory
            else:
                memory_2d_flat[:, :memory.shape[1]] = memory

        return memory_2d_flat

    @staticmethod
    def _count_kept_tokens_per_level(
        kept_indices: torch.Tensor, level_sizes: List[int]
    ) -> List[int]:
        """
        kept_indices: [B, K] 拼接后序列上的全局下标；与 forward 里先 S4 段再 S5 段的顺序一致。
        返回 batch 0 上各 level 保留个数；level i 对应 ``use_encoder_idx[i]`` 与配置中 backbone 下标。
        """
        if kept_indices is None or not level_sizes:
            return []
        idx = kept_indices[0].long()
        starts = [0]
        for s in level_sizes:
            starts.append(starts[-1] + int(s))
        counts: List[int] = []
        for i in range(len(level_sizes)):
            lo, hi = starts[i], starts[i + 1]
            counts.append(int(((idx >= lo) & (idx < hi)).sum().item()))
        return counts

    @staticmethod
    def _count_kept_tokens_per_level_all(
        kept_indices: torch.Tensor, level_sizes: List[int]
    ) -> torch.Tensor:
        """
        kept_indices: [B, K] 与 ``_count_kept_tokens_per_level`` 相同索引约定。
        返回形状 [B, L]，每行对应一张图在各 level 上保留的 token 数。
        """
        if kept_indices is None or not level_sizes:
            return torch.zeros(0, 0, device=kept_indices.device, dtype=torch.float32)
        idx = kept_indices.long()
        starts = [0]
        for s in level_sizes:
            starts.append(starts[-1] + int(s))
        counts: List[torch.Tensor] = []
        for i in range(len(level_sizes)):
            lo, hi = starts[i], starts[i + 1]
            in_level = (idx >= lo) & (idx < hi)
            counts.append(in_level.sum(dim=1).float())
        return torch.stack(counts, dim=1)

    def _reset_prune_aggregate_stats(self) -> None:
        self._prune_agg_sum_kept = None
        self._prune_agg_n_images = 0
        self._prune_agg_level_sizes = None
        self._prune_agg_level_names = None

    def finalize_prune_level_aggregate_epoch(self, epoch: int) -> None:
        """
        在训练循环每个 epoch 结束处调用。按环境变量 ``CAS_PRUNE_LEVEL_EPOCH_AGGREGATE_EVERY``
        默认每 10 个 epoch 汇总本 epoch 内全部训练 batch、全部图像上的各层平均保留数并打日志。
        设为 0 则关闭本汇总，并清空缓冲区。
        """
        interval = int(os.environ.get("CAS_PRUNE_LEVEL_EPOCH_AGGREGATE_EVERY", "10"))
        if interval <= 0:
            self._reset_prune_aggregate_stats()
            return
        if self._prune_agg_n_images == 0 or self._prune_agg_sum_kept is None:
            self._reset_prune_aggregate_stats()
            return
        if (epoch + 1) % interval != 0:
            self._reset_prune_aggregate_stats()
            return
        n = float(self._prune_agg_n_images)
        means = (self._prune_agg_sum_kept / n).tolist()
        level_sizes = self._prune_agg_level_sizes or []
        names = self._prune_agg_level_names or []
        parts = []
        for name, m, lv in zip(names, means, level_sizes):
            pct = 100.0 * m / max(float(lv), 1.0)
            parts.append(f"{name} mean_kept={m:.1f}/{lv} ({pct:.1f}%)")
        total_in = float(sum(level_sizes)) if level_sizes else 1.0
        mean_total = float(sum(means))
        _LOGGER.info(
            "[TokenPruning] epoch_agg epoch=%d images=%d | total mean_kept=%.1f/%d (%.1f%%) | %s",
            epoch,
            self._prune_agg_n_images,
            mean_total,
            int(total_in),
            100.0 * mean_total / max(total_in, 1.0),
            " | ".join(parts),
        )
        self._reset_prune_aggregate_stats()

    def set_epoch(self, epoch: int):
        """设置当前epoch"""
        if self.shared_token_pruner is not None:
            self.shared_token_pruner.set_epoch(epoch)

    def forward(self, feats, return_encoder_info=False):
        assert len(feats) == len(self.in_channels)
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        
        encoder_info = {
            'token_pruning_ratios': [],
            'importance_scores_list': [],
            'feat_shapes_list': [],  # Store feature map shapes for CASS
        }
        
        if self.num_encoder_layers > 0 and self.use_encoder_idx:
            src_flatten_list = []
            pos_embed_list = []
            spatial_shapes = []
            level_sizes = []

            for idx_level, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                spatial_shapes.append((h, w))
                level_sizes.append(h * w)
                feat_flat = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)  # [B, H*W, C]
                if self.level_embed is not None:
                    feat_flat = feat_flat + self.level_embed[idx_level].view(1, 1, -1)
                src_flatten_list.append(feat_flat)

                pos_embed_full = self.build_2d_sincos_position_embedding(
                    w, h, self.hidden_dim, self.pe_temperature
                ).to(proj_feats[enc_ind].device).squeeze(0)  # [H*W, C]
                pos_embed_list.append(pos_embed_full.unsqueeze(0))

            src_flatten_total = torch.cat(src_flatten_list, dim=1)
            pos_embed_total = torch.cat(pos_embed_list, dim=1)

            if self.shared_token_pruner is not None:
                # --- CAIP branch: global-context-aware scoring + dynamic ratio ---
                caip_scores = None
                dynamic_keep_ratio = None
                scene_complexity = None

                if self.caip_predictor is not None:
                    caip_scores, scene_complexity = self.caip_predictor(src_flatten_total)
                    # Dynamic keep-ratio: higher complexity ⇒ retain more tokens
                    complexity_norm = torch.sigmoid(scene_complexity).item()
                    base_ratio = self.shared_token_pruner.keep_ratio
                    dynamic_keep_ratio = base_ratio + (1.0 - base_ratio) * complexity_norm * self.caip_complexity_alpha
                    dynamic_keep_ratio = float(min(max(dynamic_keep_ratio, base_ratio), 1.0))
                    encoder_info['scene_complexity'] = scene_complexity

                # Global pruning across all levels
                src_pruned, kept_indices, prune_info = self.shared_token_pruner(
                    src_flatten_total,
                    spatial_shape=None,
                    return_indices=True,
                    external_scores=caip_scores,
                    dynamic_keep_ratio=dynamic_keep_ratio,
                )
                encoder_info['token_pruning_ratios'].append(prune_info.get('pruning_ratio', 0.0))

                if level_sizes and kept_indices is not None:
                    kept_per_level = self._count_kept_tokens_per_level(
                        kept_indices, level_sizes
                    )
                    encoder_info['num_kept_tokens'] = int(kept_indices.shape[1])
                    encoder_info['num_input_tokens'] = int(src_flatten_total.shape[1])
                    encoder_info['kept_tokens_per_level'] = kept_per_level
                    encoder_info['kept_tokens_level_enc_indices'] = list(
                        self.use_encoder_idx
                    )
                    # Backbone stage labels: feat index 1 -> S4, 2 -> S5, same as train.py level_names
                    level_display_names = [
                        f"S{int(enc_ind) + 3}" for enc_ind in self.use_encoder_idx
                    ]
                    encoder_info['level_display_names'] = level_display_names
                    num_kept_total = int(
                        prune_info.get('num_kept_tokens', kept_indices.shape[1])
                    )
                    num_input_total = int(
                        prune_info.get('num_tokens', src_flatten_total.shape[1])
                    )
                    did_prune = num_kept_total < num_input_total
                    agg_interval = int(
                        os.environ.get("CAS_PRUNE_LEVEL_EPOCH_AGGREGATE_EVERY", "10")
                    )
                    if "CAS_LOG_PRUNE_LEVEL_EVERY" in os.environ:
                        log_every = int(os.environ["CAS_LOG_PRUNE_LEVEL_EVERY"])
                    else:
                        log_every = 0 if agg_interval > 0 else 1
                    if log_every < 0:
                        log_every = 0
                    self._prune_level_dist_log_counter = getattr(
                        self, "_prune_level_dist_log_counter", 0
                    ) + 1
                    if did_prune and log_every > 0 and (
                        self._prune_level_dist_log_counter % log_every == 1
                    ):
                        parts = ", ".join(
                            f"{name} tokens kept={c}/{lv} ({100.0 * c / max(lv, 1):.1f}%)"
                            for name, c, lv in zip(
                                level_display_names, kept_per_level, level_sizes
                            )
                        )
                        _LOGGER.info(
                            "[TokenPruning] batch_0 total kept=%d/%d | %s",
                            num_kept_total,
                            num_input_total,
                            parts,
                        )
                    if self.training:
                        counts_all = self._count_kept_tokens_per_level_all(
                            kept_indices, level_sizes
                        )
                        sum_b = counts_all.sum(dim=0).detach().cpu().float()
                        if self._prune_agg_sum_kept is None:
                            self._prune_agg_sum_kept = sum_b
                            self._prune_agg_level_sizes = list(level_sizes)
                            self._prune_agg_level_names = list(level_display_names)
                        else:
                            self._prune_agg_sum_kept = (
                                self._prune_agg_sum_kept + sum_b
                            )
                        self._prune_agg_n_images += int(kept_indices.shape[0])

                if 'token_importance_scores' in prune_info and prune_info['token_importance_scores'] is not None:
                    global_scores = prune_info['token_importance_scores']
                    encoder_info['importance_scores_list'].append(global_scores)
                    encoder_info['feat_shapes_list'].append(spatial_shapes)

                    if level_sizes and not self.training:
                        scores_per_level = torch.split(global_scores, level_sizes, dim=1)
                        heatmaps = []
                        for scores, (h, w) in zip(scores_per_level, spatial_shapes):
                            heatmaps.append(scores.view(scores.shape[0], 1, h, w))
                        encoder_info['layer_wise_heatmaps'] = heatmaps
            else:
                src_pruned = src_flatten_total
                kept_indices = None
                encoder_info['token_pruning_ratios'].append(0.0)

            if level_sizes:
                level_sizes_tensor = torch.as_tensor(
                    level_sizes, device=src_flatten_total.device
                )
                level_start_index = torch.cat([
                    level_sizes_tensor.new_zeros(1),
                    level_sizes_tensor.cumsum(0)[:-1]
                ])
                encoder_info['spatial_shapes'] = spatial_shapes
                encoder_info['level_start_index'] = level_start_index

            # Gather positional embeddings for kept tokens
            total_tokens = src_flatten_total.shape[1]
            pos_embed_pruned = self._gather_pos_embed(
                pos_embed_total.squeeze(0), kept_indices, src_pruned.shape[0], total_tokens
            )

            memory: torch.Tensor = self.encoder(
                src_pruned,
                pos_embed=pos_embed_pruned,
            )

            # Scatter back into full sequence
            B = memory.shape[0]
            total_tokens = src_flatten_total.shape[1]
            memory_full = torch.zeros(
                B, total_tokens, self.hidden_dim,
                device=memory.device, dtype=memory.dtype
            )
            if kept_indices is not None:
                kept_indices = self._normalize_kept_indices(kept_indices, B)
                valid_mask = (kept_indices >= 0) & (kept_indices < total_tokens)
                kept_indices_clean = kept_indices.clamp(0, total_tokens - 1)

                if B == 1:
                    valid_indices = kept_indices_clean.view(-1)[valid_mask.view(-1)]
                    valid_features = memory.view(-1, self.hidden_dim)[valid_mask.view(-1)]
                else:
                    batch_offsets = torch.arange(B, device=memory.device).view(B, 1) * total_tokens
                    global_indices = (kept_indices_clean + batch_offsets).view(-1)
                    valid_mask_flat = valid_mask.view(-1)
                    valid_indices = global_indices[valid_mask_flat]
                    valid_features = memory.view(-1, self.hidden_dim)[valid_mask_flat]

                memory_full.view(-1, self.hidden_dim).index_copy_(0, valid_indices, valid_features)
            else:
                if memory.shape[1] == total_tokens:
                    memory_full = memory
                else:
                    memory_full[:, :memory.shape[1]] = memory

            memory_splits = torch.split(memory_full, level_sizes, dim=1)
            for idx, enc_ind in enumerate(self.use_encoder_idx):
                h, w = spatial_shapes[idx]
                memory_level = memory_splits[idx]
                # Verify shape consistency: memory_level should be [B, h*w, C]
                expected_tokens = h * w
                if memory_level.shape[1] != expected_tokens:
                    raise ValueError(
                        f"Shape mismatch in level {idx}: memory_level.shape[1]={memory_level.shape[1]}, "
                        f"expected h*w={expected_tokens} (h={h}, w={w})"
                    )
                memory_2d = memory_level.permute(0, 2, 1).reshape(
                    memory_level.shape[0], self.hidden_dim, h, w
                ).contiguous()
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

        if self.detail_branch is not None:
            outs[0] = self.detail_branch(outs[0])

        if return_encoder_info:
            return outs, encoder_info
        
        # 即使 return_encoder_info=False，为了可视化也通过 hack 方式挂载
        # 这是一个简单的 hack，用于在 inference mode 下让外部 hook 能够访问 encoder_info
        if not self.training:
            # 将 encoder_info 附加到输出张量列表的第一个元素上作为属性
            if hasattr(outs, '__len__') and len(outs) > 0 and isinstance(outs[0], torch.Tensor):
                setattr(outs[0], 'encoder_info', encoder_info)
        
        return outs
