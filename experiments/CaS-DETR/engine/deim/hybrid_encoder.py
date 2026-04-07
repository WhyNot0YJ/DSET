"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE/)
Copyright (c) 2024 D-FINE Authors. All Rights Reserved.
"""

import copy
from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import get_activation
from .token_level_pruning import TokenLevelPruner

from ..core import register

__all__ = ['HybridEncoder']


class ConvNormLayer_fuse(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, g=1, padding=None, bias=False, act=None):
        super().__init__()
        padding = (kernel_size-1)//2 if padding is None else padding
        self.conv = nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size,
            stride,
            groups=g,
            padding=padding,
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)
        self.ch_in, self.ch_out, self.kernel_size, self.stride, self.g, self.padding, self.bias = \
            ch_in, ch_out, kernel_size, stride, g, padding, bias

    def forward(self, x):
        if hasattr(self, 'conv_bn_fused'):
            y = self.conv_bn_fused(x)
        else:
            y = self.norm(self.conv(x))
        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, 'conv_bn_fused'):
            self.conv_bn_fused = nn.Conv2d(
                self.ch_in,
                self.ch_out,
                self.kernel_size,
                self.stride,
                groups=self.g,
                padding=self.padding,
                bias=True)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv_bn_fused.weight.data = kernel
        self.conv_bn_fused.bias.data = bias
        self.__delattr__('conv')
        self.__delattr__('norm')

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor()

        return kernel3x3, bias3x3

    def _fuse_bn_tensor(self):
        kernel = self.conv.weight
        running_mean = self.norm.running_mean
        running_var = self.norm.running_var
        gamma = self.norm.weight
        beta = self.norm.bias
        eps = self.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, g=1, padding=None, bias=False, act=None):
        super().__init__()
        padding = (kernel_size-1)//2 if padding is None else padding
        self.conv = nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size,
            stride,
            groups=g,
            padding=padding,
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


# TODO, add activation for cv1 following YOLOv10
# self.cv1 = Conv(c1, c2, 1, 1)
# self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)
class SCDown(nn.Module):
    def __init__(self, c1, c2, k, s, act=None):
        super().__init__()
        self.cv1 = ConvNormLayer_fuse(c1, c2, 1, 1)
        self.cv2 = ConvNormLayer_fuse(c2, c2, k, s, c2)

    def forward(self, x):
        return self.cv2(self.cv1(x))


class VGGBlock(nn.Module):
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
        self.__delattr__('conv1')
        self.__delattr__('conv2')

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


class CSPLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 expansion=1.0,
                 bias=False,
                 act="silu",
                 bottletype=VGGBlock):
        super(CSPLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayer_fuse(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.conv2 = ConvNormLayer_fuse(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(*[
            bottletype(hidden_channels, hidden_channels, act=act) for _ in range(num_blocks)
        ])
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer_fuse(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_2 = self.conv2(x)
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        return self.conv3(x_1 + x_2)

class RepNCSPELAN4(nn.Module):
    # csp-elan
    def __init__(self, c1, c2, c3, c4, n=3,
                 bias=False,
                 act="silu"):
        super().__init__()
        self.c = c3//2
        self.cv1 = ConvNormLayer_fuse(c1, c3, 1, 1, bias=bias, act=act)
        self.cv2 = nn.Sequential(CSPLayer(c3//2, c4, n, 1, bias=bias, act=act, bottletype=VGGBlock), ConvNormLayer_fuse(c4, c4, 3, 1, bias=bias, act=act))
        self.cv3 = nn.Sequential(CSPLayer(c4, c4, n, 1, bias=bias, act=act, bottletype=VGGBlock), ConvNormLayer_fuse(c4, c4, 3, 1, bias=bias, act=act))
        self.cv4 = ConvNormLayer_fuse(c3+(2*c4), c2, 1, 1, bias=bias, act=act)

    def forward_chunk(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


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


class CAIPPredictor(nn.Module):
    """Optional global-context scorer for token pruning."""

    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 reduction_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        reduced_dim = max(input_dim // reduction_ratio, 16)
        self.local_fc1 = nn.Linear(input_dim, hidden_dim)
        self.local_act = nn.GELU()
        self.local_dropout = nn.Dropout(dropout)
        self.local_fc2 = nn.Linear(hidden_dim, 1)

        self.global_fc1 = nn.Conv1d(input_dim, reduced_dim, kernel_size=1)
        self.global_act = nn.GELU()
        self.global_fc2 = nn.Conv1d(reduced_dim, hidden_dim, kernel_size=1)
        self._reset_parameters()

    def _reset_parameters(self):
        for m in (self.local_fc1, self.local_fc2):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
        for m in (self.global_fc1, self.global_fc2):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, tokens: torch.Tensor):
        local_feat = self.local_fc1(tokens)
        local_feat = self.local_act(local_feat)
        local_feat = self.local_dropout(local_feat)

        gap = tokens.mean(dim=1, keepdim=True).permute(0, 2, 1)
        global_feat = self.global_fc1(gap)
        global_feat = self.global_act(global_feat)
        pre_sigmoid = self.global_fc2(global_feat)
        global_weights = torch.sigmoid(pre_sigmoid).squeeze(-1).unsqueeze(1)

        importance_scores = self.local_fc2(local_feat * global_weights).squeeze(-1)
        return importance_scores


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
                 version='dfine',
                 token_keep_ratio=1.0,
                 enable_cas_predictor=False,
                 use_cass=False,
                 cass_expansion_ratio=0.3,
                 cass_min_size=1.0,
                 cass_decay_type='gaussian',
                 use_subpixel_offset=True,
                 cass_loss_type='vfl',
                 cass_focal_alpha=0.75,
                 cass_focal_beta=2.0,
                 use_caip=False,
                 caip_reduction_ratio=4,
                 caip_complexity_alpha=0.3,
                 ):
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
        self.enable_cas_predictor = enable_cas_predictor
        self.use_cass = use_cass and enable_cas_predictor
        self.use_caip = use_caip and enable_cas_predictor
        self.caip_complexity_alpha = caip_complexity_alpha

        # channel projection
        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            proj = nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False)),
                    ('norm', nn.BatchNorm2d(hidden_dim))
                ]))

            self.input_proj.append(proj)

        # encoder transformer
        encoder_layer = TransformerEncoderLayer(
            hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=enc_act
            )

        self.encoder = nn.ModuleList([
            TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers) for _ in range(len(use_encoder_idx))
        ])

        if self.enable_cas_predictor:
            self.shared_token_pruner = TokenLevelPruner(
                input_dim=hidden_dim,
                keep_ratio=token_keep_ratio,
                adaptive=True,
                min_tokens=1,
                prune_in_eval=True,
                use_cass=self.use_cass,
                cass_expansion_ratio=cass_expansion_ratio,
                cass_min_size=cass_min_size,
                cass_decay_type=cass_decay_type,
                use_subpixel_offset=use_subpixel_offset,
                cass_loss_type=cass_loss_type,
                cass_focal_alpha=cass_focal_alpha,
                cass_focal_beta=cass_focal_beta,
            )
        else:
            self.shared_token_pruner = None

        if self.use_caip:
            self.caip_predictor = CAIPPredictor(
                input_dim=hidden_dim,
                hidden_dim=128,
                reduction_ratio=caip_reduction_ratio,
                dropout=dropout if dropout > 0 else 0.1,
            )
        else:
            self.caip_predictor = None

        # top-down fpn
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1, 0, -1):
            # TODO, add activation for those lateral convs
            if version == 'dfine':
                self.lateral_convs.append(ConvNormLayer_fuse(hidden_dim, hidden_dim, 1, 1))
            else:
                self.lateral_convs.append(ConvNormLayer_fuse(hidden_dim, hidden_dim, 1, 1, act=act))
            self.fpn_blocks.append(
                RepNCSPELAN4(hidden_dim * 2, hidden_dim, hidden_dim * 2, round(expansion * hidden_dim // 2), round(3 * depth_mult), act=act) \
                if version == 'dfine' else CSPLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion, bottletype=VGGBlock)
            )

        # bottom-up pan
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            self.downsample_convs.append(
                nn.Sequential(SCDown(hidden_dim, hidden_dim, 3, 2, act=act)) \
                if version == 'dfine' else ConvNormLayer_fuse(hidden_dim, hidden_dim, 3, 2, act=act)
            )
            self.pan_blocks.append(
                RepNCSPELAN4(hidden_dim * 2, hidden_dim, hidden_dim * 2, round(expansion * hidden_dim // 2), round(3 * depth_mult), act=act) \
                if version == 'dfine' else CSPLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion, bottletype=VGGBlock)
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

    @staticmethod
    def _normalize_kept_indices(kept_indices: Optional[torch.Tensor], batch_size: int):
        if kept_indices is None:
            return None
        if kept_indices.dim() == 1:
            return kept_indices.unsqueeze(0).expand(batch_size, -1)
        if kept_indices.dim() == 2 and kept_indices.shape[0] == 1:
            return kept_indices.expand(batch_size, -1)
        return kept_indices

    @staticmethod
    def _gather_pos_embed(pos_embed_full: torch.Tensor,
                          kept_indices: Optional[torch.Tensor],
                          batch_size: int,
                          total_tokens: int) -> torch.Tensor:
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
    def _scatter_tokens_to_grid(memory: torch.Tensor,
                                kept_indices: Optional[torch.Tensor],
                                total_tokens: int,
                                hidden_dim: int) -> torch.Tensor:
        batch_size = memory.shape[0]
        memory_full = torch.zeros(
            batch_size, total_tokens, hidden_dim,
            device=memory.device, dtype=memory.dtype
        )
        if kept_indices is None:
            if memory.shape[1] == total_tokens:
                return memory
            memory_full[:, :memory.shape[1]] = memory
            return memory_full

        kept_indices = HybridEncoder._normalize_kept_indices(kept_indices, batch_size)
        valid_mask = (kept_indices >= 0) & (kept_indices < total_tokens)
        kept_indices_clean = kept_indices.clamp(0, total_tokens - 1)
        batch_offsets = torch.arange(batch_size, device=memory.device).view(batch_size, 1) * total_tokens
        global_indices = (kept_indices_clean + batch_offsets).reshape(-1)
        valid_mask_flat = valid_mask.reshape(-1)
        valid_indices = global_indices[valid_mask_flat]
        valid_features = memory.reshape(-1, hidden_dim)[valid_mask_flat]
        memory_full.reshape(-1, hidden_dim).index_copy_(0, valid_indices, valid_features)
        return memory_full

    def set_epoch(self, epoch: int):
        if self.shared_token_pruner is not None:
            self.shared_token_pruner.set_epoch(epoch)

    def forward(self, feats, return_encoder_info=False):
        assert len(feats) == len(self.in_channels)
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        encoder_info = {
            'token_pruning_ratios': [],
            'importance_scores_list': [],
            'feat_shapes_list': [],
        }

        # encoder
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                # flatten [B, C, H, W] to [B, HxW, C]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                total_tokens = src_flatten.shape[1]
                if self.training or self.eval_spatial_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature).to(src_flatten.device)
                else:
                    pos_embed = getattr(self, f'pos_embed{enc_ind}', None).to(src_flatten.device)
                pos_embed_full = pos_embed.squeeze(0)

                prune_info = {'pruning_ratio': 0.0, 'token_importance_scores': None}
                kept_indices = None
                external_scores = None
                dynamic_keep_ratio = None

                if self.caip_predictor is not None:
                    external_scores = self.caip_predictor(src_flatten)
                    base_ratio = float(self.shared_token_pruner.keep_ratio)
                    # Complexity proxy: per-image mean token confidence.
                    # Detach to avoid trivially inflating scores to disable pruning.
                    mean_conf = torch.sigmoid(external_scores.detach()).mean(dim=1)  # [B]
                    dynamic_keep_ratio = base_ratio + (1.0 - base_ratio) * mean_conf * self.caip_complexity_alpha
                    dynamic_keep_ratio = dynamic_keep_ratio.clamp(min=base_ratio, max=1.0)
                    encoder_info['dynamic_keep_ratio'] = dynamic_keep_ratio

                if self.shared_token_pruner is not None:
                    src_flatten, kept_indices, prune_info = self.shared_token_pruner(
                        src_flatten,
                        spatial_shape=(h, w),
                        return_indices=True,
                        external_scores=external_scores,
                        dynamic_keep_ratio=dynamic_keep_ratio,
                    )

                encoder_info['token_pruning_ratios'].append(prune_info.get('pruning_ratio', 0.0))
                encoder_info['importance_scores_list'].append(prune_info.get('token_importance_scores'))
                encoder_info['feat_shapes_list'].append((h, w))

                pos_embed_pruned = self._gather_pos_embed(
                    pos_embed_full, kept_indices, src_flatten.shape[0], total_tokens
                )
                memory :torch.Tensor = self.encoder[i](src_flatten, pos_embed=pos_embed_pruned)
                memory = self._scatter_tokens_to_grid(memory, kept_indices, total_tokens, self.hidden_dim)
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

        if not self.training and len(outs) > 0 and isinstance(outs[0], torch.Tensor):
            setattr(outs[0], 'encoder_info', encoder_info)
        return outs
