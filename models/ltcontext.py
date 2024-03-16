import math
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from einops import repeat, rearrange
from functools import partial
from typing import Optional, Union, Tuple
from dataclasses import dataclass
from torch.nn.init import constant_


def exists(val):
    return val is not None


def patchify(ts: Tensor, patch_size: int, stride: int):
    patches = ts.unfold(2, patch_size, stride)  # [bs, d, nw, patch_size]
    return patches


def _matmul_with_mask(a: torch.Tensor, b: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return a @ b
    att = a @ b
    if mask.dtype == torch.bool:
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).expand(att.shape[0], -1, -1)
        att[~mask] = float("-inf")
    else:
        raise ValueError("Attention Mask need to be boolean with False to ignore")

    return att


def scaled_query_key_mult(q: torch.Tensor, k: torch.Tensor, att_mask: Optional[torch.Tensor]) -> torch.Tensor:
    q = q / math.sqrt(k.size(-1))
    att = _matmul_with_mask(q, k.transpose(-2, -1), att_mask)

    return att


def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, att_mask: Optional[torch.Tensor], 
                                 position_bias: Optional[torch.Tensor] = None, dropout: Optional[torch.nn.Module] = None) -> Union[Tuple[torch.Tensor, torch.Tensor]]:
    att = scaled_query_key_mult(q, k, att_mask=att_mask)

    if position_bias is not None:
        att += position_bias

    out_mask = torch.ones_like(q)
    if (att_mask.sum((-2, -1)) == 0).any():
        att[(att_mask.sum((-2, -1)) == 0)] = 0.00001
        out_mask[(att_mask.sum((-2, -1)) == 0), :, :] = 0

    att = torch.softmax(att, dim=att.ndim - 1)
    att = dropout(att)
    y = att @ v
    y = y * out_mask
    return y, att


def convert_to_patches(x: Tensor, patch_size: int, masks: Tensor = None, overlapping_patches: bool = False):
    _, _, seq_len = x.shape
    if seq_len % patch_size != 0:
        pad_size = patch_size - (seq_len % patch_size)
    else:
        pad_size = 0
    x, padded_masks = pad_sequence(x, pad_size, masks)

    if overlapping_patches:
        half_pad = (patch_size//2, patch_size//2)
        padded_x, padding_mask = pad_sequence(x, pad_size=half_pad, masks=padded_masks)
        patches = patchify(padded_x, 2*patch_size, stride=patch_size)
        padding_mask = patchify(padding_mask, 2*patch_size, stride=patch_size)
    else:
        patches = patchify(x, patch_size, stride=patch_size)
        padding_mask = patchify(padded_masks, patch_size, stride=patch_size)
    return patches, padding_mask


def pad_sequence(x: Tensor, pad_size: Union[int, Tuple[int, int]], masks: Tensor = None):
    bs, _, seq_len = x.shape

    if isinstance(pad_size, int):
        pad_size = (0, pad_size)
    if not exists(masks):
        masks = torch.ones(bs, 1, seq_len).bool()
        masks = masks.to(x.device)
    if pad_size[-1] <= 0:
        return x, masks
    padded_x = F.pad(x, pad=pad_size, value=0.0)
    padding_mask = F.pad(masks, pad=pad_size, value=False)
    return padded_x, padding_mask


def build_attention(attention_params):
    if attention_params.type == 'full':
        return ScaledDotProduct(dropout=attention_params.dropout)
    else:
        raise ModuleNotFoundError(f"Attention with name {attention_params.type} is not found!")


class ScaledDotProduct(nn.Module):
    """
    This code is inspired from xformers lib
    """
    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.attn_drop = nn.Dropout(dropout, inplace=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, att_mask: Optional[torch.Tensor] = None, 
                position_bias: Optional[torch.Tensor] = None, return_attn_matrix: bool = False) -> Union[Tuple[torch.Tensor, torch.Tensor]]:
        y, att_matrix = scaled_dot_product_attention(q=q, k=k, v=v, att_mask=att_mask, dropout=self.attn_drop, position_bias=position_bias)
        if return_attn_matrix:
            return y, att_matrix
        else:
            return y


@dataclass
class AttentionParams:
    num_heads: int
    type: str = 'full'
    bias: bool = True
    dropout: float = 0.0
    use_separate_proj_weight: bool = True
    requires_input_projection: bool = True


class MultiHeadAttention(nn.Module):
    def __init__(self, dim_model: int, attention_params: AttentionParams, use_conv1d_proj: bool = False, 
                 rel_pos_encoder: nn.Module = None, dim_key: Optional[int] = None, dim_value: Optional[int] = None, out_proj: Optional[nn.Module] = None):
        super().__init__()
        dim_key, dim_value = map(lambda x: x if x else dim_model, (dim_key, dim_value))

        self.rel_pos_encoder = rel_pos_encoder
        self.num_heads = attention_params.num_heads
        self.dim_k = dim_key // self.num_heads
        self.dim_value = dim_value
        self.dim_model = dim_model
        self.attention = build_attention(attention_params)
        self.requires_input_projection = attention_params.requires_input_projection
        self.use_conv1d_proj = use_conv1d_proj

        if use_conv1d_proj:
            LinearProj = partial(nn.Conv1d, bias=attention_params.bias, kernel_size=1)
        else:
            LinearProj = partial(nn.Linear, bias=attention_params.bias)

        if attention_params.use_separate_proj_weight:
            self.proj_q = LinearProj(dim_model, dim_key)
            self.proj_k = LinearProj(dim_model, dim_key)
            self.proj_v = LinearProj(dim_model, dim_value)
        else:
            assert dim_key == dim_value, "To share qkv projection " \
                                         "weights dimension of q, k, v should be the same"
            self.proj_q = LinearProj(dim_model, dim_key)
            self.proj_v, self.proj_k = self.proj_q, self.proj_q

        self.dropout = nn.Dropout(0.1, inplace=False)
        self.proj = out_proj if out_proj else LinearProj(dim_value, dim_value)
        if isinstance(self.proj, nn.Linear) and self.proj.bias is not None:
            constant_(self.proj.bias, 0.0)

    def _check(self, t, name):
        if self.use_conv1d_proj:
            d = t.shape[1]
        else:
            d = t.shape[2]
        assert (
            d % self.dim_k == 0
        ), f"the {name} embeddings need to be divisible by the number of heads"

    def _split_heads(self, tensor):
        assert len(tensor.shape) == 3, "Invalid shape for splitting heads"

        batch_size, seq_len = tensor.shape[0], tensor.shape[1]
        embed_dim = tensor.shape[2]

        new_embed_dim = embed_dim // self.num_heads

        tensor = torch.reshape(tensor, (batch_size, seq_len, self.num_heads, new_embed_dim))

        tensor = torch.transpose(tensor, 1, 2).flatten(start_dim=0, end_dim=1).contiguous()
        return tensor

    def _combine_heads(self, tensor, batch_size):
        assert len(tensor.shape) == 3, "Invalid shape to combine heads"

        tensor = tensor.unflatten(0, (batch_size, self.num_heads))
        tensor = torch.transpose(tensor, 1, 2)
        seq_len = tensor.shape[1]
        embed_dim = tensor.shape[-1]

        new_embed_dim = self.num_heads * embed_dim
        tensor = torch.reshape(tensor, (batch_size, seq_len, new_embed_dim)).contiguous()
        return tensor

    def forward(self, query: torch.Tensor, key: Optional[torch.Tensor] = None, value: Optional[torch.Tensor] = None, 
                att_mask: Optional[torch.Tensor] = None):
        if key is None:
            key = query
        if value is None:
            value = query

        self._check(query, "query")
        self._check(value, "value")
        self._check(key, "key")

        bs, _, q_len = query.size()
        _, _, k_len = key.size()

        if self.requires_input_projection:
            q, k, v = self.proj_q(query),  self.proj_k(key), self.proj_v(value)
        else:
            k, q, v = key, query, value

        if self.use_conv1d_proj:
            q = rearrange(q, 'b d l -> b l d')
            k = rearrange(k, 'b d l -> b l d')
            v = rearrange(v, 'b d l -> b l d')

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        position_bias = None
        if isinstance(self.rel_pos_encoder, nn.Module):
            position_bias = self.rel_pos_encoder(q, k)

        if att_mask is not None:
            att_mask = repeat(att_mask, "b 1 k_len -> b q_len k_len", q_len=q_len, k_len=k_len)
            att_mask = repeat(att_mask, 'b l1 l2 -> (b n_heads) l1 l2', n_heads=self.num_heads)

        z, _ = self.attention(q=q, k=k, v=v, att_mask=att_mask, return_attn_matrix=True, position_bias=position_bias)

        z = self._combine_heads(z, bs)
        if self.use_conv1d_proj:
            z = rearrange(z, 'b l d -> b d l')

        output = self.dropout(self.proj(z))
        return output


class BaseAttention(nn.Module):
    def __init__(self, model_dim: int, attn_params: AttentionParams):
        super().__init__()
        out_proj = nn.Sequential(nn.GELU(), nn.Conv1d(model_dim // 2, model_dim, kernel_size=1, bias=True))

        self.attn = MultiHeadAttention(model_dim,
                                       attn_params,
                                       out_proj=out_proj,
                                       use_conv1d_proj=True,
                                       dim_key=model_dim // 2,
                                       dim_value=model_dim // 2)


class WindowedAttention(BaseAttention):
    def __init__(self, windowed_attn_w: int, model_dim: int, attn_params: AttentionParams):
        super().__init__(model_dim, attn_params)
        self.windowed_attn_w = windowed_attn_w

    def _reshape(self, x: Tensor, overlapping_patches: bool, masks: Tensor = None):
        patches, masks = convert_to_patches(x, self.windowed_attn_w, masks, overlapping_patches)
        patches = rearrange(patches, "b d num_patches patch_size -> (b num_patches) d patch_size")
        masks = rearrange(masks,   "b d num_patches patch_size -> (b num_patches) d patch_size")
        return patches.contiguous(), masks.contiguous()

    def _undo_reshape(self, patches: Tensor, batch_size: int, orig_seq_len: int):
        num_patches = patches.shape[0] // batch_size
        x = rearrange(patches,
                      "(b num_patches) d patch_size -> b d (num_patches patch_size)",
                      num_patches=num_patches,
                      b=batch_size)

        x = x[:, :, :orig_seq_len]
        return x.contiguous()

    def _prep_qkv(self, qk: Tensor, v: Tensor, masks: Tensor):
        q, k = qk, qk
        q, _ = self._reshape(q, overlapping_patches=False)
        k, attn_mask = self._reshape(k, overlapping_patches=True, masks=masks)
        if exists(v):
            v, _ = self._reshape(v, overlapping_patches=True)
        else:
            v = k

        return q, k, v, attn_mask

    def forward(self, qk: Tensor, v: Tensor = None, masks: Tensor = None):
        batch_size, _, seq_len = qk.shape
        q, k, v, att_mask = self._prep_qkv(qk=qk, v=v, masks=masks)
        windowed_attn = self.attn(query=q, key=k, value=v, att_mask=att_mask)
        out = self._undo_reshape(windowed_attn, batch_size, seq_len)
        if exists(masks):
            out = out * masks
        return out


class LTContextAttention(BaseAttention):
    def __init__(self, long_term_attn_g, model_dim, attn_params):
        super().__init__(model_dim, attn_params)
        self.long_term_attn_g = long_term_attn_g

    def _reshape(self, x: Tensor, masks: Tensor = None):
        patches, masks = convert_to_patches(x, self.long_term_attn_g, masks)

        lt_patches = rearrange(patches, "b d num_patches patch_size -> (b patch_size) d num_patches")
        masks = rearrange(masks, "b d num_patches patch_size -> (b patch_size) d num_patches")

        return lt_patches.contiguous(), masks.contiguous()

    def _undo_reshape(self, lt_patches, batch_size, orig_seq_len):
        x = rearrange(lt_patches,
                      "(b patch_size) d num_patches -> b d (num_patches patch_size)",
                      patch_size=self.long_term_attn_g,
                      b=batch_size)
        x = x[:, :, :orig_seq_len]
        return x.contiguous()

    def _prep_qkv(self, qk: Tensor, v: Tensor, masks: Tensor):
        q, k = qk, qk
        q, _ = self._reshape(q)
        k, attn_mask = self._reshape(k, masks=masks)
        if exists(v):
            v, _ = self._reshape(v)
        else:
            v = k

        return q, k, v, attn_mask

    def forward(self, qk: Tensor, v: Tensor = None, masks: Tensor = None):
        batch_size, _, seq_len = qk.shape
        q, k, v, att_mask = self._prep_qkv(qk=qk, v=v, masks=masks)
        lt_attn = self.attn(query=q, key=k, value=v, att_mask=att_mask)
        out = self._undo_reshape(lt_attn, batch_size, seq_len)
        if exists(masks):
            out = out * masks
        return out


class DilatedConv(nn.Module):
    def __init__(self, n_channels: int, dilation: int, kernel_size: int = 3):
        super().__init__()
        self.dilated_conv = nn.Conv1d(n_channels, n_channels, kernel_size=kernel_size, padding=dilation, dilation=dilation)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(self.dilated_conv(x))


class LTCBlock(nn.Module):
    def __init__(self, model_dim: int, dilation: int, windowed_attn_w: int, long_term_attn_g: int, 
                 attn_params: AttentionParams, use_instance_norm: bool, dropout_prob: float):
        super(LTCBlock, self).__init__()
        self.dilated_conv = DilatedConv(n_channels=model_dim, dilation=dilation, kernel_size=3)

        if use_instance_norm:
            self.instance_norm = nn.Identity()
        else:
            self.instance_norm = nn.InstanceNorm1d(model_dim)

        self.windowed_attn = WindowedAttention(windowed_attn_w=windowed_attn_w, model_dim=model_dim, attn_params=attn_params)
        self.ltc_attn = LTContextAttention(long_term_attn_g=long_term_attn_g, model_dim=model_dim, attn_params=attn_params)

        self.out_linear = nn.Conv1d(model_dim, model_dim, kernel_size=1, bias=True)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, inputs: Tensor, prev_stage_feat: Tensor = None):
        out = self.dilated_conv(inputs)
        out = self.windowed_attn(self.instance_norm(out), prev_stage_feat) + out
        out = self.ltc_attn(self.instance_norm(out), prev_stage_feat) + out
        out = self.out_linear(out)
        out = self.dropout(out)
        out = out + inputs
        return out


class LTCModule(nn.Module):
    def __init__(self, num_layers: int, input_dim: int, model_dim: int, num_classes: int, attn_params: AttentionParams,  dilation_factor: int, 
                 windowed_attn_w: int, long_term_attn_g: int, use_instance_norm: bool, dropout_prob: float, channel_dropout_prob: float):
        super(LTCModule, self).__init__()
        self.channel_dropout = nn.Dropout1d(channel_dropout_prob)
        self.input_proj = nn.Conv1d(input_dim, model_dim, kernel_size=1, bias=True)
        self.layers = nn.ModuleList([])
        for i in range(num_layers):
            self.layers.append(
                LTCBlock(
                    model_dim=model_dim,
                    dilation=dilation_factor**i,
                    windowed_attn_w=windowed_attn_w,
                    long_term_attn_g=long_term_attn_g,
                    attn_params=attn_params,
                    use_instance_norm=use_instance_norm,
                    dropout_prob=dropout_prob
                    )
            )
        self.out_proj = nn.Conv1d(model_dim, num_classes, kernel_size=1, bias=True)

    def forward(self, inputs: Tensor, prev_stage_feat: Tensor = None):
        inputs = self.channel_dropout(inputs)
        feature = self.input_proj(inputs)
        for layer in self.layers:
            feature = layer(feature, prev_stage_feat)
        out = self.out_proj(feature)
        return out, feature


class LTC(nn.Module):
    def __init__(self, input_dim=512):
        super(LTC, self).__init__()

        attn_params = AttentionParams(num_heads=1, dropout=0.2)

        self.stage1 = LTCModule(num_layers=9,
                                input_dim=input_dim,
                                model_dim=64,
                                num_classes=5,
                                attn_params=attn_params,
                                dilation_factor=2,
                                windowed_attn_w=64,
                                long_term_attn_g=64,
                                use_instance_norm=True,
                                dropout_prob=0.2,
                                channel_dropout_prob=0.3
                                )

        reduced_dim = int(64 // 2.0)
        self.dim_reduction = nn.Conv1d(64, reduced_dim, kernel_size=1, bias=True)
        self.stages = nn.ModuleList([])
        for s in range(1, 3):
            self.stages.append(
                LTCModule(num_layers=9,
                          input_dim=5,
                          model_dim=reduced_dim,
                          num_classes=5,
                          attn_params=attn_params,
                          dilation_factor=2,
                          windowed_attn_w=64,
                          long_term_attn_g=64,
                          use_instance_norm=True,
                          dropout_prob=0.2,
                          channel_dropout_prob=0.3
                          )
            )

    def forward(self, inputs: Tensor) -> Tensor:
        out, feature = self.stage1(inputs)
        feature_list = [feature]
        output_list = [out]
        feature = self.dim_reduction(feature)
        for stage in self.stages:
            out, feature = stage(F.softmax(out, dim=1),
                                 prev_stage_feat=feature,
                                 )
            output_list.append(out)
            feature_list.append(feature)
        logits = torch.stack(output_list, dim=0)
        return logits