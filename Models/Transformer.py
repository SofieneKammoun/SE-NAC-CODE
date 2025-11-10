#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 17:02:21 2025

@author: root
"""

import torch
from torch import nn, einsum
from einops import rearrange 


def FeedForward(*, dim, mult = 4, dropout = 0.):
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, dim * mult),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(dim * mult, dim)
    )
class Attention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        causal = True 
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.causal = causal
        inner_dim = dim_head * heads

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        h, device = self.heads, x.device

        x = self.norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        if self.causal:
            i, j = sim.shape[-2:]
            mask_value = -torch.finfo(sim.dtype).max
            mask = torch.ones((i, j), dtype=torch.bool, device=device).triu(1)
            sim = sim.masked_fill(mask, mask_value)

        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        layers,
        dim_head = 64,
        heads = 8,
        attn_dropout = 0.,
        ff_dropout = 0.,
        ff_mult = 4,
        causal = True   
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(layers):
            self.layers.append(nn.ModuleList([
                Attention(
                    dim = dim,
                    dim_head = dim_head,
                    heads = heads,
                    dropout = attn_dropout,
                    causal = causal
                ),
                FeedForward(
                    dim = dim,
                    mult = ff_mult,
                    dropout = ff_dropout
                )
            ]))

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)