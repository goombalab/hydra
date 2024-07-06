# Copyright (c) 2024, Sukjun Hwang, Aakash Lahoti, Ratish Puduppully, Tri Dao, Albert Gu.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


class Attention(nn.Module):
    def __init__(
        self,
        is_data_dependent,
        d_model,
        qk_dim,
        max_seq_len=None,   # max_seq_len is necessary for data-independent version.
        expand=2,
        headdim=128,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.is_data_dependent = is_data_dependent
        self.d_model = d_model
        self.qk_dim = qk_dim
        self.max_seq_len = max_seq_len
        self.expand = expand
        self.d_inner = self.expand * self.d_model
        self.headdim = headdim
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        self.d_state = self.nheads * qk_dim

        if not self.is_data_dependent:
            self.q_matrix = nn.Parameter(
                torch.empty(self.max_seq_len, self.nheads, self.qk_dim, **factory_kwargs))
            self.k_matrix = nn.Parameter(
                torch.empty(self.max_seq_len, self.nheads, self.qk_dim, **factory_kwargs))
            nn.init.xavier_normal_(self.q_matrix)
            nn.init.xavier_normal_(self.k_matrix)

    def forward(self, v, q=None, k=None):
        residual = v
        v = rearrange(v, 'b l (n h) -> b l n h', n=self.nheads)

        if self.is_data_dependent:
            q = rearrange(q, 'b l (n d) -> b l n d', n=self.nheads)
            k = rearrange(k, 'b l (n d) -> b l n d', n=self.nheads)
            qk = torch.einsum('b t n d, b l n d -> b n t l', q, k)
            attn_weights = torch.softmax(1 / np.sqrt(self.qk_dim) * qk, dim=-1)
            output = torch.einsum('b n t l, b l n h -> b t n h', attn_weights, v)
        else:
            qk = torch.einsum('n t d, n l d -> n t l', self.q_matrix, self.k_matrix)
            attn_weights = torch.softmax(1 / np.sqrt(self.qk_dim) * qk, dim=-1)
            output = torch.einsum('n t l, b l n h -> b t n h', attn_weights, v)

        output = rearrange(output, 'b l n h -> b l (n h)') + residual

        return output
