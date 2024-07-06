# Copyright (c) 2024, Sukjun Hwang, Aakash Lahoti, Ratish Puduppully, Tri Dao, Albert Gu.

import numpy as np
import torch
import torch.nn as nn

from einops import rearrange


class Dense(nn.Module):
    def __init__(
        self, 
        d_model,
        max_seq_len, # max_seq_len is necessary for Dense.
        expand=2,
        headdim=128,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.expand = expand
        self.d_inner = self.expand * self.d_model
        self.headdim = headdim
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // headdim

        self.std_dev = 1 / np.sqrt(self.max_seq_len)

        self.M = nn.Parameter(
            torch.empty(self.nheads, self.max_seq_len, self.max_seq_len, **factory_kwargs)
        )
        nn.init.xavier_normal_(self.M)

    def forward(self, hidden_states):
        residual = hidden_states
        # Rearrange hidden states to shape [batch, n_heads, length, headdim]
        hidden_states = rearrange(hidden_states, 'b l (n h) -> b n l h', n=self.nheads)

        output = torch.einsum('b n t h, n l t -> b n l h', hidden_states, self.M)
        output = self.std_dev * output
        output = rearrange(output, 'b n l h -> b l (n h)') + residual

        return output
