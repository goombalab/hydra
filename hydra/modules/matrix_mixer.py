# Copyright (c) 2024, Sukjun Hwang, Aakash Lahoti, Ratish Puduppully, Tri Dao, Albert Gu.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

try:
    from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
except ImportError:
    RMSNormGated = None

from .matrix_mixers import (
    Dense, 
    Toeplitz,
    Vandermonde,
    Cauchy,
    LowRank,
    Attention,
    Quasiseparable,
)


class MatrixMixer(nn.Module):
    def __init__(
        self,
        matrix_mixer_type,
        is_data_dependent,
        d_model,
        qk_dim,
        max_seq_len=None, # max_seq_len is necessary for data-independent versions.
        d_conv=7,
        conv_init=None,
        expand=2,
        headdim=128,
        activation="swish",
        bias=False,
        conv_bias=True,
        # Fused kernel and sharding options
        chunk_size=256,
        layer_idx=None,  # Absorb kwarg for general module
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.matrix_mixer_type = matrix_mixer_type
        self.is_data_dependent = is_data_dependent
        self.d_model = d_model
        self.qk_dim = qk_dim
        self.max_seq_len = max_seq_len
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.d_inner = self.expand * self.d_model
        self.headdim = headdim
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        self.d_state = self.nheads * qk_dim
        self.activation = activation
        self.chunk_size = chunk_size
        self.layer_idx = layer_idx

        matrix_mixer, d_in_proj, conv_dim = self.build_matrix_mixer()
        self.matrix_mixer = matrix_mixer

        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv // 2,
            **factory_kwargs,
        )
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)
        # self.conv1d.weight._no_weight_decay = True

        self.act = nn.SiLU()

        # Extra normalization layer right before output projection
        assert RMSNormGated is not None
        self.norm = RMSNormGated(self.d_inner, eps=1e-5, norm_before_gate=True, **factory_kwargs)

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def build_matrix_mixer(self):
        if self.matrix_mixer_type == "dense":
            assert not self.is_data_dependent, "Data dependent Dense matrix mixer is not supported."
            matrix_mixer = Dense(
                self.d_model,
                max_seq_len=self.max_seq_len,
                expand=self.expand,
                headdim=self.headdim,
            )
            d_in_proj = 2 * self.d_inner
            conv_dim = self.d_inner
        elif self.matrix_mixer_type == "toeplitz":
            matrix_mixer = Toeplitz(
                self.is_data_dependent,
                self.d_model,
                max_seq_len=self.max_seq_len,
                expand=self.expand,
                headdim=self.headdim,
            )
            d_in_proj = 2 * self.d_inner + self.is_data_dependent * (2 * self.nheads)
            conv_dim = self.d_inner + self.is_data_dependent * (2 * self.nheads)
        elif self.matrix_mixer_type == "vandermonde":
            matrix_mixer = Vandermonde(
                self.is_data_dependent,
                self.d_model,
                self.qk_dim,
                max_seq_len=self.max_seq_len,
                expand=self.expand,
                headdim=self.headdim,
            )
            d_in_proj = 2 * self.d_inner + self.is_data_dependent * (2 * self.d_state)
            conv_dim = self.d_inner + self.is_data_dependent * (2 * self.d_state)
        elif self.matrix_mixer_type == "cauchy":
            matrix_mixer = Cauchy(
                self.is_data_dependent,
                self.d_model,
                self.qk_dim,
                max_seq_len=self.max_seq_len,
                expand=self.expand,
                headdim=self.headdim,
            )
            d_in_proj = 2 * self.d_inner + self.is_data_dependent * (2 * self.d_state)
            conv_dim = self.d_inner + self.is_data_dependent * (2 * self.d_state)
        elif self.matrix_mixer_type == "low_rank":
            matrix_mixer = LowRank(
                self.is_data_dependent,
                self.d_model,
                self.qk_dim,
                max_seq_len=self.max_seq_len,
                expand=self.expand,
                headdim=self.headdim,
            )
            d_in_proj = 2 * self.d_inner + self.is_data_dependent * (2 * self.d_state)
            conv_dim = self.d_inner + self.is_data_dependent * (2 * self.d_state)
        elif self.matrix_mixer_type == "attention":
            matrix_mixer = Attention(
                self.is_data_dependent,
                self.d_model,
                self.qk_dim,
                max_seq_len=self.max_seq_len,
                expand=self.expand,
                headdim=self.headdim,
            )
            d_in_proj = 2 * self.d_inner + self.is_data_dependent * (2 * self.d_state)
            conv_dim = self.d_inner + self.is_data_dependent * (2 * self.d_state)
        elif self.matrix_mixer_type == "quasiseparable":
            matrix_mixer = Quasiseparable(
                self.is_data_dependent,
                self.d_model,
                self.qk_dim,
                max_seq_len=self.max_seq_len,
                expand=self.expand,
                headdim=self.headdim,
                chunk_size=self.chunk_size,
            )
            # Order: [z, x, B, C, dt]
            d_in_proj = 2 * self.d_inner + self.is_data_dependent * (2 * self.d_state + 2 * self.nheads)
            conv_dim = self.d_inner + self.is_data_dependent * (2 * self.d_state)
        else:
            raise NotImplementedError

        return matrix_mixer, d_in_proj, conv_dim

    def forward(self, u):
        """
        u: (B, L, D)
        Returns: same shape as u
        """
        batch, seqlen, dim = u.shape
        assert self.activation in ["silu", "swish"]

        u_proj = self.in_proj(u)  # (B, L, d_in_proj)

        if self.matrix_mixer_type == "dense":
            z, x = torch.split(u_proj, [self.d_inner, self.d_inner], dim=-1)
            x = self.act(self.conv1d(x.transpose(1, 2))).transpose(1, 2)
            y = self.matrix_mixer(x)

        elif self.matrix_mixer_type == "toeplitz":
            if self.is_data_dependent:
                z, x_and_conv = torch.split(u_proj, [self.d_inner, self.d_inner + 2 * self.nheads], dim=-1)
                x_and_conv = self.act(self.conv1d(x_and_conv.transpose(1, 2))).transpose(1, 2)
                x, forward_conv, reverse_conv = torch.split(
                    x_and_conv,
                    [self.d_inner, self.nheads, self.nheads],
                    dim=-1
                )
                y = self.matrix_mixer(x, forward_conv=forward_conv, reverse_conv=reverse_conv)
            else:
                z, x = torch.split(u_proj, [self.d_inner, self.d_inner], dim=-1)
                x = self.act(self.conv1d(x.transpose(1, 2))).transpose(1, 2)
                y = self.matrix_mixer(x)

        elif self.matrix_mixer_type in ["vandermonde", "cauchy", "attention"]:
            if self.is_data_dependent:
                z, vqk = torch.split(u_proj, [self.d_inner, self.d_inner + 2 * self.d_state], dim=-1)
                vqk = self.conv1d(vqk.transpose(1, 2)).transpose(1, 2)
                v, q, k = torch.split(vqk, [self.d_inner, self.d_state, self.d_state], dim=-1)
                v = self.act(v)
                y = self.matrix_mixer(v, q=q, k=k)
            else:
                z, x = torch.split(u_proj, [self.d_inner, self.d_inner], dim=-1)
                x = self.act(self.conv1d(x.transpose(1, 2))).transpose(1, 2)
                y = self.matrix_mixer(x)

        elif self.matrix_mixer_type == "low_rank":
            if self.is_data_dependent:
                z, vqk = torch.split(u_proj, [self.d_inner, self.d_inner + 2 * self.d_state], dim=-1)
                vqk = self.act(self.conv1d(vqk.transpose(1, 2))).transpose(1, 2)
                v, q, k = torch.split(vqk, [self.d_inner, self.d_state, self.d_state], dim=-1)
                y = self.matrix_mixer(v, q=q, k=k)
            else:
                z, x = torch.split(u_proj, [self.d_inner, self.d_inner], dim=-1)
                x = self.act(self.conv1d(x.transpose(1, 2))).transpose(1, 2)
                y = self.matrix_mixer(x)

        elif self.matrix_mixer_type == "quasiseparable":
            if self.is_data_dependent:
                z, vqk, dt = torch.split(
                    u_proj,
                    [self.d_inner, self.d_inner + 2 * self.d_state, 2 * self.nheads],
                    dim=-1
                )
                dt = dt + repeat(self.matrix_mixer.dt_bias, 'h -> (2 h)')

                vqk = self.act(self.conv1d(vqk.transpose(1, 2)).transpose(1, 2))
                v, qk = torch.split(vqk, [self.d_inner, 2 * self.d_state], dim=-1)

                y = self.matrix_mixer(v, qk, dt)
            else:
                z, v = torch.split(u_proj, [self.d_inner, self.d_inner], dim=-1)
                dt = repeat(self.matrix_mixer.dt_bias, 'h -> b l (2 h)', b=batch, l=seqlen)

                v = self.act(self.conv1d(v.transpose(1, 2)).transpose(1, 2))
                qk = repeat(self.matrix_mixer.BC, 'l d -> b l d', b=batch)

                y = self.matrix_mixer(v, qk, dt)

        # Multiply "gate" branch and apply extra normalization layer
        y = self.norm(y, z)
        out = self.out_proj(y)

        return out
