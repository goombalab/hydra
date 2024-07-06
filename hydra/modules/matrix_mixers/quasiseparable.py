# Copyright (c) 2024, Sukjun Hwang, Aakash Lahoti, Ratish Puduppully, Tri Dao, Albert Gu.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined


class Quasiseparable(nn.Module):
    # NOTE Data-dependent quasiseparable is equivalent to Hydra that
    # shares BC for forward and reverse sequence processing.
    def __init__(
        self,
        is_data_dependent,
        d_model,
        qk_dim,
        max_seq_len=None, # max_seq_len is necessary for data-independent version.
        expand=2,
        headdim=128,
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        # Fused kernel and sharding options
        chunk_size=256,
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
        self.dt_limit = dt_limit
        self.chunk_size = chunk_size

        if not self.is_data_dependent:
            self.BC = nn.Parameter(
                torch.empty(self.max_seq_len, 2 * self.d_state, **factory_kwargs)
            )
            nn.init.xavier_normal_(self.BC)

        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        # A parameter
        A = torch.ones(self.nheads, dtype=torch.float32, device=device)
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        # self.register_buffer("A_log", torch.zeros(self.nheads, dtype=torch.float32, device=device), persistent=True)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.nheads, device=device))
        self.D._no_weight_decay = True
        self.fc_D = nn.Linear(self.d_inner, self.nheads, bias=False, **factory_kwargs)

    def forward(self, x, BC, dt):
        batch, seqlen, dim = x.shape

        A = -torch.exp(self.A_log.float())  # (nheads) or (d_inner, d_state)
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)

        dt = torch.cat((dt[:, :, :self.nheads], torch.flip(dt[:, :, self.nheads:], (1,))), dim=0)
        dt = F.softplus(dt) # (2 * B, L, nheads)

        x_og = x
        x = torch.cat((x, torch.flip(x, (1,))), dim=0)
        if BC.dtype != x.dtype:
            BC = BC.to(x.dtype)
        BC = torch.cat((BC, torch.flip(BC, (1,))), dim=0)
        B, C = torch.split(BC, [self.d_state, self.d_state], dim=-1)

        y = mamba_chunk_scan_combined(
            rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
            dt,
            A,
            rearrange(B, "b l (g n) -> b l g n", g=1),  # Fixing ngroups to 1 for simplicity.
            rearrange(C, "b l (g n) -> b l g n", g=1),  # Fixing ngroups to 1 for simplicity.
            chunk_size=self.chunk_size,
            D=None,
            z=None,
            seq_idx=None,
            initial_states=None,
            **dt_limit_kwargs,
        )
        y = rearrange(y, "b l h p -> b l (h p)")
        y = torch.roll(y, shifts=1, dims=1)
        y[:, 0, :] = 0.0
        y_fw, y_bw = y[:batch], torch.flip(y[batch:], (1,))
        y = y_fw + y_bw + x_og * repeat(
            F.linear(x_og, self.fc_D.weight, bias=self.D), "b l h -> b l (h p)", p=self.headdim
        )

        return y
