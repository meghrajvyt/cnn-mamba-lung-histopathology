"""
Core Selective State Space Model (SSM) Block.

Implements the Mamba selective scan mechanism (Gu & Dao, 2023) with
input-dependent parameters B, C, Δ and discretization via Zero-Order Hold.

Reference: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
           arXiv:2312.00752
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class SelectiveSSM(nn.Module):
    """
    Selective State Space Model with input-dependent parameter generation.

    Unlike traditional SSMs where B, C, Δ are fixed, this module computes them
    as functions of the input, enabling content-aware filtering.

    Args:
        d_model: Input/output dimension.
        d_state: SSM state dimension (N in the paper).
        d_conv: Local convolution width for initial feature mixing.
        expand: Expansion factor for inner dimension.
        dt_rank: Rank of Δ projection. 'auto' sets it to ceil(d_model / 16).
        dt_min: Minimum value for Δ initialization.
        dt_max: Maximum value for Δ initialization.
        dt_init: Initialization method for Δ ('random' or 'constant').
        dt_scale: Scaling factor for Δ initialization.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        scan_chunk_size: int = 32,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.scan_chunk_size = scan_chunk_size

        # Input projection: x -> (z, x) where z is the gate
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)

        # Local convolution for initial feature mixing
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,  # Depthwise separable
            bias=True,
        )

        # Input-dependent parameter projections: S_B(x), S_C(x), S_Δ(x)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)

        # Δ projection from dt_rank to d_inner
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Initialize Δ projection bias for proper time-step range
        self._init_dt_proj(dt_min, dt_max, dt_init, dt_scale)

        # State matrix A (structured as diagonal, initialized with HiPPO)
        # A is (d_inner, d_state) — diagonal elements of the state transition
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32),
            "n -> d n",
            d=self.d_inner,
        )
        self.A_log = nn.Parameter(torch.log(A))  # Store in log space for stability
        self.A_log._no_weight_decay = True

        # D is the skip connection coefficient
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    def _init_dt_proj(self, dt_min, dt_max, dt_init, dt_scale):
        """Initialize the Δ projection for proper time-step distribution."""
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError(f"Unknown dt_init: {dt_init}")

        # Bias: inverse softplus of uniform samples in [dt_min, dt_max]
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))  # Inverse softplus
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Mark bias as non-decayed
        self.dt_proj.bias._no_weight_decay = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Selective SSM.

        Args:
            x: Input tensor of shape (B, L, D) where B=batch, L=length, D=d_model.

        Returns:
            Output tensor of shape (B, L, D).
        """
        batch, seqlen, _ = x.shape

        # Input projection: split into main branch (x) and gate (z)
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x_branch, z = xz.chunk(2, dim=-1)  # Each: (B, L, d_inner)

        # Local convolution
        x_branch = rearrange(x_branch, "b l d -> b d l")
        x_branch = self.conv1d(x_branch)[:, :, :seqlen]  # Causal: trim padding
        x_branch = rearrange(x_branch, "b d l -> b l d")
        x_branch = F.silu(x_branch)

        # Compute input-dependent parameters
        x_dbl = self.x_proj(x_branch)  # (B, L, dt_rank + 2*d_state)
        dt, B, C = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )

        # Project Δ to full dimension and apply softplus
        dt = self.dt_proj(dt)  # (B, L, d_inner)
        dt = F.softplus(dt)  # Ensure positive time steps

        # Recover A from log space
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state), negative for stability

        # Run the selective scan (pure PyTorch implementation)
        y = self._selective_scan(x_branch, dt, A, B, C)

        # Skip connection with D
        y = y + x_branch * self.D.unsqueeze(0).unsqueeze(0)

        # Gating with SiLU
        y = y * F.silu(z)

        # Output projection
        return self.out_proj(y)

    def _discretize(
        self,
        u: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Zero-order-hold discretisation of the state-space parameters.

        Returns ``(dA_log, dBu)`` where ``dA_log = Δ·A`` is the *logarithm* of
        the discretised state transition (kept in log space because ``A < 0``,
        so ``exp(dA_log) ∈ (0, 1]`` and never overflows), and
        ``dBu = Δ·B·u`` is the discretised, input-scaled drive term.

        Shapes: ``u, delta`` are (B, L, d_inner); ``A`` is (d_inner, d_state);
        ``B`` is (B, L, d_state). Both outputs are (B, L, d_inner, d_state).
        """
        dA_log = delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)
        dBu = delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)
        return dA_log, dBu

    def _selective_scan(
        self,
        u: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        """Chunked parallel selective scan.

        Solves the same first-order linear recurrence as
        :meth:`_selective_scan_reference`::

            h_k = exp(Δ_k·A) · h_{k-1} + Δ_k·B_k·u_k
            y_k = C_k · h_k

        but replaces the length-``L`` Python loop with a **Hillis–Steele
        associative scan** applied within fixed-size chunks, carrying the
        hidden state across chunk boundaries.

        The recurrence is associative under the operator

            (a₁, b₁) ∘ (a₂, b₂) = (a₂·a₁,  a₂·b₁ + b₂)

        where ``(a, b)`` is the affine transform ``h ↦ a·h + b``. Composing
        it by repeated doubling turns ``L`` sequential steps into
        ``⌈L/chunk⌉ · log₂(chunk)`` steps — for L=196 and chunk=32 that is
        7 × 5 = 35 rather than 196, and each step is a single fused
        elementwise kernel over the whole chunk instead of a tiny per-token op.

        Numerically this is *safer* than the naive cumulative-sum trick: every
        ``a`` lies in (0, 1] because ``A < 0``, so all partial products stay
        bounded by 1 and no intermediate ``exp`` can overflow.

        The scan runs in float32 regardless of autocast, since the state
        recurrence is sensitive to reduced precision, then casts back.

        Args:
            u: Input (B, L, d_inner).
            delta: Time steps Δ (B, L, d_inner).
            A: State matrix (d_inner, d_state), strictly negative.
            B: Input-dependent B (B, L, d_state).
            C: Input-dependent C (B, L, d_state).

        Returns:
            Output (B, L, d_inner).
        """
        out_dtype = u.dtype

        # The state recurrence is precision-sensitive; force fp32 under AMP.
        u, delta, B, C = u.float(), delta.float(), B.float(), C.float()
        A = A.float()

        batch, seqlen, d_inner = u.shape
        dA_log, dBu = self._discretize(u, delta, A, B)

        chunk = self.scan_chunk_size if self.scan_chunk_size > 0 else seqlen
        chunk = min(chunk, seqlen)

        h = u.new_zeros(batch, d_inner, self.d_state)
        ys = []

        for start in range(0, seqlen, chunk):
            stop = min(start + chunk, seqlen)
            a = torch.exp(dA_log[:, start:stop])  # (B, Lc, d, n), in (0, 1]
            b = dBu[:, start:stop]                # (B, Lc, d, n)
            Lc = stop - start

            # Inclusive associative scan by doubling.
            step = 1
            while step < Lc:
                a_prev = torch.cat(
                    [a.new_ones(batch, step, d_inner, self.d_state), a[:, :-step]],
                    dim=1,
                )
                b_prev = torch.cat(
                    [b.new_zeros(batch, step, d_inner, self.d_state), b[:, :-step]],
                    dim=1,
                )
                # Order matters: `a` is the later transform, `*_prev` the earlier.
                b = b + a * b_prev
                a = a * a_prev
                step *= 2

            # `b[t]` is now the within-chunk state assuming h_prev = 0, and
            # `a[t]` the cumulative decay, so fold in the carried state.
            h_chunk = b + a * h.unsqueeze(1)  # (B, Lc, d, n)

            ys.append(torch.einsum("bldn,bln->bld", h_chunk, C[:, start:stop]))
            h = h_chunk[:, -1]

        y = torch.cat(ys, dim=1)  # (B, L, d_inner)
        return y.to(out_dtype)

    def _selective_scan_reference(
        self,
        u: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        """Naive sequential selective scan. Kept as the correctness oracle.

        This is the original O(L)-sequential-steps implementation. It is far
        slower than :meth:`_selective_scan` and is retained only so that the
        parallel scan can be checked against it in tests. Do not use for
        training.
        """
        out_dtype = u.dtype
        u, delta, B, C = u.float(), delta.float(), B.float(), C.float()
        A = A.float()

        batch, seqlen, d_inner = u.shape
        dA_log, dBu = self._discretize(u, delta, A, B)
        deltaA = torch.exp(dA_log)

        h = u.new_zeros(batch, d_inner, self.d_state)
        ys = []
        for i in range(seqlen):
            h = deltaA[:, i] * h + dBu[:, i]              # (B, d_inner, d_state)
            ys.append(torch.einsum("bdn,bn->bd", h, C[:, i]))  # (B, d_inner)

        return torch.stack(ys, dim=1).to(out_dtype)


class SS2D(nn.Module):
    """
    2D Selective Scan module — scans 2D feature maps in 4 directions.

    Bridges the gap between 1D selective scan and 2D image structure by
    processing the feature map along four scan paths:
        1. Top-left → Bottom-right (row-major)
        2. Bottom-right → Top-left (reverse row-major)
        3. Top-right → Bottom-left (column-major)
        4. Bottom-left → Top-right (reverse column-major)

    Reference: VMamba (Liu et al., 2024) - arXiv:2401.10166
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: str = "auto",
        num_directions: int = 4,
        scan_chunk_size: int = 32,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_directions = num_directions

        # One SSM per scan direction
        self.ssm_blocks = nn.ModuleList([
            SelectiveSSM(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dt_rank=dt_rank,
                scan_chunk_size=scan_chunk_size,
            )
            for _ in range(num_directions)
        ])

        # Merge projections from all directions
        self.merge_proj = nn.Linear(d_model * num_directions, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Forward pass with 4-directional selective scan.

        Args:
            x: Flattened 2D features (B, H*W, D).
            H: Height of the 2D feature map.
            W: Width of the 2D feature map.

        Returns:
            Output tensor of shape (B, H*W, D).
        """
        B, L, D = x.shape
        assert L == H * W, f"Sequence length {L} != H*W ({H}*{W})"

        # Reshape to 2D grid
        x_2d = rearrange(x, "b (h w) d -> b h w d", h=H, w=W)

        # Generate 4 scan sequences
        sequences = self._create_scan_sequences(x_2d, H, W)

        # Run SSM on each direction
        outputs = []
        for i, (seq, ssm) in enumerate(zip(sequences, self.ssm_blocks)):
            out = ssm(seq)  # (B, L, D)
            outputs.append(out)

        # Reverse the scan orders to restore original spatial positions
        restored = self._restore_scan_order(outputs, H, W)

        # Concatenate and merge
        merged = torch.cat(restored, dim=-1)  # (B, L, D*num_directions)
        out = self.merge_proj(merged)  # (B, L, D)
        out = self.norm(out)

        return out

    def _create_scan_sequences(self, x_2d: torch.Tensor, H: int, W: int):
        """Create 4-directional scan sequences from a 2D feature map."""
        # Direction 1: Top-left → Bottom-right (standard row-major)
        seq1 = rearrange(x_2d, "b h w d -> b (h w) d")

        # Direction 2: Bottom-right → Top-left (reverse of direction 1)
        seq2 = torch.flip(seq1, dims=[1])

        # Direction 3: Top-right → Bottom-left (column-major, flipped horizontally)
        x_flip_w = torch.flip(x_2d, dims=[2])  # Flip along width
        seq3 = rearrange(x_flip_w, "b h w d -> b (w h) d")  # Column-major order

        # Direction 4: Bottom-left → Top-right (reverse of direction 3)
        seq4 = torch.flip(seq3, dims=[1])

        return [seq1, seq2, seq3, seq4]

    def _restore_scan_order(self, outputs, H: int, W: int):
        """Reverse each scan direction to restore original spatial ordering."""
        out1 = outputs[0]  # Already in row-major order

        out2 = torch.flip(outputs[1], dims=[1])  # Reverse of reverse row-major

        # Direction 3: column-major of flipped → restore
        B, L, D = outputs[2].shape
        out3_2d = rearrange(outputs[2], "b (w h) d -> b h w d", h=H, w=W)
        out3_2d = torch.flip(out3_2d, dims=[2])
        out3 = rearrange(out3_2d, "b h w d -> b (h w) d")

        # Direction 4: reverse of direction 3
        out4_flipped = torch.flip(outputs[3], dims=[1])
        out4_2d = rearrange(out4_flipped, "b (w h) d -> b h w d", h=H, w=W)
        out4_2d = torch.flip(out4_2d, dims=[2])
        out4 = rearrange(out4_2d, "b h w d -> b (h w) d")

        return [out1, out2, out3, out4]
