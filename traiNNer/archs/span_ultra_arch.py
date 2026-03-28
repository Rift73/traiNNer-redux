"""
SPANUltra: Enhanced SPAN with Near-Free Quality Tricks

Based on SPAN (Swift Parameter-free Attention Network) with additional
near-free tricks adapted from DRFT, ESCReal, and SST architectures:

1. Conv3XC reparam: inherited from SPAN, multi-branch bottleneck fuses
   to plain 3×3 at inference (zero extra cost)
2. Depthwise 3×3 conv per SPAB block: spatial refinement (near-zero cost)
3. Dense skip connections: 1×1 projection of all block outputs (DRCT-style)
4. Dual input skip: 1×1→DW 7×7→1×1 from raw input (ESCReal-style)
5. Pixel Attention: 1×1 + sigmoid gating before PixelShuffle

Enhanced Architecture:
    LR Input ─────────────────────────────────────────┐
        ↓                                             │
    Head (Conv3XC)  ─────────────────────┐            │
        ↓                                │            │
    Body (N × SPAB) ─┬─ Dense Fusion ─┐  │            │
        ↓             └───────────────┘  │            │
    Add ←────────────────────────────────┘            │
        ↓                                             │
    Tail (Conv3XC)                                    │
        ↓                                             │
    Add ←── Input Skip (1×1→DW7×7→1×1) ──────────────┘
        ↓
    Upsampler (Conv → PixelAttention → PixelShuffle)
        ↓
    HR Output

Each enhanced SPAB block:
    Input → Conv3XC→SiLU → Conv3XC→SiLU → Conv3XC ─┐
      │                                              ↓
      │                           SimAtt = σ(out) - 0.5
      │                                              ↓
      └───────────────────── Add → DWConv3×3 → Mul(SimAtt)
                                                   ↓
                                                  Out

Usage:
    model = span_ultra(scale=4)
    output = model(lr_input)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from traiNNer.utils.registry import ARCH_REGISTRY


# =============================================================================
# Reparameterizable Convolution
# =============================================================================


class Conv3XC(nn.Module):
    """Reparameterizable 3×3 convolution.

    During training, uses a multi-branch topology for richer learning:
        Branch 1: 1×1 → 3×3 → 1×1 bottleneck (with channel expansion)
        Branch 2: 1×1 skip connection
    These are summed to produce the output.

    During inference, both branches fuse into a single 3×3 conv via
    ``update_params()``, so there is zero extra cost.

    Args:
        c_in: Input channels.
        c_out: Output channels.
        gain: Channel expansion factor in the bottleneck. Default: 2.
        s: Stride. Default: 1.
        bias: Whether to use bias. Default: True.
        act: Whether to apply LeakyReLU(0.05) after the conv. Default: False.
    """

    def __init__(
        self,
        c_in: int,
        c_out: int,
        gain: int = 2,
        s: int = 1,
        bias: bool = True,
        act: bool = False,
    ) -> None:
        super().__init__()
        self.has_act = act
        self.stride = s

        # Skip branch: 1×1 conv
        self.sk = nn.Conv2d(c_in, c_out, 1, stride=s, bias=bias)

        # Bottleneck branch: 1×1 → 3×3 → 1×1
        self.conv = nn.Sequential(
            nn.Conv2d(c_in, c_in * gain, 1, bias=bias),
            nn.Conv2d(c_in * gain, c_out * gain, 3, stride=s, padding=0, bias=bias),
            nn.Conv2d(c_out * gain, c_out, 1, bias=bias),
        )

        # Fused inference conv (populated by update_params)
        self.fused_conv = nn.Conv2d(c_in, c_out, 3, padding=1, stride=s, bias=bias)
        self.fused_conv.weight.requires_grad = False
        self.fused_conv.bias.requires_grad = False  # type: ignore[union-attr]
        self.update_params()

    def update_params(self) -> None:
        """Fuse bottleneck + skip into a single 3×3 kernel."""
        w1 = self.conv[0].weight.data.clone().detach()
        b1 = self.conv[0].bias.data.clone().detach()
        w2 = self.conv[1].weight.data.clone().detach()
        b2 = self.conv[1].bias.data.clone().detach()
        w3 = self.conv[2].weight.data.clone().detach()
        b3 = self.conv[2].bias.data.clone().detach()

        # Merge first two layers: 1×1 ⊛ 3×3
        w = (
            F.conv2d(w1.flip(2, 3).permute(1, 0, 2, 3), w2, padding=2, stride=1)
            .flip(2, 3)
            .permute(1, 0, 2, 3)
        )
        b = (w2 * b1.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b2

        # Merge with third layer: result ⊛ 1×1
        weight_concat = (
            F.conv2d(w.flip(2, 3).permute(1, 0, 2, 3), w3, padding=0, stride=1)
            .flip(2, 3)
            .permute(1, 0, 2, 3)
        )
        bias_concat = (w3 * b.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b3

        # Pad skip 1×1 to 3×3 and add
        sk_w = self.sk.weight.data.clone().detach()
        sk_b = self.sk.bias.data.clone().detach()  # type: ignore[union-attr]
        sk_w = F.pad(sk_w, [1, 1, 1, 1])

        self.fused_conv.weight.data = (weight_concat + sk_w).contiguous()
        self.fused_conv.bias.data = (bias_concat + sk_b).contiguous()  # type: ignore[union-attr]

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            x_pad = F.pad(x, (1, 1, 1, 1), "constant", 0)
            out = self.conv(x_pad) + self.sk(x)
        else:
            self.update_params()
            out = self.fused_conv(x)

        if self.has_act:
            out = F.leaky_relu(out, negative_slope=0.05)
        return out


# =============================================================================
# Building Blocks
# =============================================================================


class PixelAttention(nn.Module):
    """Pixel Attention for enhanced upsampling.

    Generates per-pixel attention weights via a 1×1 conv + sigmoid to
    refine features before PixelShuffle. Near-zero inference cost.

    Args:
        dim: Number of input channels.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.pa_conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()
        nn.init.trunc_normal_(self.pa_conv.weight, std=0.02)
        nn.init.zeros_(self.pa_conv.bias)

    def forward(self, x: Tensor) -> Tensor:
        return x * self.sigmoid(self.pa_conv(x))


class SPAB(nn.Module):
    """Swift Parameter-free Attention Block (Enhanced).

    Three Conv3XC layers with SiLU activation, followed by SPAN's
    signature parameter-free attention: sigma(out) - 0.5 scaling.

    Optionally enhanced with depthwise 3×3 conv for spatial refinement,
    applied additively after the local residual and before attention gating.

    Args:
        in_channels: Number of input channels.
        mid_channels: Number of intermediate channels. Default: same as in_channels.
        out_channels: Number of output channels. Default: same as in_channels.
        bias: Unused (Conv3XC always uses bias). Kept for API compatibility.
        use_dwconv: Add depthwise 3×3 conv for spatial refinement. Default: False.
    """

    def __init__(
        self,
        in_channels: int,
        mid_channels: int | None = None,
        out_channels: int | None = None,
        bias: bool = False,
        use_dwconv: bool = False,
    ) -> None:
        super().__init__()
        if mid_channels is None:
            mid_channels = in_channels
        if out_channels is None:
            out_channels = in_channels

        self.c1_r = Conv3XC(in_channels, mid_channels, gain=2, s=1)
        self.c2_r = Conv3XC(mid_channels, mid_channels, gain=2, s=1)
        self.c3_r = Conv3XC(mid_channels, out_channels, gain=2, s=1)
        self.act1 = nn.SiLU(inplace=True)

        self.use_dwconv = use_dwconv
        if use_dwconv:
            self.dwconv = nn.Conv2d(
                out_channels, out_channels, 3, padding=1, groups=out_channels
            )

    def forward(self, x: Tensor) -> Tensor:
        out1 = self.c1_r(x)
        out1_act = self.act1(out1)

        out2 = self.c2_r(out1_act)
        out2_act = self.act1(out2)

        out3 = self.c3_r(out2_act)

        sim_att = torch.sigmoid(out3) - 0.5
        residual = out3 + x
        if self.use_dwconv:
            residual = residual + self.dwconv(residual)
        out = residual * sim_att

        return out


# =============================================================================
# Main Network
# =============================================================================


class SPANUltra(nn.Module):
    """Enhanced SPAN with near-free quality tricks.

    A lightweight SR network combining SPAN's Conv3XC reparameterization and
    parameter-free attention (SPAB) with additional near-free enhancements:
    - Depthwise 3×3 conv per block for spatial refinement
    - Dense skip connections (DRCT-style) for improved gradient flow
    - Dual input skip (ESCReal-style) with wider receptive field
    - Pixel Attention in upsampler for per-pixel feature gating

    Args:
        num_in_ch: Number of input channels. Default: 3.
        num_out_ch: Number of output channels. Default: 3.
        feature_channels: Number of feature channels. Default: 48.
        num_block: Number of SPAB blocks. Default: 6.
        upscale: Upscaling factor. Default: 4.
        bias: Bias for SPAB blocks (Conv3XC always uses bias). Default: True.
        norm: Apply input normalization (mean subtraction + range scaling). Default: True.
        img_range: Input image range for normalization. Default: 255.0.
        rgb_mean: Mean RGB values for normalization. Default: ImageNet mean.
        use_dwconv: Add depthwise 3×3 conv per SPAB block. Default: False.
        dense_skip: Use dense skip connections across blocks. Default: False.
        dual_input_skip: Add ESCReal-style second skip from raw input. Default: False.
        pixel_attention: Add Pixel Attention before PixelShuffle. Default: False.
    """

    def __init__(
        self,
        *,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        feature_channels: int = 64,
        num_block: int = 6,
        upscale: int = 4,
        bias: bool = True,
        norm: bool = True,
        img_range: float = 255.0,
        rgb_mean: tuple[float, float, float] = (0.4488, 0.4371, 0.4040),
        use_dwconv: bool = False,
        dense_skip: bool = False,
        dual_input_skip: bool = False,
        pixel_attention: bool = False,
    ) -> None:
        super().__init__()
        self.img_range = img_range
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        self.dense_skip = dense_skip
        self.dual_input_skip = dual_input_skip

        # Normalization flag (stored as buffer for serialization)
        self.no_norm: Tensor | None
        if not norm:
            self.register_buffer("no_norm", torch.zeros(1))
        else:
            self.no_norm = None

        # Head: shallow feature extraction
        self.conv_1 = Conv3XC(num_in_ch, feature_channels, gain=2, s=1)

        # Body: deep feature extraction (ModuleList for dense skip support)
        self.body = nn.ModuleList(
            [
                SPAB(feature_channels, bias=bias, use_dwconv=use_dwconv)
                for _ in range(num_block)
            ]
        )

        # Dense skip: fuse all block outputs via 1×1 projection (DRCT-style)
        if dense_skip:
            self.dense_fusion = nn.Conv2d(
                feature_channels * num_block, feature_channels, 1
            )
            nn.init.trunc_normal_(self.dense_fusion.weight, std=0.02)
            nn.init.zeros_(self.dense_fusion.bias)  # type: ignore[arg-type]

        # Tail: feature refinement after global residual
        self.conv_2 = Conv3XC(feature_channels, feature_channels, gain=2, s=1)

        # Dual input skip: second path from raw input (ESCReal-style)
        # 1×1 expand → DW 7×7 spatial → LeakyReLU → 1×1 compress
        if dual_input_skip:
            self.input_skip = nn.Sequential(
                nn.Conv2d(num_in_ch, feature_channels * 2, 1),
                nn.Conv2d(
                    feature_channels * 2,
                    feature_channels * 2,
                    7,
                    padding=3,
                    groups=feature_channels * 2,
                ),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(feature_channels * 2, feature_channels, 1),
            )

        # Upsampler (optionally with Pixel Attention)
        up_ch = num_out_ch * upscale * upscale
        up_layers: list[nn.Module] = [nn.Conv2d(feature_channels, up_ch, 3, padding=1)]
        if pixel_attention:
            up_layers.append(PixelAttention(up_ch))
        up_layers.append(nn.PixelShuffle(upscale))
        self.upsampler = nn.Sequential(*up_layers)

    @property
    def is_norm(self) -> bool:
        return self.no_norm is None

    def forward(self, x: Tensor) -> Tensor:
        if self.is_norm:
            self.mean = self.mean.type_as(x)
            x = (x - self.mean) * self.img_range

        out_feature = self.conv_1(x)

        # Body with optional dense skip collection
        feat = out_feature
        if self.dense_skip:
            block_outputs: list[Tensor] = []
            for block in self.body:
                feat = block(feat)
                block_outputs.append(feat)
            # Additive dense fusion: sequential output + projected concat
            dense_cat = torch.cat(block_outputs, dim=1)
            feat = feat + self.dense_fusion(dense_cat)
        else:
            for block in self.body:
                feat = block(feat)

        # Global residual + tail
        feat = self.conv_2(feat + out_feature)

        # Dual input skip
        if self.dual_input_skip:
            feat = feat + self.input_skip(x)

        return self.upsampler(feat)

    def load_state_dict(  # type: ignore[override]
        self, state_dict: dict[str, Tensor], strict: bool = True, **kwargs
    ):
        """Load weights, auto-converting original SPAN keys if needed."""
        if any(k.startswith("block_1.") for k in state_dict):
            state_dict = self._convert_span_keys(state_dict)
        return super().load_state_dict(state_dict, strict=strict, **kwargs)

    @staticmethod
    def _convert_span_keys(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        """Map original SPAN block_N.* keys to body.N-1.* format.

        Skips conv_cat weights since SPANUltra uses dense_fusion instead.
        """
        converted: dict[str, Tensor] = {}
        for key, value in state_dict.items():
            # Skip conv_cat (different architecture)
            if key.startswith("conv_cat."):
                continue
            new_key = key
            for i in range(1, 20):
                prefix = f"block_{i}."
                if key.startswith(prefix):
                    new_key = f"body.{i - 1}." + key[len(prefix):]
                    break
            converted[new_key] = value
        return converted


# =============================================================================
# Registry Functions (traiNNer-redux integration)
# =============================================================================


def _make_span_ultra(
    feature_channels: int,
    num_in_ch: int = 3,
    num_out_ch: int = 3,
    num_block: int = 6,
    scale: int = 4,
    bias: bool = True,
    norm: bool = False,
    img_range: float = 255.0,
    rgb_mean: tuple[float, float, float] = (0.4488, 0.4371, 0.4040),
) -> SPANUltra:
    return SPANUltra(
        num_in_ch=num_in_ch,
        num_out_ch=num_out_ch,
        feature_channels=feature_channels,
        num_block=num_block,
        upscale=scale,
        bias=bias,
        norm=norm,
        img_range=img_range,
        rgb_mean=rgb_mean,
        use_dwconv=True,
        dense_skip=True,
        dual_input_skip=True,
        pixel_attention=True,
    )


@ARCH_REGISTRY.register()
def span_ultra(
    num_in_ch: int = 3,
    num_out_ch: int = 3,
    feature_channels: int = 64,
    num_block: int = 6,
    scale: int = 4,
    bias: bool = True,
    norm: bool = False,
    img_range: float = 255.0,
    rgb_mean: tuple[float, float, float] = (0.4488, 0.4371, 0.4040),
) -> SPANUltra:
    """SPANUltra — 64 channel default. All near-free tricks enabled."""
    return _make_span_ultra(
        feature_channels=feature_channels,
        num_in_ch=num_in_ch,
        num_out_ch=num_out_ch,
        num_block=num_block,
        scale=scale,
        bias=bias,
        norm=norm,
        img_range=img_range,
        rgb_mean=rgb_mean,
    )


@ARCH_REGISTRY.register()
def span_ultra_f32(
    num_in_ch: int = 3,
    num_out_ch: int = 3,
    feature_channels: int = 32,
    num_block: int = 6,
    scale: int = 4,
    bias: bool = True,
    norm: bool = False,
    img_range: float = 255.0,
    rgb_mean: tuple[float, float, float] = (0.4488, 0.4371, 0.4040),
) -> SPANUltra:
    """SPANUltra — 32 channel lightweight variant."""
    return _make_span_ultra(
        feature_channels=feature_channels,
        num_in_ch=num_in_ch,
        num_out_ch=num_out_ch,
        num_block=num_block,
        scale=scale,
        bias=bias,
        norm=norm,
        img_range=img_range,
        rgb_mean=rgb_mean,
    )


@ARCH_REGISTRY.register()
def span_ultra_f48(
    num_in_ch: int = 3,
    num_out_ch: int = 3,
    feature_channels: int = 48,
    num_block: int = 6,
    scale: int = 4,
    bias: bool = True,
    norm: bool = False,
    img_range: float = 255.0,
    rgb_mean: tuple[float, float, float] = (0.4488, 0.4371, 0.4040),
) -> SPANUltra:
    """SPANUltra — 48 channel variant."""
    return _make_span_ultra(
        feature_channels=feature_channels,
        num_in_ch=num_in_ch,
        num_out_ch=num_out_ch,
        num_block=num_block,
        scale=scale,
        bias=bias,
        norm=norm,
        img_range=img_range,
        rgb_mean=rgb_mean,
    )


@ARCH_REGISTRY.register()
def span_ultra_f96(
    num_in_ch: int = 3,
    num_out_ch: int = 3,
    feature_channels: int = 96,
    num_block: int = 6,
    scale: int = 4,
    bias: bool = True,
    norm: bool = False,
    img_range: float = 255.0,
    rgb_mean: tuple[float, float, float] = (0.4488, 0.4371, 0.4040),
) -> SPANUltra:
    """SPANUltra — 96 channel heavy variant."""
    return _make_span_ultra(
        feature_channels=feature_channels,
        num_in_ch=num_in_ch,
        num_out_ch=num_out_ch,
        num_block=num_block,
        scale=scale,
        bias=bias,
        norm=norm,
        img_range=img_range,
        rgb_mean=rgb_mean,
    )


# =============================================================================
# Utility
# =============================================================================


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("SPANUltra Architecture Variants:")
    print("=" * 80)

    variants = [
        ("span_ultra_f32", span_ultra_f32),
        ("span_ultra_f48", span_ultra_f48),
        ("span_ultra", span_ultra),
        ("span_ultra_f96", span_ultra_f96),
    ]

    x = torch.randn(1, 3, 64, 64)

    for name, factory in variants:
        model = factory(scale=4)
        train_params = count_parameters(model)

        with torch.no_grad():
            y = model(x)
        print(
            f"{name:20} | train: {train_params:>10,} params"
            f" | {tuple(x.shape)} -> {tuple(y.shape)}"
        )

    print("\n(inference uses fused 3x3 convs — same speed as plain SPAN)")
