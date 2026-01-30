"""
Dynamic MobileNet (DyMN) implementation for EfficientAT.

DyMN replaces static convolutions in MobileNetV3 with input-dependent dynamic
convolutions, dynamic activations (DyReLU-B), and coordinate attention (ContextGen).
Each convolution uses k=4 learned basis kernels combined via attention weights
derived from an input-dependent context vector.

Reference: https://github.com/fschmid56/EfficientAT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Union
import logging

from .utils import make_divisible

logger = logging.getLogger(__name__)


class ContextGen(nn.Module):
    """Coordinate attention context generator.

    Produces a global context vector (for dynamic conv attention) and
    frequency/time gating signals (for coordinate attention).
    """

    def __init__(self, in_channels: int, context_dim: int, expanded_channels: int):
        super().__init__()
        self.joint_conv = nn.Conv2d(in_channels, context_dim, kernel_size=1, bias=False)
        self.joint_norm = nn.BatchNorm2d(context_dim)
        self.conv_f = nn.Conv2d(context_dim, expanded_channels, kernel_size=1, bias=True)
        self.conv_t = nn.Conv2d(context_dim, expanded_channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, in_channels, F, T)
        Returns:
            context: (B, context_dim) global context vector
            f_gate: (B, expanded_channels, F, 1) frequency gating
            t_gate: (B, expanded_channels, 1, T) time gating
        """
        B, C, F_dim, T_dim = x.shape

        # Pool along time and frequency separately
        x_f = x.mean(dim=3, keepdim=True)   # (B, C, F, 1)
        x_t = x.mean(dim=2, keepdim=True)   # (B, C, 1, T)
        x_t = x_t.permute(0, 1, 3, 2)       # (B, C, T, 1)

        # Concatenate along spatial dimension
        joint_input = torch.cat([x_f, x_t], dim=2)  # (B, C, F+T, 1)

        # Joint convolution + norm + activation
        joint = self.joint_conv(joint_input)   # (B, ctx_dim, F+T, 1)
        joint = self.joint_norm(joint)
        joint = F.hardswish(joint)

        # Global context vector for dynamic conv attention
        context = joint.mean(dim=[2, 3])  # (B, ctx_dim)

        # Split back into frequency and time parts
        f_part = joint[:, :, :F_dim, :]  # (B, ctx_dim, F, 1)
        t_part = joint[:, :, F_dim:, :]  # (B, ctx_dim, T, 1)
        t_part = t_part.permute(0, 1, 3, 2)  # (B, ctx_dim, 1, T)

        # Project to expanded channel dimension for gating
        f_gate = self.conv_f(f_part)  # (B, exp_c, F, 1)
        t_gate = self.conv_t(t_part)  # (B, exp_c, 1, T)

        return context, f_gate, t_gate


class DynamicConv(nn.Module):
    """Dynamic convolution with k learned basis kernels.

    Stores k flattened basis kernels and uses a small attention network
    to combine them based on the input context.
    """

    def __init__(self, flat_size: int, k: int = 4, context_dim: int = 32):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1, 1, k, flat_size))
        self.residuals = nn.ModuleList([nn.Linear(context_dim, k)])
        self.k = k

    def forward(self, context: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Args:
            context: (B, context_dim)
            temperature: softmax temperature (higher = softer selection)
        Returns:
            aggregated flat weight: (B, flat_size)
        """
        # Compute attention over k basis kernels
        attn = self.residuals[0](context)  # (B, k)
        attn = F.softmax(attn / temperature, dim=1)  # (B, k)

        # Weighted sum of basis kernels
        # weight: (1, 1, k, flat) ; attn: (B, 1, k, 1)
        attn = attn.unsqueeze(1).unsqueeze(3)  # (B, 1, k, 1)
        agg = (self.weight * attn).sum(dim=2)  # (B, 1, flat)

        return agg.squeeze(1)  # (B, flat)


class DyReLUB(nn.Module):
    """Dynamic ReLU-B activation.

    Generates input-dependent piecewise linear activation functions with
    2 branches (4 coefficients per channel: slope1, intercept1, slope2, intercept2).
    """

    def __init__(self, channels: int, context_dim: int):
        super().__init__()
        self.channels = channels
        self.coef_net = nn.Sequential(
            nn.Linear(context_dim, 4 * channels)
        )
        self.lambdas = nn.Parameter(torch.ones(4))
        self.init_v = nn.Parameter(torch.zeros(4))

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, F, T)
            context: (B, context_dim)
        Returns:
            activated: (B, C, F, T)
        """
        B = x.size(0)

        # Generate per-channel, per-sample coefficients
        coefs = self.coef_net(context)  # (B, 4*C)
        coefs = coefs.view(B, 4, self.channels)  # (B, 4, C)

        # Apply lambdas and init_v
        # lambdas and init_v: (4,) -> (1, 4, 1)
        lam = self.lambdas.view(1, 4, 1)
        iv = self.init_v.view(1, 4, 1)
        activations = lam * coefs + iv  # (B, 4, C)

        # Reshape for broadcasting: (B, 4, C, 1, 1)
        activations = activations.unsqueeze(3).unsqueeze(4)

        # Two piecewise linear branches: max(a1*x + b1, a2*x + b2)
        a1 = activations[:, 0:1]  # (B, 1, C, 1, 1) - slope 1
        b1 = activations[:, 1:2]  # (B, 1, C, 1, 1) - intercept 1
        a2 = activations[:, 2:3]  # (B, 1, C, 1, 1) - slope 2
        b2 = activations[:, 3:4]  # (B, 1, C, 1, 1) - intercept 2

        x_unsq = x.unsqueeze(1)  # (B, 1, C, F, T)
        branch1 = a1 * x_unsq + b1
        branch2 = a2 * x_unsq + b2
        out = torch.max(branch1, branch2)  # (B, 1, C, F, T)

        return out.squeeze(1)  # (B, C, F, T)


class DyMNBlock(nn.Module):
    """Dynamic inverted residual block.

    Replaces all static convolutions with DynamicConv and standard
    activations with DyReLU-B. Uses ContextGen for coordinate attention.
    """

    def __init__(self, in_channels: int, expanded_channels: int,
                 out_channels: int, kernel_size: int, stride: int,
                 context_dim: int, k: int = 4, has_expansion: bool = True):
        super().__init__()
        self.use_res_connect = (stride == 1 and in_channels == out_channels)
        self.has_expansion = has_expansion
        self.in_channels = in_channels
        self.expanded_channels = expanded_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        depth_channels = expanded_channels if has_expansion else in_channels

        # Context generation
        self.context_gen = ContextGen(in_channels, context_dim, expanded_channels)

        # Expansion 1x1 conv (blocks 1-14)
        if has_expansion:
            self.exp_conv = DynamicConv(in_channels * expanded_channels, k=k, context_dim=context_dim)
            self.exp_norm = nn.BatchNorm2d(expanded_channels)

        # Depthwise conv
        self.depth_conv = DynamicConv(depth_channels * kernel_size * kernel_size, k=k, context_dim=context_dim)
        self.depth_norm = nn.BatchNorm2d(depth_channels)

        # Dynamic activation
        self.depth_act = DyReLUB(depth_channels, context_dim)

        # Projection 1x1 conv
        self.proj_conv = DynamicConv(depth_channels * out_channels, k=k, context_dim=context_dim)
        self.proj_norm = nn.BatchNorm2d(out_channels)

    def _apply_dynamic_1x1(self, x: torch.Tensor, flat_w: torch.Tensor,
                           in_c: int, out_c: int) -> torch.Tensor:
        """Apply per-sample 1x1 convolution using grouped conv trick."""
        B, C, H, W = x.shape
        # Reshape: merge batch into channels for grouped conv
        x_grouped = x.reshape(1, B * C, H, W)
        w = flat_w.reshape(B * out_c, in_c, 1, 1)
        out = F.conv2d(x_grouped, w, groups=B)
        return out.reshape(B, out_c, H, W)

    def _apply_dynamic_dw(self, x: torch.Tensor, flat_w: torch.Tensor,
                          channels: int, kernel_size: int, stride: int) -> torch.Tensor:
        """Apply per-sample depthwise convolution using grouped conv trick."""
        B, C, H, W = x.shape
        padding = (kernel_size - 1) // 2
        x_grouped = x.reshape(1, B * C, H, W)
        w = flat_w.reshape(B * channels, 1, kernel_size, kernel_size)
        out = F.conv2d(x_grouped, w, padding=padding, stride=stride, groups=B * channels)
        return out.reshape(B, channels, out.size(2), out.size(3))

    def forward(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        identity = x
        B = x.size(0)

        # Generate context and gating signals
        context, f_gate, t_gate = self.context_gen(x)

        if self.has_expansion:
            # Expansion
            exp_w = self.exp_conv(context, temperature)
            out = self._apply_dynamic_1x1(x, exp_w, self.in_channels, self.expanded_channels)
            out = self.exp_norm(out)
            # Coordinate attention gating
            out = out * torch.sigmoid(f_gate) * torch.sigmoid(t_gate)
        else:
            out = x
            # Apply gating on the input directly for block 0
            out = out * torch.sigmoid(f_gate) * torch.sigmoid(t_gate)

        # Depthwise
        depth_channels = self.expanded_channels if self.has_expansion else self.in_channels
        dw_w = self.depth_conv(context, temperature)
        out = self._apply_dynamic_dw(out, dw_w, depth_channels, self.kernel_size, self.stride)
        out = self.depth_norm(out)
        out = self.depth_act(out, context)

        # Projection
        proj_w = self.proj_conv(context, temperature)
        out = self._apply_dynamic_1x1(out, proj_w, depth_channels, self.out_channels)
        out = self.proj_norm(out)

        # Residual connection
        if self.use_res_connect:
            out = out + identity

        return out


class DyMN(nn.Module):
    """Dynamic MobileNet for audio classification.

    Uses dynamic convolutions, DyReLU-B activations, and coordinate attention
    instead of the static convolutions and SE blocks of standard MobileNetV3.

    The forward() method returns (logits, features) matching the MN interface.
    """

    # Block configs: (in_c, exp_c, out_c, kernel, stride)
    # Same channel layout as MobileNetV3-Large
    BLOCK_CONFIGS = [
        (16,   16,   16,  3, 1),   # block 0: no expansion
        (16,   64,   24,  3, 2),
        (24,   72,   24,  3, 1),
        (24,   72,   40,  5, 2),
        (40,  120,   40,  5, 1),
        (40,  120,   40,  5, 1),
        (40,  240,   80,  3, 2),
        (80,  200,   80,  3, 1),
        (80,  184,   80,  3, 1),
        (80,  184,   80,  3, 1),
        (80,  480,  112,  3, 1),
        (112, 672,  112,  3, 1),
        (112, 672,  160,  5, 2),
        (160, 960,  160,  5, 1),
        (160, 960,  160,  5, 1),
    ]

    def __init__(self, num_classes: int = 527, width_mult: float = 1.0,
                 dropout: float = 0.2, k: int = 4, **kwargs):
        super().__init__()

        def adj(c):
            return make_divisible(c * width_mult, 8)

        first_conv_out = adj(16)

        # Input convolution
        self.in_c = nn.Sequential(
            nn.Conv2d(1, first_conv_out, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(first_conv_out),
            nn.Hardswish(),
        )

        # Dynamic blocks
        layers = []
        for i, (in_c, exp_c, out_c, kernel, stride) in enumerate(self.BLOCK_CONFIGS):
            in_c = adj(in_c)
            exp_c = adj(exp_c)
            out_c = adj(out_c)

            has_expansion = (i > 0)
            eff_exp = exp_c if has_expansion else in_c

            # Context dimension: min(128, max(32, make_divisible(exp * 0.25, 8)))
            context_dim = min(128, max(32, make_divisible(eff_exp * 0.25, 8)))

            layers.append(DyMNBlock(
                in_channels=in_c,
                expanded_channels=exp_c,
                out_channels=out_c,
                kernel_size=kernel,
                stride=stride,
                context_dim=context_dim,
                k=k,
                has_expansion=has_expansion,
            ))
        self.layers = nn.Sequential(*layers)

        last_conv_in = adj(160)
        last_conv_out = adj(960)
        last_channel = adj(1280)

        # Output convolution
        self.out_c = nn.Sequential(
            nn.Conv2d(last_conv_in, last_conv_out, kernel_size=1, bias=False),
            nn.BatchNorm2d(last_conv_out),
            nn.Hardswish(),
        )

        # Classifier head (MLP, matches MN)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
            nn.Linear(last_conv_out, last_channel),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(last_channel, num_classes),
        )

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.in_c(x)

        # Pass through dynamic blocks
        for layer in self.layers:
            x = layer(x)

        x = self.out_c(x)

        # Global average pool for features
        features = F.adaptive_avg_pool2d(x, (1, 1)).squeeze()
        logits = self.classifier(x).squeeze()

        if features.dim() == 1 and logits.dim() == 1:
            features = features.unsqueeze(0)
            logits = logits.unsqueeze(0)

        return logits, features

    def set_temperature(self, temperature: float):
        """Set temperature for all DynamicConv modules (for training annealing)."""
        self._temperature = temperature
