from typing import Dict, Callable, List
import torch
import torch.nn as nn
from torch import Tensor
from torchvision.ops.misc import ConvNormActivation

from .utils import make_divisible, cnn_out_size


class ConcurrentSEBlock(torch.nn.Module):
    def __init__(
        self,
        c_dim: int,
        f_dim: int,
        t_dim: int,
        se_cnf: Dict
    ) -> None:
        super().__init__()
        dims = [c_dim, f_dim, t_dim]
        self.conc_se_layers = nn.ModuleList()
        for d in se_cnf['se_dims']:
            input_dim = dims[d-1]
            squeeze_dim = make_divisible(input_dim // se_cnf['se_r'], 8)
            self.conc_se_layers.append(SqueezeExcitation(input_dim, squeeze_dim, d))
        if se_cnf['se_agg'] == "max":
            self.agg_op = lambda x: torch.max(x, dim=0)[0]
        elif se_cnf['se_agg'] == "avg":
            self.agg_op = lambda x: torch.mean(x, dim=0)
        elif se_cnf['se_agg'] == "add":
            self.agg_op = lambda x: torch.sum(x, dim=0)
        elif se_cnf['se_agg'] == "min":
            self.agg_op = lambda x: torch.min(x, dim=0)[0]
        else:
            raise NotImplementedError(f"SE aggregation operation '{self.agg_op}' not implemented")

    def forward(self, input: Tensor) -> Tensor:
        # apply all concurrent se layers
        se_outs = []
        for se_layer in self.conc_se_layers:
            se_outs.append(se_layer(input))
        out = self.agg_op(torch.stack(se_outs, dim=0))
        return out


class SqueezeExcitation(torch.nn.Module):
    """
    This block implements the Squeeze-and-Excitation block from https://arxiv.org/abs/1709.01507.
    Args:
        input_dim (int): Input dimension
        squeeze_dim (int): Size of Bottleneck
        activation (Callable): activation applied to bottleneck
        scale_activation (Callable): activation applied to the output
    """

    def __init__(
        self,
        input_dim: int,
        squeeze_dim: int,
        se_dim: int,
        activation: Callable[..., torch.nn.Module] = torch.nn.ReLU,
        scale_activation: Callable[..., torch.nn.Module] = torch.nn.Sigmoid,
    ) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, squeeze_dim)
        self.fc2 = torch.nn.Linear(squeeze_dim, input_dim)
        assert se_dim in [1, 2, 3]
        self.se_dim = [1, 2, 3]
        self.se_dim.remove(se_dim)
        self.activation = activation()
        self.scale_activation = scale_activation()

    def _scale(self, input: Tensor) -> Tensor:
        scale = torch.mean(input, self.se_dim, keepdim=True)
        shape = scale.size()
        scale = self.fc1(scale.squeeze(2).squeeze(2))
        scale = self.activation(scale)
        scale = self.fc2(scale)
        scale = scale
        return self.scale_activation(scale).view(shape)

    def forward(self, input: Tensor) -> Tensor:
        scale = self._scale(input)
        return scale * input


class InvertedResidualConfig:
    # Stores information listed at Tables 1 and 2 of the MobileNetV3 paper
    def __init__(
        self,
        input_channels: int,
        kernel: int,
        expanded_channels: int,
        out_channels: int,
        use_se: bool,
        activation: str,
        stride: int,
        dilation: int,
        width_mult: float,
    ):
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.kernel = kernel
        self.expanded_channels = self.adjust_channels(expanded_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.use_se = use_se
        self.use_hs = activation == "HS"
        self.stride = stride
        self.dilation = dilation
        self.f_dim = None
        self.t_dim = None

    @staticmethod
    def adjust_channels(channels: int, width_mult: float):
        return make_divisible(channels * width_mult, 8)

    def out_size(self, in_size):
        padding = (self.kernel - 1) // 2 * self.dilation
        return cnn_out_size(in_size, padding, self.dilation, self.kernel, self.stride)


class InvertedResidual(nn.Module):
    def __init__(
        self,
        cnf: InvertedResidualConfig,
        se_cnf: Dict,
        norm_layer: Callable[..., nn.Module],
        depthwise_norm_layer: Callable[..., nn.Module]
    ):
        super().__init__()
        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: List[nn.Module] = []
        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU

        # expand
        if cnf.expanded_channels != cnf.input_channels:
            layers.append(
                ConvNormActivation(
                    cnf.input_channels,
                    cnf.expanded_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        # depthwise
        stride = 1 if cnf.dilation > 1 else cnf.stride
        layers.append(
            ConvNormActivation(
                cnf.expanded_channels,
                cnf.expanded_channels,
                kernel_size=cnf.kernel,
                stride=stride,
                dilation=cnf.dilation,
                groups=cnf.expanded_channels,
                norm_layer=depthwise_norm_layer,
                activation_layer=activation_layer,
            )
        )
        if cnf.use_se and se_cnf['se_dims'] is not None:
            layers.append(ConcurrentSEBlock(cnf.expanded_channels, cnf.f_dim, cnf.t_dim, se_cnf))

        # project
        layers.append(
            ConvNormActivation(
                cnf.expanded_channels, cnf.out_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=None
            )
        )

        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_channels
        self._is_cn = cnf.stride > 1

    def forward(self, inp: Tensor) -> Tensor:
        result = self.block(inp)
        if self.use_res_connect:
            result += inp
        return result
