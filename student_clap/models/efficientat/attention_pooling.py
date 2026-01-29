import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .utils import collapse_dim


class MultiHeadAttentionPooling(nn.Module):
    """Multi-Head Attention as used in PSLA paper (https://arxiv.org/pdf/2102.01243.pdf)
    """
    def __init__(self, in_dim, out_dim, att_activation: str = 'sigmoid',
                 clf_activation: str = 'ident', num_heads: int = 4, epsilon: float = 1e-7):
        super(MultiHeadAttentionPooling, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.epsilon = epsilon

        self.att_activation = att_activation
        self.clf_activation = clf_activation

        # out size: out dim x 2 (att and clf paths) x num_heads
        self.subspace_proj = nn.Linear(self.in_dim, self.out_dim * 2 * self.num_heads)
        self.head_weight = nn.Parameter(torch.tensor([1.0 / self.num_heads] * self.num_heads).view(1, -1, 1))

    def activate(self, x, activation):
        if activation == 'linear':
            return x
        elif activation == 'relu':
            return F.relu(x)
        elif activation == 'sigmoid':
            return torch.sigmoid(x)
        elif activation == 'softmax':
            return F.softmax(x, dim=1)
        elif activation == 'ident':
            return x

    def forward(self, x) -> Tensor:
        """x: Tensor of size (batch_size, channels, frequency bands, sequence length)
        """
        x = collapse_dim(x, dim=2)  # results in tensor of size (batch_size, channels, sequence_length)
        x = x.transpose(1, 2)  # results in tensor of size (batch_size, sequence_length, channels)
        b, n, c = x.shape

        x = self.subspace_proj(x).reshape(b, n, 2, self.num_heads, self.out_dim).permute(2, 0, 3, 1, 4)
        att, val = x[0], x[1]
        val = self.activate(val, self.clf_activation)
        att = self.activate(att, self.att_activation)
        att = torch.clamp(att, self.epsilon, 1. - self.epsilon)
        att = att / torch.sum(att, dim=2, keepdim=True)

        out = torch.sum(att * val, dim=2) * self.head_weight
        out = torch.sum(out, dim=1)
        return out
