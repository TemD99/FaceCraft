
import torch
import torch.nn as nn
import numpy as np
from torch_utils.ops import bias_act
from torch_utils import misc
from typing import Union, Any
from .CLIP import CLIP
from typing import Callable

def is_list_of_strings(arr: Any) -> bool:
    if arr is None: return False
    is_list = isinstance(arr, list) or isinstance(arr, np.ndarray) or  isinstance(arr, tuple)
    entry_is_str = isinstance(arr[0], str)
    return is_list and entry_is_str


def normalize_2nd_moment(x: torch.Tensor, dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()


class ResidualBlock(nn.Module):
    def __init__(self, fn: Callable):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.fn(x) + x) / np.sqrt(2)


class FullyConnectedLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation: str = 'linear',
        lr_multiplier: float = 1.0,
        weight_init: float = 1.0,
        bias_init: float = 0.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) * (weight_init / lr_multiplier))
        bias_init = np.broadcast_to(np.asarray(bias_init, dtype=np.float32), [out_features])
        self.bias = torch.nn.Parameter(torch.from_numpy(bias_init / lr_multiplier)) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        # print(f"x: {x.shape}")  # Debug print
        # print(f"w: {w.shape}")  # Debug print
        if b is not None:
            # print(f"b: {b.shape}")  # Debug print
            pass

        if self.activation == 'linear' and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = bias_act.bias_act(x, b, act=self.activation)
        return x

    def extra_repr(self) -> str:
        return f'in_features={self.in_features:d}, out_features={self.out_features:d}, activation={self.activation:s}'



class MLP(nn.Module):
    def __init__(
        self,
        features_list: list[int],    # Number of features in each layer of the MLP.
        activation: str = 'linear',  # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier: float = 1.0,  # Learning rate multiplier.
        linear_out: bool = False     # Use the 'linear' activation function for the output layer?
    ):
        super().__init__()
        num_layers = len(features_list) - 1
        self.num_layers = num_layers
        self.out_dim = features_list[-1]

        for idx in range(num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            if linear_out and idx == num_layers-1:
                activation = 'linear'
            layer = FullyConnectedLayer(in_features, out_features, activation=activation, lr_multiplier=lr_multiplier)
            setattr(self, f'fc{idx}', layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ''' if x is sequence of tokens, shift tokens to batch and apply MLP to all'''
        shift2batch = (x.ndim == 3)

        if shift2batch:
            B, K, C = x.shape
            x = x.flatten(0,1)

        for idx in range(self.num_layers):
            layer = getattr(self, f'fc{idx}')
            x = layer(x)

        if shift2batch:
            x = x.reshape(B, K, -1)

        #(f"x after MLP: {x.shape}")  # Debug print
        return x

class MappingNetwork(torch.nn.Module):
    def __init__(
        self,
        z_dim: int,                   # Input latent (Z) dimensionality, 0 = no latent.
        conditional: bool = True,     
        num_layers: int = 2,          # Number of mapping layers.
        activation: str = 'lrelu',    # Activation function:'lrelu'
        lr_multiplier: float = 0.01,  # Learning rate multiplier for the mapping layers.
        x_avg_beta: float = 0.995,    # Decay for tracking the moving average of W during training.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.x_avg_beta = x_avg_beta
        self.num_ws = None

        self.mlp = MLP([z_dim]*(num_layers+1), activation=activation,
                       lr_multiplier=lr_multiplier, linear_out=True)

        if conditional:
            self.clip = CLIP()
            del self.clip.model.visual # only using the text encoder
            self.c_dim = self.clip.txt_dim
        else:
            self.c_dim = 0

        self.w_dim = self.c_dim + self.z_dim
        self.register_buffer('x_avg', torch.zeros([self.z_dim]))

    def forward(
        self,
        z: torch.Tensor,
        c: Union[None, torch.Tensor, list[str]],
        truncation_psi: float = 1.0,
    ) -> torch.Tensor:
        misc.assert_shape(z, [None, self.z_dim])

        # Forward pass.
        x = self.mlp(normalize_2nd_moment(z))

        # Update moving average.
        if self.x_avg_beta is not None and self.training:
            self.x_avg.copy_(x.detach().mean(0).lerp(self.x_avg, self.x_avg_beta))

        # Apply truncation.
        if truncation_psi != 1:
            assert self.x_avg_beta is not None
            x = self.x_avg.lerp(x, truncation_psi)

        # Build latent.
        if self.c_dim > 0:
            assert c is not None
            c = self.clip.encode_text(c) if is_list_of_strings(c) else c
            w = torch.cat([x, c], 1)
        else:
            w = x

        # Broadcast latent codes.
        if self.num_ws is not None:
            w = w.unsqueeze(1).repeat([1, self.num_ws, 1])

        return w
