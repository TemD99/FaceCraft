import torch
from torch import nn
from torch.nn import functional as F
from .mapping_network import FullyConnectedLayer, MLP, MappingNetwork
from .CLIP import CLIP
from typing import Union, Any, Optional
from torch_utils import misc
from torch_utils.ops import upfirdn2d, bias_act
import numpy as np
from torch.nn.parameter import Parameter

class StyleSplit(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        super().__init__()
        self.proj = FullyConnectedLayer(in_channels, 3*out_channels, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        m1, m2, m3 = x.chunk(3, 1)
        return m1 * m2 + m3


class ModulatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, demodulate=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.demodulate = demodulate

    def forward(self, x, styles):
        batch_size = x.shape[0]
        weight = self.weight * styles[:, None, :, None, None]

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([1, 2, 3]) + 1e-8)
            weight = weight * demod[:, None, None, None]

        x = F.conv2d(x, weight.reshape(-1, *weight.shape[2:]), padding=1, groups=batch_size)
        return x


class ToRGBLayer(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        w_dim: int,
        kernel_size: int = 1,
        conv_clamp: Optional[int] = None,
        channels_last: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w_dim = w_dim
        self.conv_clamp = conv_clamp
        self.affine = StyleSplit(w_dim, in_channels, bias_init=1)

        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        self.weight = Parameter(0.1*torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format))
        self.bias = Parameter(torch.zeros([out_channels]))
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))

    def forward(self, x: torch.Tensor, w: torch.Tensor, fused_modconv: bool=True) -> torch.Tensor:
        styles = self.affine(w) * self.weight_gain
        x = ModulatedConv2d(x=x, weight=self.weight, styles=styles, demodulate=False)
        x = bias_act.bias_act(x, self.bias.to(x.dtype), clamp=self.conv_clamp)
        return x

    def extra_repr(self) -> str:
        return f'in_channels={self.in_channels:d}, out_channels={self.out_channels:d}, w_dim={self.w_dim:d}'

class SynthesisLayer(nn.Module):
    def __init__(self, in_channels, out_channels, w_dim, resolution, kernel_size=3, up=1, use_noise=True, activation='lrelu', resample_filter=[1,3,3,1], conv_clamp=None, channels_last=False, layer_scale_init=1e-5, residual=False, gn_groups=32):
        super().__init__()
        if residual: assert in_channels == out_channels

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.up = up
        self.use_noise = use_noise
        self.activation = activation
        self.conv_clamp = conv_clamp
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.act_gain = bias_act.activation_funcs[activation].def_gain

        self.residual = residual

        if use_noise:
            self.register_buffer('noise_const', torch.randn([resolution, resolution]))
            self.noise_strength = nn.Parameter(torch.zeros([]))

        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)

        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        self.weight = nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format))
        self.bias = nn.Parameter(torch.zeros([out_channels]))

        if self.residual:
            assert up == 1
            self.norm = nn.GroupNorm(gn_groups, out_channels)
            self.gamma = nn.Parameter(layer_scale_init * torch.ones([1, out_channels, 1, 1])).to(memory_format=memory_format)

    def forward(self, x, w, noise_mode='random', fused_modconv=True, gain=1):
        dtype = x.dtype
        in_resolution = self.resolution // self.up
        misc.assert_shape(x, [None, self.in_channels, in_resolution, in_resolution])

        noise = None
        if self.use_noise and noise_mode == 'random':
            noise = torch.randn([x.shape[0], 1, self.resolution, self.resolution], device=x.device) * self.noise_strength
        if self.use_noise and noise_mode == 'const':
            noise = self.noise_const * self.noise_strength

        flip_weight = (self.up == 1)
        styles = self.affine(w)

        if self.residual:
            x = self.norm(x)

        y = ModulatedConv2d(x=x, weight=self.weight, styles=styles, noise=noise, up=self.up, fused_modconv=fused_modconv, padding=self.padding, resample_filter=self.resample_filter, flip_weight=flip_weight)
        y = y.to(dtype)

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        y = bias_act.bias_act(y, self.bias.to(x.dtype), act=self.activation, gain=act_gain, clamp=act_clamp)

        if self.residual:
            y = self.gamma * y
            y = y.to(dtype).add_(x).mul(np.sqrt(2))

        return y

    def extra_repr(self):
        return f'in_channels={self.in_channels:d}, out_channels={self.out_channels:d}, w_dim={self.w_dim:d}, resolution={self.resolution:d}, up={self.up}, activation={self.activation:s}'


class SynthesisInput(nn.Module):
    def __init__(self, w_dim, channels, size, sampling_rate, bandwidth):
        super().__init__()
        self.w_dim = w_dim
        self.channels = channels
        self.size = np.broadcast_to(np.asarray(size), [2])
        self.sampling_rate = sampling_rate
        self.bandwidth = bandwidth

        freqs = torch.randn([self.channels, 2])
        radii = freqs.square().sum(dim=1, keepdim=True).sqrt()
        freqs /= radii * radii.square().exp().pow(0.25)
        freqs *= bandwidth
        phases = torch.rand([self.channels]) - 0.5

        self.weight = nn.Parameter(torch.randn([self.channels, self.channels]))
        self.affine = FullyConnectedLayer(w_dim, 4, weight_init=0, bias_init=[1,0,0,0])

        self.register_buffer('transform', torch.eye(3, 3))
        self.register_buffer('freqs', freqs)
        self.register_buffer('phases', phases)

    def forward(self, w):
        transforms = self.transform.unsqueeze(0)
        freqs = self.freqs.unsqueeze(0)
        phases = self.phases.unsqueeze(0)

        t = self.affine(w)
        t = t / t[:, :2].norm(dim=1, keepdim=True)
        m_r = torch.eye(3, device=w.device).unsqueeze(0).repeat([w.shape[0], 1, 1])
        m_r[:, 0, 0] = t[:, 0]
        m_r[:, 0, 1] = -t[:, 1]
        m_r[:, 1, 0] = t[:, 1]
        m_r[:, 1, 1] = t[:, 0]
        m_t = torch.eye(3, device=w.device).unsqueeze(0).repeat([w.shape[0], 1, 1])
        m_t[:, 0, 2] = -t[:, 2]
        m_t[:, 1, 2] = -t[:, 3]
        transforms = m_r @ m_t @ transforms

        phases = phases + (freqs @ transforms[:, :2, 2:]).squeeze(2)
        freqs = freqs @ transforms[:, :2, :2]

        amplitudes = (1 - (freqs.norm(dim=2) - self.bandwidth) / (self.sampling_rate / 2 - self.bandwidth)).clamp(0, 1)

        theta = torch.eye(2, 3, device=w.device)
        theta[0, 0] = 0.5 * self.size[0] / self.sampling_rate
        theta[1, 1] = 0.5 * self.size[1] / self.sampling_rate
        grids = F.affine_grid(theta.unsqueeze(0), [1, 1, self.size[1], self.size[0]], align_corners=False)

        x = (grids.unsqueeze(3) @ freqs.permute(0, 2, 1).unsqueeze(1).unsqueeze(2)).squeeze(3)
        x = x + phases.unsqueeze(1).unsqueeze(2)
        x = torch.sin(x * (np.pi * 2))
        x = x * amplitudes.unsqueeze(1).unsqueeze(2)

        weight = self.weight / np.sqrt(self.channels)
        x = x @ weight.t()

        x = x.permute(0, 3, 1, 2)
        misc.assert_shape(x, [w.shape[0], self.channels, int(self.size[1]), int(self.size[0])])
        return x.contiguous()


class SynthesisBlock(nn.Module):
    def __init__(self, in_channels, out_channels, w_dim, resolution, img_channels, is_last, num_res_blocks=1, architecture='orig', resample_filter=[1,3,3,1], conv_clamp=256, use_fp16=False, fp16_channels_last=False, fused_modconv_default='inference_only', **layer_kwargs):
        assert architecture in ['orig', 'skip']
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.is_last = is_last
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.fused_modconv_default = fused_modconv_default
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.num_conv = 0
        self.num_torgb = 0

        if in_channels == 0:
            self.input = SynthesisInput(w_dim=self.w_dim, channels=out_channels, size=resolution, sampling_rate=resolution, bandwidth=2)
            self.num_conv += 1

        if in_channels != 0:
            self.conv0 = SynthesisLayer(in_channels, out_channels, w_dim=w_dim, resolution=resolution, up=2, resample_filter=resample_filter, conv_clamp=conv_clamp, channels_last=self.channels_last, **layer_kwargs)
            self.num_conv += 1

        convs = []
        for _ in range(num_res_blocks):
            convs.append(SynthesisLayer(out_channels, out_channels, w_dim=w_dim, resolution=resolution, conv_clamp=conv_clamp, channels_last=self.channels_last, **layer_kwargs))
            convs.append(SynthesisLayer(out_channels, out_channels, w_dim=w_dim, resolution=resolution, conv_clamp=conv_clamp, channels_last=self.channels_last, residual=True, **layer_kwargs))

        self.convs1 = nn.ModuleList(convs)
        self.num_conv += len(convs)

        if is_last or architecture == 'skip':
            self.torgb = ToRGBLayer(out_channels, img_channels, w_dim=w_dim, conv_clamp=conv_clamp, channels_last=self.channels_last)
            self.num_torgb += 1

    def forward(self, x, img, ws, force_fp32=False, fused_modconv=True, **layer_kwargs):
        w_iter = iter(ws.unbind(dim=1))
        if ws.device.type != 'cuda':
            force_fp32 = True
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format
        if fused_modconv is None:
            fused_modconv = self.fused_modconv_default
        if fused_modconv == 'inference_only':
            fused_modconv = (not self.training)

        if self.in_channels == 0:
            x = self.input(next(w_iter))
        else:
            misc.assert_shape(x, [None, self.in_channels, self.resolution // 2, self.resolution // 2])
        x = x.to(dtype=dtype, memory_format=memory_format)

        if self.in_channels == 0:
            for conv in self.convs1:
                x = conv(x, next(w_iter), fused_modconv=fused_modconv, gain=np.sqrt(0.5), **layer_kwargs)
        else:
            x = self.conv0(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
            for conv in self.convs1:
                x = conv(x, next(w_iter), fused_modconv=fused_modconv, gain=np.sqrt(0.5), **layer_kwargs)

        if img is not None:
            misc.assert_shape(img, [None, self.img_channels, self.resolution // 2, self.resolution // 2])
            img = upfirdn2d.upsample2d(img, self.resample_filter)

        if self.is_last or self.architecture == 'skip':
            y = self.torgb(x, next(w_iter), fused_modconv=fused_modconv)
            y = y.to(dtype=torch.float32, memory_format=torch.contiguous_format)
            img = img.add_(y) if img is not None else y

        assert x.dtype == dtype
        assert img is None or img.dtype == torch.float32
        return x, img

    def extra_repr(self):
        return f'resolution={self.resolution:d}, architecture={self.architecture:s}'

    

class SynthesisNetwork(torch.nn.Module):
    def __init__(
        self,
        w_dim: int,                 # Intermediate latent (W) dimensionality.
        img_resolution: int,        # Output image resolution.
        img_channels: int = 3,      # Number of color channels.
        channel_base: int = 32768,  # Overall multiplier for the number of channels.
        channel_max: int = 512,     # Maximum number of channels in any layer.
        num_fp16_res: int = 4,      # Use FP16 for the N highest resolutions.
        base_mult: int = 3,         # Start resolution (SG2: 2, SG3: 4, SG-T: 3).
        num_res_blocks: int = 3,    # Number of residual blocks.
        **block_kwargs,             # Arguments for SynthesisBlock.
    ):
        assert img_resolution >= 4 and img_resolution & (img_resolution - 1) == 0
        super().__init__()
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.num_fp16_res = num_fp16_res
        self.block_resolutions = [2 ** i for i in range(base_mult, self.img_resolution_log2 + 1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        self.num_ws = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res // 2] if res > self.block_resolutions[0] else 0
            out_channels = channels_dict[res]
            use_fp16 = (res >= fp16_resolution)
            is_last = (res == self.img_resolution)
            block = SynthesisBlock(in_channels, out_channels, w_dim=w_dim, resolution=res, img_channels=img_channels, is_last=is_last, use_fp16=use_fp16, num_res_blocks=num_res_blocks, **block_kwargs)
            self.num_ws += block.num_conv
            if is_last:
                self.num_ws += block.num_torgb
            setattr(self, f'b{res}', block)

    def forward(self, ws: torch.Tensor, **block_kwargs) -> torch.Tensor:
        block_ws = []
        with torch.autograd.profiler.record_function('split_ws'):
            ws = ws.to(torch.float32)
            w_idx = 0
            for res in self.block_resolutions:
                block = getattr(self, f"b{res}")
                block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                w_idx += block.num_conv

        x = img = None
        for res, cur_ws in zip(self.block_resolutions, block_ws):
            block = getattr(self, f'b{res}')
            x, img = block(x, img, cur_ws, **block_kwargs)
        return img

    def extra_repr(self) -> str:
        return ' '.join([
            f'w_dim={self.w_dim:d}, num_ws={self.num_ws:d},',
            f'img_resolution={self.img_resolution:d}, img_channels={self.img_channels:d},',
            f'num_fp16_res={self.num_fp16_res:d}'])



class Generator(nn.Module):
    def __init__(self, z_dim: int, conditional: bool, img_resolution: int, img_channels: int = 3, train_mode: str = 'all', synthesis_kwargs: dict = {}):
        super().__init__()
        self.z_dim = z_dim
        self.conditional = conditional
        self.img_resolution = img_resolution
        self.img_channels = img_channels

        self.mapping = MappingNetwork(z_dim=z_dim, conditional=conditional)
        self.synthesis = SynthesisNetwork(w_dim=self.mapping.w_dim, img_resolution=img_resolution, img_channels=img_channels, **synthesis_kwargs)

        self.w_dim = self.synthesis.w_dim
        self.c_dim = self.mapping.c_dim
        self.num_ws = self.synthesis.num_ws
        self.mapping.num_ws = self.num_ws

        # Set trainable layers
        self.train_mode = train_mode
        if train_mode == 'all':
            self.trainable_layers = ['synthesis', 'mapping.mlp']
        elif train_mode == 'text_encoder':
            self.trainable_layers = ['clip']
        elif train_mode == 'freeze64':
            self.trainable_layers = [f"b{x}" for x in self.synthesis.block_resolutions if x > 64]
            self.trainable_layers += ['torgb']

    def forward(self, z: torch.Tensor, c: Union[None, torch.Tensor, list[str]], truncation_psi: float = 1.0, **synthesis_kwargs) -> torch.Tensor:
        if self.conditional and c is None:
            raise ValueError("Conditional generator requires labels `c`.")
        ws = self.mapping(z, c, truncation_psi=truncation_psi)
        img = self.synthesis(ws, **synthesis_kwargs)
        return img