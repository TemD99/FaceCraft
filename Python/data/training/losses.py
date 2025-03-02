
# __init__.py
import sys
if '/app' not in sys.path:
    sys.path.append('/app')
import torch
from config import *
from torch import nn
from math import sqrt
from torch.nn import functional as F
from torch_utils import training_stats
from torch_utils.ops import upfirdn2d
from torchvision.transforms import RandomCrop
from models.CLIP import CLIP
import numpy as np

# Perceptual path length normalization
#    encourages a fixed-size step in w to result in a fixed-magnitude change in the image
class PathLengthPenalty(nn.Module):
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.steps = nn.Parameter(torch.tensor(0.), requires_grad=False)
        self.exp_sum_a = nn.Parameter(torch.tensor(0.), requires_grad=False)

    def forward(self, w, x):
        image_size = x.shape[2] * x.shape[3]
        y = torch.randn(x.shape, device=x.device)

        output = (x * y).sum() / sqrt(image_size)

        gradients, *_ = torch.autograd.grad(
            outputs=output,
            inputs=w,
            grad_outputs=torch.ones(output.shape, device=w.device),
            create_graph=True
        )

        norm = (gradients ** 2).sum(dim=2).mean(dim=1).sqrt()

        if self.steps > 0:
            a = self.exp_sum_a / (1 - self.beta ** self.steps)
            loss = torch.mean((norm - a) ** 2)
        else:
            loss = norm.new_tensor(0)

        mean = norm.mean().detach()
        self.exp_sum_a.mul_(self.beta).add_(mean, alpha=1 - self.beta)
        self.steps.add_(1.)

        return loss
    
# gradient_penalty
#    for WGAN-GP loss
def gradient_penalty(critic, real, fake):
    BATCH_SIZE, C, H, W = real.shape
    beta = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(real.device)
    interpolated_images = real * beta + fake.detach() * (1 - beta)
    interpolated_images.requires_grad_(True)

    mixed_scores = critic(interpolated_images)
 
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty


def clip_loss(image_features, text_features):
    image_features = F.normalize(image_features, p=2, dim=1)
    text_features = F.normalize(text_features, p=2, dim=1)
    loss = 1 - (image_features * text_features).sum(dim=1)
    return loss.mean()

def semantic_loss(real_img_features, fake_img_features, text_features):
    real_similarity = torch.cosine_similarity(real_img_features, text_features)
    fake_similarity = torch.cosine_similarity(fake_img_features, text_features)
    return torch.mean(fake_similarity - real_similarity)

class ProjectedGANLoss:
    def __init__(self, device, G, D, blur_init_sigma=2, blur_fade_kimg=0, clip_weight=0.0, r1_gamma = None):
        super().__init__()
        self.device = device
        self.G = G
        self.D = D
        self.blur_init_sigma = blur_init_sigma
        self.blur_curr_sigma = blur_init_sigma
        self.blur_fade_kimg = blur_fade_kimg
        self.train_text_encoder = 'clip' in G.trainable_layers
        self.clip = CLIP().eval().to(self.device).requires_grad_(False)
        self.clip_weight = clip_weight

    @staticmethod
    def spherical_distance(x, y):
        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)
        return (x * y).sum(-1).arccos().pow(2)

    @staticmethod
    def blur(img, blur_sigma):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img.device).div(blur_sigma).square().neg().exp2()
                img = upfirdn2d.filter2d(img, f / f.sum())
        return img

    def set_blur_sigma(self, cur_nimg):
        if self.blur_fade_kimg > 1:
            self.blur_curr_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma
        else:
            self.blur_curr_sigma = 0

    def run_G(self, z, c):
        ws = self.G.mapping(z, c)
        img = self.G.synthesis(ws)
        return img

    def run_D(self, img, c):
        if img.size(-1) > self.G.img_resolution:
            img = F.interpolate(img, self.G.img_resolution, mode='area')
        img = self.blur(img, self.blur_curr_sigma)
        return self.D(img, c)

    def accumulate_gradients(self, phase, real_img, c_raw, gen_z, cur_nimg):
        batch_size = real_img.size(0)
        self.set_blur_sigma(cur_nimg)

        c_enc = None
        if isinstance(c_raw[0], str):
            c_enc = self.clip.encode_text(c_raw)

        if phase == 'D':
            gen_img = self.run_G(gen_z, c=c_raw if self.train_text_encoder else c_enc)
            gen_logits = self.run_D(gen_img, c=c_enc)
            loss_gen = (F.relu(torch.ones_like(gen_logits) + gen_logits)).mean() / batch_size
            loss_gen.backward()

            real_img_tmp = real_img.detach().requires_grad_(False)
            real_logits = self.run_D(real_img_tmp, c=c_enc)
            loss_real = (F.relu(torch.ones_like(real_logits) - real_logits)).mean() / batch_size
            loss_real.backward()

            training_stats.report('Loss/scores/fake', gen_logits)
            training_stats.report('Loss/scores/real', real_logits)
            training_stats.report('Loss/signs/fake', gen_logits.sign())
            training_stats.report('Loss/signs/real', real_logits.sign())
            training_stats.report('Loss/D/loss', loss_gen + loss_real)

        elif phase == 'G':
            gen_img = self.run_G(gen_z, c=c_raw if self.train_text_encoder else c_enc)
            gen_logits = self.run_D(gen_img, c=c_enc)
            loss_gen = (-gen_logits).mean() / batch_size

            clip_loss = 0
            if self.clip_weight > 0:
                if gen_img.size(-1) > 64:
                    gen_img = RandomCrop(64)(gen_img)
                gen_img = F.interpolate(gen_img, 224, mode='area')
                gen_img_features = self.clip.encode_image(gen_img.add(1).div(2))
                clip_loss = self.spherical_distance(gen_img_features, c_enc).mean()

            (loss_gen + self.clip_weight * clip_loss).backward()

            training_stats.report('Loss/scores/fake', gen_logits)
            training_stats.report('Loss/signs/fake', gen_logits.sign())
            training_stats.report('Loss/G/loss', loss_gen)
            if self.clip_weight > 0:
                training_stats.report('Loss/G/clip_loss', clip_loss)

def normalize_2nd_moment(x: torch.Tensor, dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()
