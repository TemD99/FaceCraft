import sys
if '/app' not in sys.path:
    sys.path.append('/app')

from config import *
from dataloader import get_loader, CelebDataset
from models.generator import Generator
from models.mapping_network import MappingNetwork
from utils import generate_examples, load_checkpoint, get_noise, get_w, save_checkpoint, ProjectTextFeatures
from Python.data.SG2.training.losses import PathLengthPenalty
from Python.data.SG2.models.discriminator import ProjectedDiscriminator
from Python.data.SG2.models.CLIP import CLIP
import matplotlib.pyplot as plt
import os
import nltk
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from Python.data.SG2.training.losses import ProjectedGANLoss
from tqdm import tqdm
from Python.data.SG2.zip_utils import ImageFolderDataset
import psutil
import time
import copy
import json
import PIL.Image
import numpy as np
import torch.utils.tensorboard as tensorboard
from train import main

# Ensure you have the WordNet data
def setup_nltk():
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('punkt')
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def main():

    TRAIN = True
    EVAL = False

    # random_seed = 42
    # img_resolution = 1024
    # z_dim = 512
    # c_dim = 512
    # total_kimg = 25000
    # batch_size = 4
    # ema_kimg = 10
    # ema_rampup = 0.05
    # kimg_per_tick = 4
    # image_snapshot_ticks = 50
    # network_snapshot_ticks = 50


    # # Initialize networks
    # gen = Generator(z_dim=z_dim, conditional=True, img_resolution=img_resolution).to(DEVICE)
    # G_ema = copy.deepcopy(gen).eval().to(DEVICE)
    # disc = ProjectedDiscriminator(c_dim=c_dim).to(DEVICE)
    # mapping_network = MappingNetwork(z_dim=z_dim, conditional=True).to(DEVICE)

    # # Initialize optimizers
    # opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    # opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    # opt_mapping_network = optim.Adam(mapping_network.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

    # # Define the transformation for the dataset
    # transform = transforms.Compose([
    #     transforms.Resize((256, 256)),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.5]*3, [0.5]*3)
    # ])

    # # Load dataset
    # dataset = ImageFolderDataset(DATASET)
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # # Loss function
    # loss_fn = ProjectedGANLoss(device=DEVICE, G=gen, D=disc)

    # writer = tensorboard.SummaryWriter('runs/experiment_1')

    # # Training setup
    # start_time = time.time()
    # np.random.seed(random_seed)
    # torch.manual_seed(random_seed)
    # torch.backends.cudnn.benchmark = True
    # torch.backends.cuda.matmul.allow_tf32 = False
    # torch.backends.cudnn.allow_tf32 = False

    # cur_nimg = 0
    # cur_tick = 0
    # tick_start_nimg = cur_nimg
    # tick_start_time = time.time()
    # maintenance_time = tick_start_time - start_time
    # batch_idx = 0

    # # Training loop
    # for epoch in range(EPOCHS):
    #     for i, (real_img, labels) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")):
    #         real_img = real_img.to(DEVICE)
    #         batch_size = real_img.size(0)
    #         z = torch.randn(batch_size, z_dim).to(DEVICE)

    #         labels = labels.to(DEVICE)

    #         # Generate fake images
    #         fake_img = gen(z, c=labels)
    #         opt_disc.zero_grad()
    #         real_logits = disc(real_img, c=labels)
    #         fake_logits = disc(fake_img.detach(), c=labels)
    #         d_loss_real = F.relu(1.0 - real_logits).mean()
    #         d_loss_fake = F.relu(1.0 + fake_logits).mean()
    #         d_loss = d_loss_real + d_loss_fake
    #         d_loss.backward()
    #         opt_disc.step()

    #         opt_gen.zero_grad()
    #         fake_logits = disc(fake_img)
    #         g_loss = -fake_logits.mean()
    #         g_loss.backward()
    #         opt_gen.step()

    #         cur_nimg += batch_size
    #         batch_idx += 1

    #         # Log metrics
    #         writer.add_scalar('Loss/Discriminator', d_loss.item(), epoch * len(dataloader) + i)
    #         writer.add_scalar('Loss/Generator', g_loss.item(), epoch * len(dataloader) + i)


    #         if i % 100 == 0:
    #             print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i}/{len(dataloader)}], D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")

    #     if epoch % 10 == 0 or epoch == EPOCHS - 1:
    #         save_checkpoint(gen, disc, opt_gen, opt_disc, epoch, "checkpoints", f"epoch_{epoch}")

    #     generate_examples(gen, epoch, mapping_network)

    #     # Update G_ema
    #     ema_nimg = ema_kimg * 1000
    #     if ema_rampup is not None:
    #         ema_nimg = min(ema_nimg, cur_nimg * ema_rampup)
    #     ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))
    #     for p_ema, p in zip(G_ema.parameters(), gen.parameters()):
    #         p_ema.data.copy_(p.data.lerp(p_ema.data, ema_beta))
    #     for b_ema, b in zip(G_ema.buffers(), gen.buffers()):
    #         b_ema.copy_(b)

    #     # Update state
    #     cur_nimg += batch_size
    #     batch_idx += 1

    #     tick_end_time = time.time()
    #     fields = []
    #     fields += [f"tick {cur_tick:<5d}"]
    #     fields += [f"kimg {cur_nimg / 1e3:<8.1f}"]
    #     fields += [f"time {time.strftime('%H:%M:%S', time.gmtime(tick_end_time - start_time)):<12s}"]
    #     fields += [f"sec/tick {tick_end_time - tick_start_time:<7.1f}"]
    #     fields += [f"sec/kimg {(tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3:<7.2f}"]
    #     fields += [f"maintenance {maintenance_time:<6.1f}"]
    #     fields += [f"cpumem {psutil.Process(os.getpid()).memory_info().rss / 2**30:<6.2f}"]
    #     fields += [f"gpumem {torch.cuda.max_memory_allocated(DEVICE) / 2**30:<6.2f}"]
    #     fields += [f"reserved {torch.cuda.max_memory_reserved(DEVICE) / 2**30:<6.2f}"]
    #     torch.cuda.reset_peak_memory_stats()
    #     print(' '.join(fields))

    #     if cur_tick and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
    #         continue

    #     cur_tick += 1
    #     tick_start_nimg = cur_nimg
    #     tick_start_time = time.time()
    #     maintenance_time = tick_start_time - tick_end_time

    # writer.close()
    # print("Training completed.")

    if TRAIN:
        main()

if __name__ == "__main__":
    #setup_nltk()
    main()
