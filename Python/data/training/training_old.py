import sys
from tqdm import tqdm
import torch
from Python.data.SG2.utils import *
from Python.data.SG2.config import *
from .losses import gradient_penalty
import time
import psutil
import dnnlib
from torch_utils import training_stats, misc, distributed as dist
from torch.utils.tensorboard import SummaryWriter
if '/app' not in sys.path:
    sys.path.append('/app')

# Define the training function
def train_fn(
    critic, gen, clip_model, loader, opt_critic, opt_gen, opt_mapping_network,
    mapping_network, curr_epoch, project_text_features, run_dir, total_epochs=50, resume=False, snapshot_interval=10,
):
    gen.train()
    critic.train()
    mapping_network.train()
    clip_model.eval()

    start_time = time.time()
    cur_nimg = curr_epoch * len(loader) * loader.batch_size
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    batch_idx = 0

    writer = SummaryWriter(log_dir=run_dir)
    stats_collector = training_stats.Collector(regex='.*')

    for epoch in range(curr_epoch, total_epochs):
        loop = tqdm(loader, leave=True)
        for batch_idx, (real_images, real_attributes) in enumerate(loop):
            real_images = real_images.to(DEVICE)
            real_attributes = real_attributes.to(DEVICE)

            text_features = process_text(clip_model, real_attributes, project_text_features)

            # Generate w and noise
            noise = torch.randn(real_images.size(0), 512).to(DEVICE)
            w = get_w(real_images.size(0), mapping_network, noise, project_text_features)

            with torch.cuda.amp.autocast():
                fake_images = gen(w, text_features, real_attributes)
                critic_fake = critic(fake_images.detach(), real_attributes)
                critic_real = critic(real_images, real_attributes)
                gp = gradient_penalty(critic, real_images, fake_images, device=DEVICE)
                loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake)) + gp

                opt_critic.zero_grad()
                loss_critic.backward()
                opt_critic.step()

            if batch_idx % 5 == 0:
                gen_fake = critic(fake_images, real_attributes)
                loss_gen = -torch.mean(gen_fake)

                opt_gen.zero_grad()
                opt_mapping_network.zero_grad()
                loss_gen.backward()
                opt_gen.step()
                opt_mapping_network.step()

            loop.set_description(f"Epoch [{epoch+1}/{total_epochs}]")
            loop.set_postfix(loss_critic=loss_critic.item(), loss_gen=loss_gen.item())

            # Log metrics
            writer.add_scalar('Loss/Discriminator', loss_critic.item(), epoch * len(loader) + batch_idx)
            writer.add_scalar('Loss/Generator', loss_gen.item(), epoch * len(loader) + batch_idx)

            cur_nimg += real_images.size(0)
            batch_idx += 1

        if epoch % snapshot_interval == 0 or epoch == total_epochs - 1:
            torch.save(gen.state_dict(), f'{run_dir}/generator_epoch_{epoch}.pth')
            torch.save(critic.state_dict(), f'{run_dir}/critic_epoch_{epoch}.pth')
            print(f"Saved model snapshots at epoch {epoch}")

        if resume:
            checkpoint = torch.load("path_to_checkpoint.pth")
            gen.load_state_dict(checkpoint['gen_state_dict'])
            critic.load_state_dict(checkpoint['critic_state_dict'])
            opt_gen.load_state_dict(checkpoint['opt_gen_state_dict'])
            opt_critic.load_state_dict(checkpoint['opt_critic_state_dict'])
            print("Resumed training from checkpoint.")

        # Update state
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {cur_tick:<5d}"]
        fields += [f"kimg {cur_nimg / 1e3:<8.1f}"]
        fields += [f"time {time.strftime('%H:%M:%S', time.gmtime(tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {tick_end_time - tick_start_time:<7.1f}"]
        fields += [f"sec/kimg {(tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3:<7.2f}"]
        fields += [f"maintenance {maintenance_time:<6.1f}"]
        fields += [f"cpumem {psutil.Process(os.getpid()).memory_info().rss / 2**30:<6.2f}"]
        fields += [f"gpumem {torch.cuda.max_memory_allocated(DEVICE) / 2**30:<6.2f}"]
        fields += [f"reserved {torch.cuda.max_memory_reserved(DEVICE) / 2**30:<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        print(' '.join(fields))

        if cur_tick and (cur_nimg < tick_start_nimg + 4 * 1000):
            continue

        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time

    writer.close()
    print("Training completed.")