import os
import torch
import torch.nn as nn
import numpy as np
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from tqdm import tqdm
from basicsr.losses.basic_loss import *
from basicsr.metrics.psnr_ssim import calculate_psnr_pt, calculate_ssim_pt
from data.Custom_image_dataset import Train_dataset, Validation_dataset
from arch.Efficient_HAT import HAT
from data.data_util import parse_from_yaml
from utils import get_loss, get_optimizer, get_scheduler, resume_training
from torch.utils.tensorboard import SummaryWriter
from train import train_step, validation_step
import argparse

parser = argparse.ArgumentParser(description='Imagenet dataset distributed data parallel test')

parser.add_argument('--init_method', default='tcp://127.0.0.1:3456', type=str, help='')
parser.add_argument('--dist-backend', default='nccl', type=str, help='')
parser.add_argument('--world_size', default=1, type=int, help='')
parser.add_argument('--distributed', action='store_true', help='')
# ---------- Main training ----------
def main(rank, world_size):
    args = parser.parse_args()
    local_rank = int(os.environ.get("SLURM_LOCALID")) 
    rank = int(os.environ.get("SLURM_PROCID"))
    current_device = local_rank

    torch.cuda.set_device(current_device)
    init_process_group(backend=args.dist_backend, init_method=args.init_method, world_size=args.world_size, rank=rank)
    # Load config
    file_pth = "CHANGE HERE"
    config = parse_from_yaml(file_pth)

    train_data_pth = config['datasets']['train']['dataroot_gt']
    val_data_pth = config['datasets']['val']['dataroot_gt']
    gt_size = config["datasets"]['train']["gt_size"]  # adjust if in YAML under 'train'
    epochs = config['train']['epoch']

    device = torch.device(f"cuda:{rank}")

    # Model
    hat = HAT(
        in_chans=config['network_g']['in_chans'],
        embed_dim=config['network_g']['embed_dims'],
        depths=config['network_g']['hat_depth'],
        num_heads=config['network_g']['hat_num_heads'],
        window_size=config['network_g']['window_size'],
        compress_ratio=config['network_g']['compress_ratio'],
        squeeze_factor=config['network_g']['squeeze_factor'],
        conv_scale=config['network_g']['conv_scale'],
        overlap_ratio=config['network_g']['overlap_ratio'],
        mlp_ratio=config['network_g']['mlp_ratio'],
        upscale=config['network_g']['upscale'],
        resi_connection='1conv',
        upsampler='pixelshuffle'
    ).to(device)

    if config['resume_training']:
        hat = resume_training(config, hat)
        print(f"[Rank {rank}] Resuming training")

    hat = DDP(hat, device_ids=[rank])

    # Loss, optimizer, scheduler
    loss_fn = get_loss(config)
    optimizer = get_optimizer(config, hat.parameters())
    lr_scheduler = get_scheduler(config, hat.parameters())

    # Datasets and loaders
    train_dataset = Train_dataset(train_data_pth, scale=4, gt_size=gt_size)
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['train_dataloader']['batch_size'],
        shuffle= config['train_dataloader']['shuffle'],
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )

    val_loader = None
    if val_data_pth is not None:
        val_dataset = Validation_dataset(val_data_pth, scale=4, gt_size=gt_size)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['val_dataloader']['batch_size'],
            shuffle= config['train_dataloader']['shuffle'],
            sampler=val_sampler,
            num_workers=4,
            pin_memory=True
        )

    # TensorBoard (rank 0 only)
    writer = SummaryWriter() if rank == 0 else None

    # Training loop
    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)
        train_step(hat, loss_fn, optimizer, lr_scheduler, train_loader, device, epoch)

        if rank == 0 and epoch % 10 == 0:
            if val_loader is not None:
                validation_step(hat, loss_fn, val_loader, device)
            torch.save(
                hat.module.state_dict(),
                f"CHANGE HERE"
            )

    if writer:
        writer.close()

    cleanup()


# ---------- Launch ----------
if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
