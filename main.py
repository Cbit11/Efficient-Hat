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
from data.Custom_image_dataset import dataset
from arch.Efficient_HAT import HAT
from data.data_util import parse_from_yaml
from utils import get_loss, get_optimizer, get_scheduler, resume_training
from torch.utils.tensorboard import SummaryWriter
from train import train_step, validation_step
import datetime
import wandb
# ---------- Main training ----------
def main():
    local_rank = int(os.environ.get("SLURM_LOCALID")) 
    rank = int(os.environ.get("SLURM_PROCID"))
    num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    world_size = int(os.environ.get("SLURM_NTASKS"))
    current_device = local_rank
    torch.cuda.set_device(current_device)
    init_process_group(backend='nccl', world_size=world_size, rank=rank, timeout= datetime.timedelta(seconds=7200))
    # Load config
    file_pth = "/home/cjrathod/projects/def-mhassanz/cjrathod/Efficient-Hat/options/Efficient_hat_X2.yaml"
    config = parse_from_yaml(file_pth)

    train_pth = config['datasets']['train']['file_pth']
    val_data_pth = config['datasets']['val']['file_pth']
    epochs = config['train']['epoch']

    device = torch.device(f"cuda:{local_rank}")

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
        if rank == 0:
            print(f"Resuming training")
    hat = DDP(hat, device_ids=[current_device])

    # Loss, optimizer, scheduler
    loss_fn = get_loss(config)
    optimizer = get_optimizer(config, hat.parameters())
    lr_scheduler = get_scheduler(config, hat.parameters())

    # Datasets and loaders
    train_dataset = dataset(train_pth)
    train_sampler = DistributedSampler(train_dataset, shuffle=config['train_dataloader']['shuffle'])
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['train_dataloader']['batch_size'],
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last= True
    )

    val_loader = None
    if val_data_pth is not None:
        val_dataset = dataset(val_data_pth)
        val_sampler = DistributedSampler(val_dataset, shuffle=config['train_dataloader']['shuffle'] )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['val_dataloader']['batch_size'],
            sampler=val_sampler,
            num_workers=num_workers,
            pin_memory=True, 
            drop_last= True
        )

    if rank ==0:
        run = wandb.init(project= "SR_Model", config= config, mode= "offline") 
        
    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)
        train_loss= train_step(hat, loss_fn, optimizer, train_loader, device, epoch, rank ,  world_size)
        if rank == 0 :
            run.log({"Train loss": train_loss})
        if epoch % 10 == 0:
            if val_loader is not None:
                # ALL processes run validation
                val_loss, PSNR, SSIM = validation_step(hat, loss_fn, val_loader, device, epoch, rank,world_size)
                
                # ONLY rank 0 logs and saves
                if rank == 0:
                    run.log({"Validation loss": val_loss, 
                             "PSNR": PSNR, 
                             "SSIM": SSIM})
                    torch.save(
                        hat.module.state_dict(),
                        config['checkpoint'] + f"/Imagenet_{epoch}.pth"
                    )

            elif rank == 0: 
                torch.save(
                    hat.module.state_dict(),
                    config['checkpoint'] + f"/Imagenet_{epoch}.pth"
                )
        lr_scheduler.step()
    if rank == 0:
        wandb.finish()
    destroy_process_group()
# ---------- Launch ----------
if __name__ == "__main__":
    main()
