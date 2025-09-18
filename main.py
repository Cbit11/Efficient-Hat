import os
import torch
import torch.nn as nn
import torch.functional as f
import numpy as np
from basicsr.losses.basic_loss import *
from torch.utils.data import DataLoader, Dataset
from data.Custom_image_dataset import Train_dataset, Validation_dataset
from tqdm import tqdm
import yaml
from arch.Efficient_HAT import HAT
from data.data_util import parse_from_yaml
from utils import get_loss, get_optimizer, get_scheduler, resume_training
from basicsr.metrics.psnr_ssim import calculate_psnr_pt, calculate_ssim_pt
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from train import train_step, validation_step

file_pth = "/home/cj/new_network_modified/options/Efficient_hat_X2.yaml"

config = parse_from_yaml(file_pth)
train_data_pth = config['datasets']['train']['dataroot_gt']
val_data_pth = config['datasets']['val']['dataroot_gt']
device= config['device'] if torch.cuda.is_available() else 'cpu'
hat= HAT(in_chans=config['network_g']['in_chans'], embed_dim=config['network_g']['embed_dims'], depths= config['network_g']['hat_depth'],
         num_heads=config['network_g']['hat_num_heads'], window_size= config['network_g']['window_size'],compress_ratio=config['network_g']['compress_ratio'],
         squeeze_factor=config['network_g']['squeeze_factor'],conv_scale=config['network_g']['conv_scale'],overlap_ratio=config['network_g']['overlap_ratio'],
         mlp_ratio=config['network_g']['mlp_ratio'],upscale=config['network_g']['upscale'],resi_connection='1conv',upsampler='pixelshuffle')

loss= get_loss(config)
optimizer= get_optimizer(config, hat.parameters())
lr_scheduler= get_scheduler(config,  hat.parameters())
writer = SummaryWriter()
epochs = config['train']['epoch']
train_data= Train_dataset(train_data_pth, scale = 2)
val_data= Validation_dataset(val_data_pth, scale= 2)
train_loader= DataLoader(train_data, batch_size= config['train_dataloader']['batch_size'], shuffle = config['train_dataloader']['shuffle'])
val_loader= DataLoader(val_data, batch_size= config['val_dataloader']['batch_size'], shuffle = config['val_dataloader']['shuffle'])

if config['resume_training']:
  hat= resume_training(config, hat)
  print("Resuming Training")

for epoch in tqdm(range(epochs)):
      train_step(hat, loss, optimizer, lr_scheduler, train_loader, device, epoch)
      if epoch%10==0:
           validation_step(hat, loss, val_loader, device)
           torch.save(hat.state_dict(),f"/home/cj/new_network_modified/checkpoints/D2FK_{epoch}") 
           