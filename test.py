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
import cv2
from basicsr.utils.img_util import tensor2img

file_pth = "/home/cj/new_network_modified/options/Efficient_hat_X2.yaml"

config = parse_from_yaml(file_pth)

test_data_pth = config['datasets']['val']['dataroot_gt']

device= config['device'] if torch.cuda.is_available() else 'cpu'
hat= HAT(in_chans=config['network_g']['in_chans'], embed_dim=config['network_g']['embed_dims'], depths= config['network_g']['hat_depth'],
         num_heads=config['network_g']['hat_num_heads'], window_size= config['network_g']['window_size'],compress_ratio=config['network_g']['compress_ratio'],
         squeeze_factor=config['network_g']['squeeze_factor'],conv_scale=config['network_g']['conv_scale'],overlap_ratio=config['network_g']['overlap_ratio'],
         mlp_ratio=config['network_g']['mlp_ratio'],upscale=config['network_g']['upscale'],resi_connection='1conv',upsampler='pixelshuffle')
chkpt= config['checkpoints']
test_data = Validation_dataset(test_data_pth, 4, 256)
test_loader= DataLoader(test_data, batch_size=1)
hat.to(device)
hat.eval()
with torch.no_grad():
    for batch, images in enumerate(test_loader):
        hr_tensor= images['GT'].to(device)
        lr_tensor= images['LR'].to(device)
        sr_tensor= hat(lr_tensor)
        sr_image= tensor2img(sr_tensor, rgb2bgr=False)
        hr_image= tensor2img(hr_tensor, rgb2bgr=False)

        cv2.imwrite(f"/home/cj/new_network_modified/results/gt_img_{batch}.png", hr_image)
        cv2.imwrite(f"/home/cj/new_network_modified/results/sr_img_{batch}.png", sr_image)
    