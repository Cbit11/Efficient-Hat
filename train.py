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
from utils import get_loss, get_optimizer, get_scheduler
from basicsr.metrics.psnr_ssim import calculate_psnr_pt, calculate_ssim_pt
from torch.utils.tensorboard import SummaryWriter

writer= SummaryWriter()

def train_step(model, loss_fn, optimizer,scheduler ,dataloader, device, epoch):
    print(f'The model is training on {device}')
    train_loss = 0.
    model.to(device)
    for batch, data in enumerate(dataloader):
        optimizer.zero_grad()
        gt= data['GT'].to(device)
        lr= data['LR'].to(device)
        pred= model(lr)
        loss= loss_fn(pred,gt)
        train_loss+= loss
        loss.backward()
        optimizer.step()
    scheduler.step()
    train_loss/= len(dataloader)
    writer.add_scalar("Loss", train_loss, epoch)
    print(f" | Train loss:{train_loss:.2f}")
    
def validation_step(model, loss_fn, dataloader, device):
  val_loss= 0
  total_psnr= 0
  total_ssim= 0
  model.to(device)
  model.eval()
  with torch.no_grad():
    for batch, data in enumerate(dataloader):
      gt= data['GT'].to(device)
      lr= data['LR'].to(device)
      pred= model(lr)
      loss= loss_fn(pred,gt)
      val_loss+= loss
      psnr= calculate_psnr_pt(pred, gt,crop_border= 0,test_y_channel = True)
      ssim= calculate_ssim_pt(pred, gt, crop_border= 0,test_y_channel= True)
      total_psnr+= psnr
      total_ssim+= ssim
      
    total_psnr/= len(dataloader)
    total_ssim/= len(dataloader)
    val_loss/=  len(dataloader)
    print(f"Validation Loss: {val_loss} | PSNR: {total_psnr} | SSIM: {total_ssim}")
    writer.add_scalar("Validation PSNR", total_psnr)
    writer.add_scalar("Validation_SSIM", total_ssim)
    
