import torch
import torch.nn as nn 
import yaml
from basicsr.losses.basic_loss import L1Loss
from PIL import Image
import numpy as np 
def get_loss(opt):
    loss= opt['train']['pixel_opt']['type']
    if loss == 'L1Loss':
        loss_fn= L1Loss(reduction='mean')
    return loss_fn

def get_optimizer(opt,prms):
    optimizer= opt['train']['optim_g']['type']
    lr= opt['train']['optim_g']['lr']
    if optimizer== 'Adam':
        optimizer= torch.optim.Adam(params= prms, lr= lr)
    return optimizer
    
def get_scheduler(opt,params):
    scheduler= opt['train']['scheduler']['type']
    milestones= opt['train']['scheduler']['milestones']
    optimizer= get_optimizer(opt,params)
    if scheduler== 'MultiStepLR':
        schdlr= torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones)
    return schdlr
    
def resume_training(config, model):
  chkpt = torch.load(config['checkpoint'])
  model.load_state_dict(chkpt)
  print("Model weights uploaded successfully")
  return model

       