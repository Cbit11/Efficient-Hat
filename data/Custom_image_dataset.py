import torch 
import numpy as np 
import os
import cv2
from torch.utils.data import Dataset, DataLoader
from basicsr.data.transforms import augment, paired_random_crop
from PIL import Image
from basicsr.utils.matlab_functions import imresize
from basicsr.utils.color_util import rgb2ycbcr
from basicsr.utils import FileClient, imfrombytes, img2tensor

class Train_dataset(Dataset):
    def __init__(self, hr_pth, scale, gt_size):
        super(Train_dataset, self).__init__()
        self.hr_pth = hr_pth
        self.hr_imgs= os.listdir(self.hr_pth)
        self.scale= scale
        self.gt_size= gt_size
    def __len__(self):
        return len(self.hr_imgs)
    def __getitem__(self, index):
        gt_img_pth = os.path.join(self.hr_pth, self.hr_imgs[index])
        gt_img = np.array(Image.open(gt_img_pth), dtype = np.float32)
        
        size_h, size_w, _ = gt_img.shape
        size_h = size_h - size_h % self.scale
        size_w = size_w - size_w % self.scale
        img_gt = gt_img[0:size_h, 0:size_w, :]

        # generate training pairs
        size_h = max(size_h, 128)
        size_w = max(size_w, 128)
        img_gt = cv2.resize(gt_img, (size_w, size_h))
        img_lq = imresize(img_gt, 1 / self.scale)

        img_gt = np.ascontiguousarray(img_gt, dtype=np.float32)
        img_lq = np.ascontiguousarray(img_lq, dtype=np.float32)
            # random crop
        img_gt, img_lq = paired_random_crop(img_gt, img_lq, self.gt_size, self.scale, self.hr_pth)
            # flip, rotation
        img_gt, img_lq = augment([img_gt, img_lq], True, True)
        # img_gt = rgb2ycbcr(img_gt)[..., None]
        # img_lq = rgb2ycbcr(img_lq)[..., None]
        img_gt, img_lq = img2tensor([img_gt, img_lq], float32=True)
        return {"GT":img_gt/255., "LR":img_lq/255.}

class Validation_dataset(Dataset):
    def __init__(self, hr_pth, scale, gt_size):
        super(Validation_dataset).__init__()
        self.hr_pth = hr_pth
        self.hr_imgs= os.listdir(self.hr_pth)
        self.scale= scale
        self.gt_size= gt_size
    def __len__(self):
        return len(self.hr_imgs)
    def __getitem__(self, index):
        gt_img_pth = os.path.join(self.hr_pth, self.hr_imgs[index])
        gt_img = np.array(Image.open(gt_img_pth), dtype = np.float32)
        
        size_h, size_w, _ = gt_img.shape
        size_h = size_h - size_h % self.scale
        size_w = size_w - size_w % self.scale
        img_gt = gt_img[0:size_h, 0:size_w, :]

        # generate training pairs
        size_h = max(size_h, 128)
        size_w = max(size_w, 128)
        img_gt = cv2.resize(gt_img, (size_w, size_h))
        img_lq = imresize(img_gt, 1 / self.scale)
        
        img_gt = np.ascontiguousarray(img_gt, dtype=np.float32)
        img_lq = np.ascontiguousarray(img_lq, dtype=np.float32)
        # random crop
        img_gt, img_lq = paired_random_crop(img_gt, img_lq, self.gt_size, self.scale, self.hr_pth)
        img_gt, img_lq = img2tensor([img_gt, img_lq], float32=True)
        return {"GT":img_gt/255., "LR":img_lq/255.}
