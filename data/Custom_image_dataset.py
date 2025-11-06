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
import h5py 
class dataset(Dataset):
    def __init__(self,file_pth):
        super(dataset, self).__init__()
        self.file = h5py(file_pth, "r")
        self.hr_imgs= self.file[list(self.file.keys()[0])]
        self.lr_imgs= self.file[list(self.file.keys()[1])]
    def __len__(self):
        return len(self.hr_imgs)
    def __getitem__(self, index):
        img_gt = self.hr_imgs[index]
        img_lq = self.lr_imgs[index]
        img_gt, img_lq = img2tensor([img_gt, img_lq], float32=True)
        return {"GT":img_gt/255., "LR":img_lq/255.}

