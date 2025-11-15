import numpy as np 
from torch.utils.data import Dataset
from basicsr.utils import img2tensor
import h5py 

class dataset(Dataset):
    def __init__(self, file_pth, indices_pth):
        super(dataset, self).__init__()
        self.file_pth = file_pth 
        
        # 1. Load indices ONCE (in the main process)
        self.indices = np.load(indices_pth)
        
        # 2. Set file handle to None. Each worker will 
        #    create its own.
        self.file = None
        self.hr_imgs = None
        self.lr_imgs = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        
        # 3. This 'if' check is the key
        #    It runs ONCE PER WORKER
        if self.file is None:
            self.file = h5py.File(self.file_pth, "r")
            self.hr_imgs = self.file["HR_Image_256"]
            self.lr_imgs = self.file["LR_Image_X4"] 
        
        # 4. Get the true index from the list (already in memory)
        true_index = self.indices[index]
        
        # 5. Read from the worker's open file handle
        img_gt = self.hr_imgs[true_index]
        img_lq = self.lr_imgs[true_index]
        
        img_gt, img_lq = img2tensor([img_gt, img_lq], float32=True)
        return {"GT": img_gt / 255., "LR": img_lq / 255.}