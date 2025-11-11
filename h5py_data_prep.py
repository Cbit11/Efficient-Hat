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
from basicsr.utils.img_util import imwrite, tensor2img
import h5py
img_pth = "/home/cjrathod/scratch/datasets/DIV2K/DIV2K_train_HR"
output_pth = "/home/cjrathod/scratch/Paired_DIV2K_data.h5"
gt_size= 256
scale = 4
num_patches_image= 10
image_files =[f for f in os.listdir(img_pth) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
num_images= len(image_files)
HR_SHAPE = (0, gt_size, gt_size, 3) 
LR_SHAPE = (0, gt_size // scale, gt_size // scale, 3)

with h5py.File(output_pth, "w") as f:
    HR_CHUNK_SHAPE = (10, gt_size, gt_size, 3) 
    LR_CHUNK_SHAPE = (10, gt_size // scale, gt_size // scale, 3)
    # Create datasets and pre-allocate space. dtype=np.uint8 is usually better for images.
    hr_dset = f.create_dataset(
        "HR_Image_256", 
        shape = HR_SHAPE,
        dtype=np.uint8, # Use uint8 for image data
        maxshape=(None, gt_size, gt_size, 3),
        compression='gzip',
        compression_opts=4, # Lower compression level (1-9) for slightly faster writing
        chunks = HR_CHUNK_SHAPE
    )
    lr_dset = f.create_dataset(
        "LR_Image_X4", 
        shape=LR_SHAPE, 
        dtype=np.uint8,
        maxshape=(None, gt_size//scale, gt_size//scale, 3),
        compression='gzip',
        compression_opts=4,
        chunks = LR_CHUNK_SHAPE
    )
# --- 3. Process and Write Patches Incrementally ---
    print(f"Starting patch extraction...")
    
    current_idx = 0
    for i, img_name in enumerate(image_files):
        img_path = os.path.join(img_pth, img_name)
        try:
            # Load and ensure it's large enough (standard float preprocessing)
            # Use .convert('RGB') to handle grayscale/PNGs consistently
            gt_img = np.array(Image.open(img_path).convert('RGB'), dtype=np.float32)
        except Exception as e:
            print(f"Error loading {img_name}: {e}. Skipping.")
            continue
        size_h, size_w, _ = gt_img.shape
        
        # Skip image if it's too small for even one crop
        if size_h < gt_size or size_w < gt_size:
            continue

        # --- Patch Extraction Loop ---
        hr_patches_list = []
        lr_patches_list = []
        
        for _ in range(num_patches_image):
            # 1. Randomly choose crop starting points (or use a fixed grid)
            h_start = np.random.randint(0, size_h - gt_size + 1)
            w_start = np.random.randint(0, size_w - gt_size + 1)
            
            # 2. Extract HR patch
            hr_patch = gt_img[h_start:h_start + gt_size, w_start:w_start + gt_size, :]
            
            # 3. Create corresponding LR patch
            lr_patch = imresize(hr_patch, 1 / scale)
            
            # Convert back to uint8 for storage
            hr_patches_list.append(np.ascontiguousarray(hr_patch, dtype=np.uint8))
            lr_patches_list.append(np.ascontiguousarray(lr_patch, dtype=np.uint8))

        # Convert the collected patches for this single image into a NumPy array
        hr_batch = np.stack(hr_patches_list, axis=0)
        lr_batch = np.stack(lr_patches_list, axis=0)

        # --- 4. Dynamically Resize and Write ---
        # Extend the dataset by the number of patches we just generated
        new_size = current_idx + num_patches_image
        hr_dset.resize(new_size, axis=0)
        lr_dset.resize(new_size, axis=0)
        
        # Write the new patches to the end of the dataset
        hr_dset[current_idx:new_size, :, :, :] = hr_batch
        lr_dset[current_idx:new_size, :, :, :] = lr_batch
        
        current_idx = new_size # Update the index for the next write

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(image_files)} files. Total patches: {current_idx}")

print("Conversion Successful!")
print(f"Data saved to {output_pth}. Total patches stored: {current_idx}")