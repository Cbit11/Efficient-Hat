import torch
from torch.distributed import all_reduce, ReduceOp, barrier
from basicsr.metrics.psnr_ssim import calculate_psnr_pt, calculate_ssim_pt
import torch
from torch.distributed import all_reduce, ReduceOp, barrier
from basicsr.metrics.psnr_ssim import calculate_psnr_pt, calculate_ssim_pt

def train_step(model, loss_fn, optimizer, dataloader, device, epoch, rank=0, world_size=1):
    model.train()
    
    # Initialize as a tensor on the correct device
    train_loss = torch.tensor(0.0, device=device)

    for batch, data in enumerate(dataloader):
        optimizer.zero_grad()
        gt = data['GT'].to(device, non_blocking=True)
        lr = data['LR'].to(device, non_blocking=True)

        pred = model(lr)
        loss = loss_fn(pred, gt)

        loss.backward()
        optimizer.step()

        # Accumulate loss
        train_loss += loss.detach()

    # Get local average
    train_loss_avg = train_loss / len(dataloader)
    
    # Get global average across all GPUs
    all_reduce(train_loss_avg, op=ReduceOp.AVG)
    
    # Only rank 0 prints
    if rank == 0:
        print(f"[Epoch {epoch}] Train Loss: {train_loss_avg.item():.4f}")
    
    # ALL ranks must return the value
    return train_loss_avg.item()

def validation_step(model, loss_fn, dataloader, device, epoch=0, rank=0, world_size=1):
    model.eval()
    
    # Initialize metrics as tensors on the correct device
    val_loss = torch.tensor(0.0, device=device)
    total_psnr = torch.tensor(0.0, device=device)
    total_ssim = torch.tensor(0.0, device=device)

    with torch.no_grad():
        for batch, data in enumerate(dataloader):
            gt = data['GT'].to(device, non_blocking=True)
            lr = data['LR'].to(device, non_blocking=True)
            pred = model(lr)

            loss = loss_fn(pred, gt)
            # Make sure these functions return tensors
            psnr = calculate_psnr_pt(pred, gt, crop_border=0, test_y_channel=True)
            ssim = calculate_ssim_pt(pred, gt, crop_border=0, test_y_channel=True)

            val_loss += loss.detach()
            total_psnr += psnr.detach()
            total_ssim += ssim.detach()

    # Compute local averages
    val_loss_avg = val_loss / len(dataloader)
    psnr_avg = total_psnr / len(dataloader)
    ssim_avg = total_ssim / len(dataloader)

    # Reduce across all GPUs (get global average)
    all_reduce(val_loss_avg, op=ReduceOp.AVG)
    all_reduce(psnr_avg, op=ReduceOp.AVG)
    all_reduce(ssim_avg, op=ReduceOp.AVG)

    if rank == 0:
        print(f"[Validation] Loss: {val_loss_avg.item():.4f}, PSNR: {psnr_avg.item():.2f}, SSIM: {ssim_avg.item():.4f}")
    
    # ALL ranks must return to prevent a TypeError
    return val_loss_avg.item(), psnr_avg.item(), ssim_avg.item()