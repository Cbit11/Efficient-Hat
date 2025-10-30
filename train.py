import torch
from torch.distributed import all_reduce, ReduceOp, barrier
from basicsr.metrics.psnr_ssim import calculate_psnr_pt, calculate_ssim_pt
from torch.utils.tensorboard import SummaryWriter

# create writer only on rank 0
writer = None

def train_step(model, loss_fn, optimizer, scheduler, dataloader, device, epoch, rank=0, world_size=1):
    if rank == 0:
        print(f"[Rank {rank}] Training on {device}")

    model.train()
    train_loss = 0.0

    for batch, data in enumerate(dataloader):
        optimizer.zero_grad()
        gt = data['GT'].to(device, non_blocking=True)
        lr = data['LR'].to(device, non_blocking=True)

        pred = model(lr)
        loss = loss_fn(pred, gt)

        loss.backward()
        optimizer.step()

        # Accumulate loss (as tensor)
        train_loss += loss.detach()

    # Average loss across GPUs
    train_loss = train_loss / len(dataloader)
    
    train_loss_tensor =train_loss.detach().clone().to(device)
    if rank == 0:
        print("Starting pre-ALLREDUCE sync...")

    # Force all CUDA operations to complete
    torch.cuda.synchronize() 
    
    # ALL ranks MUST reach and pass this barrier at the same time
    # The rank that is slow will cause the others to wait here.
    barrier() 
    
    if rank == 0:
        print("Pre-ALLREDUCE sync passed. Running ALLREDUCE.")
    all_reduce(train_loss_tensor, op=ReduceOp.AVG)
    avg_loss = train_loss_tensor.item()

    # Step scheduler once per epoch
    scheduler.step()

    # Only rank 0 logs
    if rank == 0 and writer is not None:
        writer.add_scalar("Loss/train", avg_loss, epoch)
        print(f"[Epoch {epoch}] Train Loss: {avg_loss:.4f}")

    return avg_loss


def validation_step(model, loss_fn, dataloader, device, epoch=0, rank=0, world_size=1):
    model.eval()
    val_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0

    with torch.no_grad():
        for batch, data in enumerate(dataloader):
            gt = data['GT'].to(device, non_blocking=True)
            lr = data['LR'].to(device, non_blocking=True)
            pred = model(lr)

            loss = loss_fn(pred, gt)
            psnr = calculate_psnr_pt(pred, gt, crop_border=0, test_y_channel=True)
            ssim = calculate_ssim_pt(pred, gt, crop_border=0, test_y_channel=True)

            val_loss += loss.detach()
            total_psnr += psnr
            total_ssim += ssim

    # Compute local averages
    val_loss /= len(dataloader)
    total_psnr /= len(dataloader)
    total_ssim /= len(dataloader)

    # Reduce across all GPUs (get global average)
    val_loss_t = val_loss.detach().clone().to(device)
    psnr_t = total_psnr.detach().clone().to(device)
    ssim_t = total_ssim.detach().clone().to(device)
    
    if rank == 0:
        print("Starting pre-ALLREDUCE sync...")

    # Force all CUDA operations to complete
    torch.cuda.synchronize() 
    
    # ALL ranks MUST reach and pass this barrier at the same time
    # The rank that is slow will cause the others to wait here.
    barrier() 
    
    if rank == 0:
        print("Pre-ALLREDUCE sync passed. Running ALLREDUCE.")
        
    all_reduce(val_loss_t, op=ReduceOp.AVG)
    all_reduce(psnr_t, op=ReduceOp.AVG)
    all_reduce(ssim_t, op=ReduceOp.AVG)

    if rank == 0:
        print(f"[Validation] Loss: {val_loss_t.item():.4f}, PSNR: {psnr_t.item():.2f}, SSIM: {ssim_t.item():.4f}")
        if writer is not None:
            writer.add_scalar("Validation/Loss", val_loss_t.item(), epoch)
            writer.add_scalar("Validation/PSNR", psnr_t.item(), epoch)
            writer.add_scalar("Validation/SSIM", ssim_t.item(), epoch)

    return val_loss_t.item(), psnr_t.item(), ssim_t.item()
