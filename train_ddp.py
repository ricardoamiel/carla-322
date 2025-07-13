
import os
import random
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision.transforms as transforms
import torch.amp as amp
import model_v2 as M2
import loader2 as L2

def setup_ddp():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return local_rank

def cleanup():
    dist.destroy_process_group()

def main():
    print("🚀 Entrenando con DDP + AMP en múltiples GPUs")

    local_rank = setup_ddp()

    # Transformaciones
    transform = transforms.Compose([
        transforms.Resize((88, 200)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor()
    ])

    h5_dir = os.getcwd() + "/data/SeqTrain/"
    h5_files = [os.path.join(h5_dir, f) for f in os.listdir(h5_dir) if f.endswith('.h5')]
    if local_rank == 0:
        print(f"✅ {len(h5_files)} archivos H5 encontrados")

    dataset = L2.CarlaDataset(h5_files, seq_len=16, transform=transform)
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=16, sampler=sampler,
                            num_workers=4, pin_memory=True, persistent_workers=True)

    model = M2.CNNAttentionLSTM(pretrained=True).to(local_rank)
    model = DDP(model, device_ids=[local_rank])
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = amp.GradScaler("cuda")

    save_dir = os.getcwd() + "/checkpoints_ddp"
    os.makedirs(save_dir, exist_ok=True)
    best_loss = float('inf')

    for epoch in range(100):
        model.train()
        sampler.set_epoch(epoch)

        total_loss = 0
        total_loss_steer = 0
        total_loss_gas = 0
        total_loss_brake = 0

        for batch_idx, (img_seq, cmd, speed, targets) in enumerate(dataloader):
            img_seq = img_seq.to(local_rank, non_blocking=True)
            cmd = cmd.to(local_rank, non_blocking=True)
            speed = speed.to(local_rank, non_blocking=True)
            targets = targets.to(local_rank, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(img_seq, cmd, speed)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            with torch.no_grad():
                total_loss_steer += criterion(outputs[:, 0], targets[:, 0]).item()
                total_loss_gas += criterion(outputs[:, 1], targets[:, 1]).item()
                total_loss_brake += criterion(outputs[:, 2], targets[:, 2]).item()

            if batch_idx % 100 == 0 and local_rank == 0:
                print(f"🔁 Epoch {epoch+1}, Batch {batch_idx}/{len(dataloader)} - "
                      f"Loss: {loss.item():.4f} | "
                      f"Steer: {total_loss_steer/(batch_idx+1):.4f} | "
                      f"Gas: {total_loss_gas/(batch_idx+1):.4f} | "
                      f"Brake: {total_loss_brake/(batch_idx+1):.4f}")

        avg_loss = total_loss / len(dataloader)
        if local_rank == 0:
            print(f"🔁 Epoch {epoch+1} - Loss: {avg_loss:.4f} | "
                  f"Steer: {total_loss_steer/len(dataloader):.4f} | "
                  f"Gas: {total_loss_gas/len(dataloader):.4f} | "
                  f"Brake: {total_loss_brake/len(dataloader):.4f}")

            torch.save(model.module.state_dict(), os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pt"))
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.module.state_dict(), os.path.join(save_dir, "best_model.pt"))
                print(f"✅ Best model actualizado (Loss: {best_loss:.4f})")

    cleanup()

if __name__ == "__main__":
    main()
