import os
from torch.utils.data import DataLoader
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.amp as amp
import model as M
import loader as L

# Transform for input images
transform = transforms.Compose([
    transforms.Resize((88, 200)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # added
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)), # added
    transforms.ToTensor()
])

# Load H5 files
h5_dir = os.getcwd() + "/data/SeqTrain/"
h5_files = [os.path.join(h5_dir, f) for f in os.listdir(h5_dir) if f.endswith('.h5')]
dataset = L.CarlaDataset(h5_files, seq_len=16, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=8
                        , pin_memory=True, persistent_workers=True)


# Model, loss, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model = M.CNNLSTM().to(device)
model = M.CNNAttentionLSTM().to(device)
model = torch.compile(model)
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scaler = amp.GradScaler("cuda")

# Checkpoint setup
best_loss = float('inf')
save_dir = os.getcwd() + "/checkpoints"
os.makedirs(save_dir, exist_ok=True)

# Training loop
for epoch in range(300):
    model.train()
    total_loss = 0
    total_loss_steer = 0
    total_loss_gas = 0
    total_loss_brake = 0
    for img_seq, cmd, speed, targets in dataloader:
        img_seq = img_seq.to(device, non_blocking=True)
        cmd = cmd.to(device, non_blocking=True)
        speed = speed.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            outputs = model(img_seq, cmd, speed)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        # Individual losses (with autocast disabled for loss calculation)
        with torch.no_grad():
            total_loss_steer += criterion(outputs[:, 0], targets[:, 0]).item()
            total_loss_gas += criterion(outputs[:, 1], targets[:, 1]).item()
            total_loss_brake += criterion(outputs[:, 2], targets[:, 2]).item()

    # Epoch statistics
    num_batches = len(dataloader)
    print(f"Epoch {epoch+1}: "
          f"Loss: {total_loss/num_batches:.4f} | "
          f"Steer: {total_loss_steer/num_batches:.4f} | "
          f"Gas: {total_loss_gas/num_batches:.4f} | "
          f"Brake: {total_loss_brake/num_batches:.4f}")

    # Save checkpoint
    checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pt")
    torch.save(model.state_dict(), checkpoint_path)

    # Save best model
    epoch_loss = total_loss / num_batches
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        best_model_path = os.path.join(save_dir, "best_model.pt")
        torch.save(model.state_dict(), best_model_path)
        print(f"✅ Best model updated (Loss: {best_loss:.4f})")