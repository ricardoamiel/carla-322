import os
from torch.utils.data import DataLoader
import torch
import torchvision.transforms as transforms
import torch.nn as nn
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
dataset = L.CarlaDataset(h5_files[:4], seq_len=5, transform=transform)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Model, loss, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model = M.CNNLSTM().to(device)
model = M.CNNAttentionLSTM().to(device)
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Checkpoint setup
best_loss = float('inf')
save_dir = os.getcwd() + "/checkpoints"
os.makedirs(save_dir, exist_ok=True)

# Training loop
for epoch in range(5):
    model.train()
    total_loss = 0
    total_loss_steer = 0
    total_loss_gas = 0
    total_loss_brake = 0
    for img_seq, cmd, speed, targets in dataloader:
        img_seq, cmd, speed, targets = img_seq.to(device), cmd.to(device), speed.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(img_seq, cmd, speed)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Calcular loss por cada salida
        loss_steer = criterion(outputs[:, 0], targets[:, 0])
        loss_gas = criterion(outputs[:, 1], targets[:, 1])
        loss_brake = criterion(outputs[:, 2], targets[:, 2])
        total_loss_steer += loss_steer.item()
        total_loss_gas += loss_gas.item()
        total_loss_brake += loss_brake.item()

    avg_loss = total_loss / len(dataloader)
    avg_loss_steer = total_loss_steer / len(dataloader)
    avg_loss_gas = total_loss_gas / len(dataloader)
    avg_loss_brake = total_loss_brake / len(dataloader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f} | Steer: {avg_loss_steer:.4f} | Gas: {avg_loss_gas:.4f} | Brake: {avg_loss_brake:.4f}")

    # Save checkpoint
    torch.save(model.state_dict(), os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pt"))

    # Save best model
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))
        print(f"✅ Best model updated at epoch {epoch+1} with loss {best_loss:.4f}")