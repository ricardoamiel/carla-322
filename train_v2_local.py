import os
from torch.utils.data import DataLoader
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import model_v2 as M2
import loader2 as L2

# Transform for input images
transform = transforms.Compose([
    transforms.Resize((88, 200)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.ToTensor()
])

# Load H5 files
print("🔍 Buscando archivos H5...")
h5_dir = os.getcwd() + "/data/SeqTrain/"
h5_files = [os.path.join(h5_dir, f) for f in os.listdir(h5_dir) if f.endswith('.h5')][:5]
print(f"✅ Se encontraron {len(h5_files)} archivos H5")

dataset = L2.CarlaDataset(h5_files, seq_len=5, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
print(f"✅ Dataset cargado con {len(dataloader)} batches")

# Model, loss, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Usando dispositivo: {device}")
model = M2.CNNAttentionLSTM(pretrained=True).to(device)
print("✅ Modelo cargado y movido a dispositivo")

criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Checkpoint setup
best_loss = float('inf')
save_dir = os.getcwd() + "/checkpoints_local"
os.makedirs(save_dir, exist_ok=True)

# Training loop
print("🏁 Comenzando entrenamiento...")
for epoch in range(10):
    model.train()
    total_loss = 0
    total_loss_steer = 0
    total_loss_gas = 0
    total_loss_brake = 0

    print(f"\n📚 Epoch {epoch+1}/10")
    for i, (img_seq, cmd, speed, targets) in enumerate(dataloader):
        img_seq = img_seq.to(device)
        cmd = cmd.to(device)
        speed = speed.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(img_seq, cmd, speed)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        with torch.no_grad():
            total_loss_steer += criterion(outputs[:, 0], targets[:, 0]).item()
            total_loss_gas += criterion(outputs[:, 1], targets[:, 1]).item()
            total_loss_brake += criterion(outputs[:, 2], targets[:, 2]).item()

        if i % 10 == 0:
            print(f"   Batch {i}/{len(dataloader)} - Loss: {loss.item():.4f}")

    # Epoch statistics
    num_batches = len(dataloader)
    print(f"📊 Epoch {epoch+1} terminada - "
          f"Loss: {total_loss/num_batches:.4f} | "
          f"Steer: {total_loss_steer/num_batches:.4f} | "
          f"Gas: {total_loss_gas/num_batches:.4f} | "
          f"Brake: {total_loss_brake/num_batches:.4f}")

    # Save checkpoint
    checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pt")
    torch.save(model.state_dict(), checkpoint_path)

    if (total_loss / num_batches) < best_loss:
        best_loss = total_loss / num_batches
        best_model_path = os.path.join(save_dir, "best_model.pt")
        torch.save(model.state_dict(), best_model_path)
        print(f"💾 ✅ Mejor modelo actualizado (Loss: {best_loss:.4f})")
