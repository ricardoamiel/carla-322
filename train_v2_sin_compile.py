import os
from torch.utils.data import DataLoader
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.amp as amp
import model_v2 as M2
import loader2 as L2

print("🚀 Inicializando entrenamiento con AMP en Khipu")

# Transformaciones para imágenes
transform = transforms.Compose([
    transforms.Resize((88, 200)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.ToTensor()
])

# Cargar archivos H5
h5_dir = os.getcwd() + "/data/SeqTrain/"
h5_files = [os.path.join(h5_dir, f) for f in os.listdir(h5_dir) if f.endswith('.h5')]
print(f"✅ Se encontraron {len(h5_files)} archivos H5")

# Escoger aleatoriamente 500 archivos
import random
random.seed(42)
random.shuffle(h5_files)
h5_files = h5_files[:1500]
print(f"🔍 Cargando {len(h5_files)} archivos H5 para el dataset")

# Dataset y DataLoader
dataset = L2.CarlaDataset(h5_files, seq_len=16, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4,
                        pin_memory=True, persistent_workers=True)
print(f"✅ Dataset cargado con {len(dataloader)} batches")

# Modelo, loss, optimizador
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = M2.CNNAttentionLSTM(pretrained=True).to(device)
print("✅ Modelo cargado, preentrenado y movido a CUDA")

criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scaler = amp.GradScaler("cuda")

# Carpeta de checkpoints
save_dir = os.getcwd() + "/checkpoints_usando_amp_toda_la_data"
os.makedirs(save_dir, exist_ok=True)
best_loss = float('inf')

# Entrenamiento
for epoch in range(100):
    model.train()
    total_loss = 0
    total_loss_steer = 0
    total_loss_gas = 0
    total_loss_brake = 0
    print(f"\n📚 Epoch {epoch+1}/100 - {torch.cuda.get_device_name(0)}")
    
    for batch_idx, (img_seq, cmd, speed, targets) in enumerate(dataloader):
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

        with torch.no_grad():
            total_loss_steer += criterion(outputs[:, 0], targets[:, 0]).item()
            total_loss_gas += criterion(outputs[:, 1], targets[:, 1]).item()
            total_loss_brake += criterion(outputs[:, 2], targets[:, 2]).item()

        if batch_idx % 100 == 0:
            print(f"   Batch {batch_idx}/{len(dataloader)} - Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    print(f"🔁 Epoch {epoch+1} - Loss: {avg_loss:.4f} | "
          f"Steer: {total_loss_steer/len(dataloader):.4f} | "
          f"Gas: {total_loss_gas/len(dataloader):.4f} | "
          f"Brake: {total_loss_brake/len(dataloader):.4f}")

    # Guardar checkpoint
    torch.save(model.state_dict(), os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pt"))
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))
        print(f"✅ Best model actualizado (Loss: {best_loss:.4f})")
