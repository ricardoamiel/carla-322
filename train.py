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
    transforms.ToTensor()
])

# Load H5 files
h5_dir = os.getcwd() + "/deep_learning_proyecto/data/SeqTrain/"
h5_files = [os.path.join(h5_dir, f) for f in os.listdir(h5_dir) if f.endswith('.h5')]
dataset = L.CarlaDataset(h5_files[:2], seq_len=5, transform=transform)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Model, loss, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = M.CNNLSTM().to(device)
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Checkpoint setup
best_loss = float('inf')
save_dir = os.getcwd() + "/deep_learning_proyecto/checkpoints"
os.makedirs(save_dir, exist_ok=True)

# Training loop
for epoch in range(5):
    model.train()
    total_loss = 0
    for img_seq, cmd, speed, targets in dataloader:
        img_seq, cmd, speed, targets = img_seq.to(device), cmd.to(device), speed.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(img_seq, cmd, speed)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    # Save checkpoint
    torch.save(model.state_dict(), os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pt"))

    # Save best model
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))
        print(f"✅ Best model updated at epoch {epoch+1} with loss {best_loss:.4f}")