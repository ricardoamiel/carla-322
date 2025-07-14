import os
import time
from torch.utils.data import DataLoader
import torch
from torchvision.transforms import v2
import torch.nn as nn
import torch.amp as amp
import model as M
import loader as L

total_epochs = 200
seq_len = 16

# Transform for input images (working with tensors directly)
transform = v2.Compose([
    v2.Resize((88, 200)),
    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    # Normalize using calculated dataset statistics
    v2.Normalize(mean=[0.2632695734500885, 0.26334860920906067, 0.2629428505897522],
                 std=[0.1732328087091446, 0.17342117428779602, 0.17333601415157318])
])

# Load H5 files
h5_dir = os.getcwd() + "/data/SeqTrain/"
h5_files = [os.path.join(h5_dir, f) for f in os.listdir(h5_dir) if f.endswith('.h5')]
dataset = L.CarlaDataset(h5_files, seq_len=seq_len, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=8,
                        pin_memory=True, persistent_workers=True)

# Model, loss, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = M.CNNAttentionLSTM(
    cnn_output_dim=1280,
    lstm_hidden=256,
    fc_hidden=64
).to(device)

# Compile model for better performance
model = torch.compile(model)

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)
scaler = amp.GradScaler()

# Checkpoint directory
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# Resume training function
def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_loss = checkpoint['best_loss']
    print(f"Resuming from epoch {start_epoch}, best loss: {best_loss:.4f}")
    return start_epoch, best_loss

# Save checkpoint function
def save_checkpoint(epoch, loss, is_best=False):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'best_loss': loss,
        'lr': optimizer.param_groups[0]['lr']
    }
    
    # Save regular checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
    torch.save(checkpoint, checkpoint_path)
    
    # Save best model
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_model.pt')
        torch.save(checkpoint, best_path)
        print(f"✅ New best model saved! Loss: {loss:.4f}")

# Try to resume from checkpoint
start_epoch = 0
best_loss = float('inf')
last_checkpoint = os.path.join(checkpoint_dir, 'best_model.pt')

if os.path.exists(last_checkpoint):
    try:
        start_epoch, best_loss = load_checkpoint(last_checkpoint)
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        print("Starting from scratch...")

# Training loop
model.train()

for epoch in range(start_epoch, total_epochs):
    epoch_start_time = time.time()
    total_loss = 0
    total_loss_steer = 0
    total_loss_gas = 0
    total_loss_brake = 0
    num_batches = 0
    
    for batch_idx, (img_seq, cmd, speed, targets) in enumerate(dataloader):
        # Non-blocking device transfer
        img_seq = img_seq.to(device, non_blocking=True)
        cmd = cmd.to(device, non_blocking=True)
        speed = speed.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        
        # Use bfloat16 autocast for better performance
        with amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            outputs = model(img_seq, cmd, speed)
            loss = criterion(outputs, targets)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Accumulate losses
        total_loss += loss.item()
        num_batches += 1
        
        # Individual component losses for detailed statistics
        with torch.no_grad():
            total_loss_steer += criterion(outputs[:, 0], targets[:, 0]).item()
            total_loss_gas += criterion(outputs[:, 1], targets[:, 1]).item()
            total_loss_brake += criterion(outputs[:, 2], targets[:, 2]).item()
        
        if batch_idx % 10 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
    
    # Epoch statistics
    epoch_time = time.time() - epoch_start_time
    avg_loss = total_loss / num_batches
    avg_steer_loss = total_loss_steer / num_batches
    avg_gas_loss = total_loss_gas / num_batches
    avg_brake_loss = total_loss_brake / num_batches
    current_lr = optimizer.param_groups[0]['lr']
    
    print(f"\n" + "="*70)
    print(f"EPOCH {epoch+1}/{total_epochs} SUMMARY")
    print(f"="*70)
    print(f"Time: {epoch_time:.2f}s | LR: {current_lr:.6f}")
    print(f"Total Loss: {avg_loss:.4f}")
    print(f"Steer Loss: {avg_steer_loss:.4f}")
    print(f"Gas Loss:   {avg_gas_loss:.4f}")
    print(f"Brake Loss: {avg_brake_loss:.4f}")
    print(f"="*70 + "\n")
    
    # Update scheduler
    scheduler.step()
    
    # Save checkpoint
    is_best = avg_loss < best_loss
    if is_best:
        best_loss = avg_loss
    
    # Save checkpoint every 10 epochs or if it's the best model
    if (epoch + 1) % 10 == 0 or is_best:
        save_checkpoint(epoch, avg_loss, is_best)
    
    # Save final checkpoint
    if epoch == total_epochs - 1:
        save_checkpoint(epoch, avg_loss, False)

print(f"\nTraining completed! Best loss: {best_loss:.4f}")
print(f"Model checkpoints saved in: {checkpoint_dir}")