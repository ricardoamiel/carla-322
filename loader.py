from torch.utils.data import Dataset
import torch
import numpy as np
import h5py
from PIL import Image

class CarlaDataset(Dataset):
    def __init__(self, h5_files, seq_len=5, transform=None):
        self.seq_len = seq_len
        self.transform = transform
        self.data = []

        for file in h5_files:
            with h5py.File(file, 'r') as f:#     H    W  #im  C       
                images = f['rgb'][:]  # shape: (200, 88, 200, 3)
                targets = f['targets'][:]       # shape: (200, 28) #im, C

                for i in range(len(images) - seq_len): # 200 - 5 = 195
                    img_seq = images[i:i+seq_len] # tomar de 5 frames en 5
                    steer, gas, brake = targets[i+seq_len-1, 0:3] # 0,1,2
                    cmd = int(targets[i+seq_len-1, 23]) #high level command 2 Follow lane, 3 Left, 4 Right, 5 Straight)
                    speed = targets[i+seq_len-1, 10] # speed
                    self.data.append((img_seq, cmd, speed, [steer, gas, brake]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_seq, cmd, speed, targets = self.data[idx]
        img_seq_tensor = []

        for img in img_seq:
            img = Image.fromarray(img.astype(np.uint8))
            if self.transform:
                img = self.transform(img)
            img_seq_tensor.append(img)

        img_seq_tensor = torch.stack(img_seq_tensor)  # (seq_len, C, H, W)
        cmd_index = cmd - 2
        if cmd_index < 0 or cmd_index > 3:
            cmd_index = 0  # o salta este sample, o usa vector neutral

        cmd_tensor = torch.nn.functional.one_hot(torch.tensor(cmd_index), num_classes=4).float()
        # 2(1,0,0,0), 3(0,1,0,0), 4(0,0,1,0), 5(0,0,0,1): 

        speed_tensor = torch.tensor([speed]).float()
        targets_tensor = torch.tensor(targets).float()

        return img_seq_tensor, cmd_tensor, speed_tensor, targets_tensor