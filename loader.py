from torch.utils.data import Dataset
import torch
import numpy as np
import h5py

class CarlaDataset(Dataset):
    def __init__(self, h5_files, seq_len=5, transform=None, oversample=True):
        self.seq_len = seq_len
        self.transform = transform
        self.data = []

        for file in h5_files:
            try:
                with h5py.File(file, 'r') as f:
                    images = f['rgb'][:]  # shape: (200, 88, 200, 3)
                    targets = f['targets'][:]  # shape: (200, 28)

                    for i in range(len(images) - seq_len): # 200 - 5 = 195
                        img_seq = images[i:i+seq_len] # tomar de 5 frames en 5
                        steer, gas, brake = targets[i+seq_len-1, 0:3] # 0,1,2
                        # High-level command values in dataset: 2(Follow), 3(Left), 4(Right), 5(Straight)
                        cmd_raw = int(targets[i+seq_len-1, 24])
                        #cmd_idx = cmd_raw - 2  # Map to 0-3 for one-hot encoding
                        cmd_idx = cmd_raw  # esto es temporallll
                        speed = targets[i+seq_len-1, 10] # speed
                        sample = (img_seq, cmd_idx, speed, [steer, gas, brake])
                        self.data.append(sample)

                        # Oversample if command is Left (1) or Right (2) after mapping
                        # if oversample and cmd_idx in [1, 2]:
                        #     for _ in range(2):  # duplicate twice
                        #         self.data.append(sample)
            except OSError as e:
                print(f"corrupto:V : {file}")
                continue
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_seq, cmd, speed, targets = self.data[idx]
        
        # Convert numpy array directly to tensor (H, W, C) -> (C, H, W)
        img_seq_tensor = []
        for img in img_seq:
            img_tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
            if self.transform:
                img_tensor = self.transform(img_tensor)
            img_seq_tensor.append(img_tensor)

        img_seq_tensor = torch.stack(img_seq_tensor)  # (seq_len, C, H, W)
        
        cmd_tensor = torch.nn.functional.one_hot(torch.tensor(cmd), num_classes=6).float()
        # Mapping: 0->Follow lane(2), 1->Left(3), 2->Right(4), 3->Straight(5)
        
        speed_tensor = torch.tensor([speed]).float()
        targets_tensor = torch.tensor(targets).float()

        return img_seq_tensor, cmd_tensor, speed_tensor, targets_tensor