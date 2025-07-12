from torch.utils.data import Dataset
import torch
import numpy as np
import h5py
from PIL import Image
from collections import Counter

class CarlaDataset(Dataset):
    def __init__(self, h5_files, seq_len=5, transform=None, oversample=False):
        self.seq_len = seq_len
        self.transform = transform
        self.data = []
        self.cmd_counter_raw = Counter()
        self.cmd_counter_final = Counter()

        print("🔍 Iniciando carga de archivos H5...")

        for file in h5_files:
            try:
                with h5py.File(file, 'r') as f:
                    images = f['rgb'][:]
                    targets = f['targets'][:]

                    for i in range(len(images) - seq_len):
                        img_seq = images[i:i+seq_len]
                        steer, gas, brake = targets[i+seq_len-1, 0:3]
                        cmd = int(targets[i+seq_len-1, 24])  # comando
                        speed = targets[i+seq_len-1, 10]

                        self.cmd_counter_raw[cmd] += 1

                        if cmd not in [2, 3, 4, 5]:
                            print(f"⚠️  Comando atípico encontrado: {cmd} en archivo: {file}")
                            continue  # saltar secuencia inválida

                        sample = (img_seq, cmd, speed, [steer, gas, brake])
                        self.data.append(sample)
                        self.cmd_counter_final[cmd] += 1

                        # Oversample para izquierda y derecha
                        if oversample and cmd in [3, 4]:
                            for _ in range(2):
                                self.data.append(sample)
                                self.cmd_counter_final[cmd] += 1
            except OSError as e:
                print(f"💥 Archivo corrupto omitido: {file}")
                continue

        print("📊 Frecuencia de comandos (original):", dict(self.cmd_counter_raw))
        print("📊 Frecuencia de comandos (final con oversample):", dict(self.cmd_counter_final))
        print(f"✅ Total de secuencias cargadas: {len(self.data)}")

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
        cmd_tensor = torch.tensor(cmd - 2).long()  # Para tener rango 0-3
        speed_tensor = torch.tensor([speed]).float()
        targets_tensor = torch.tensor(targets).float()

        return img_seq_tensor, cmd_tensor, speed_tensor, targets_tensor
