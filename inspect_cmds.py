# inspect_cmds.py
import os
from torchvision import transforms
from loader2 import CarlaDataset

transform = transforms.Compose([
    transforms.Resize((88, 200)),
    transforms.ToTensor()
])

# Ruta a los H5
h5_dir = os.getcwd() + "/data/SeqTrain/"
h5_files = [os.path.join(h5_dir, f) for f in os.listdir(h5_dir) if f.endswith('.h5')][:1500]

# Carga del dataset
dataset = CarlaDataset(h5_files, seq_len=5, transform=transform, oversample=False)

print(f"🔍 Se analizaron {len(h5_files)} archivos H5")
print(f"📦 Total de secuencias: {len(dataset)}")
print(f"📊 Comandos observados: {dataset.cmd_counter_final}")
