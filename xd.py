import h5py
import os

h5_dir = os.getcwd() + "/deep_learning_proyecto/data/SeqTrain/"
h5_files = [os.path.join(h5_dir, f) for f in os.listdir(h5_dir) if f.endswith('.h5')]

for path in h5_files:
    print(f"\n📂 Revisando archivo: {os.path.basename(path)}")
    with h5py.File(path, 'r') as f:
        print("🔑 Claves disponibles:", list(f.keys()))
        for key in f.keys():
            print(f"   - {key}: shape {f[key].shape}")
    break  # revisa solo el primero
