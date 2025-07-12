# check_cmd_values.py
import h5py
import os

file_path = os.path.join(os.getcwd(), "data/SeqTrain", "data_06951.h5")

with h5py.File(file_path, 'r') as f:
    commands = f['targets'][:, 24]
    print("Valores únicos de los comandos:", set(commands.tolist()))
    print("Muestra:", commands)
    print("Cantidad de comandos:", len(commands))
