import os
import json
import h5py
import numpy as np
from PIL import Image # Import Pillow for image processing

def get_numeric_value(data_field, default_value=0.0):

    if isinstance(data_field, (int, float)):
        return float(data_field)
    elif isinstance(data_field, dict):
        if "value" in data_field: 
            return float(data_field["value"])
        elif "x" in data_field: 
            return float(data_field["x"])
        return default_value 
    return default_value

def process_driving_data(folder_path, output_folder="output_h5"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    json_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".json") and file.startswith("measurements_"):
                json_files.append(os.path.join(root, file))

    all_targets_records = []
    all_rgb_images = []
    record_count = 0
    h5_file_index = 0

    print(f"Found {len(json_files)} JSON files to process.")

    for json_path in json_files:
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)

            #extraer id, luego buscar png con esep
            file_id = os.path.splitext(os.path.basename(json_path))[0].replace("measurements_", "")

            #separar targets rgb/imagenes
            targets_record = {}
    
            if "playerMeasurements" in data and "autopilotControl" in data["playerMeasurements"]:
                steer_data = data["playerMeasurements"]["autopilotControl"].get("steer", 0.0)
                targets_record["steer"] = get_numeric_value(steer_data, 0.0)
            else:
                targets_record["steer"] = 0.0

            gas_data = data.get("throttle", 0.0)
            targets_record["gas"] = get_numeric_value(gas_data, 0.0)

            brake_data = data.get("brake", 0.0)
            targets_record["brake"] = get_numeric_value(brake_data, 0.0)

            dir =  data.get("directions", 0.0)
            targets_record["cmd"] = get_numeric_value(dir, 0.0)

            if "playerMeasurements" in data:
                speed_data = data["playerMeasurements"].get("forwardSpeed", 0.0)
                targets_record["speed"] = get_numeric_value(speed_data, 0.0)
            else:
                targets_record["speed"] = 0.0
            
            all_targets_records.append(targets_record)

            rgb_image_filename = f"CentralRGB_{file_id}.png"
            rgb_image_path = os.path.join(os.path.dirname(json_path), rgb_image_filename)

            if os.path.exists(rgb_image_path):
                try:
                    img = Image.open(rgb_image_path)
                    rgb_array = np.array(img) 
                    all_rgb_images.append(rgb_array)
                except Exception as img_e:
                    print(f"Warning: Could not load image {rgb_image_path}: {img_e}. Skipping image for this record.")
                    #dimensiones ajuste
                    if len(all_rgb_images) > 0:
                         all_rgb_images.append(np.zeros_like(all_rgb_images[0])) 
                    else:
                        
                        #(88, 200, 3)
                        print("ERROR: First image failed to load and no default shape known. Please define a default placeholder shape.")
                        raise Exception("Initial image load failed, cannot determine placeholder shape.")
            else:
                print(f"Warning: RGB image not found for ID {file_id} at {rgb_image_path}. Skipping image for this record.")
                
                if len(all_rgb_images) > 0:
                    all_rgb_images.append(np.zeros_like(all_rgb_images[0]))
                else:
                    print("ERROR: No images loaded yet and first one not found. Please define a default placeholder shape.")
                    raise Exception("Initial image not found, cannot determine placeholder shape.")

            record_count += 1

            #limite 200 por h5
            if record_count % 200 == 0:
                h5_filename = os.path.join(output_folder, f"driving_data_{h5_file_index:03d}.h5")
                with h5py.File(h5_filename, 'w') as hf:
                    dt_targets = np.dtype([
                        ('steer', 'f4'),
                        ('gas', 'f4'),
                        ('brake', 'f4'),
                        ('cmd', 'f4'),
                        ('speed', 'f4')
                    ])
                    structured_targets_array = np.array([
                        (r['steer'], r['gas'], r['brake'], r['cmd'], r['speed'])
                        for r in all_targets_records
                    ], dtype=dt_targets)
                    hf.create_dataset('targets', data=structured_targets_array)

                    if all_rgb_images:
                        rgb_data_array = np.array(all_rgb_images)
                        hf.create_dataset('rgb', data=rgb_data_array, compression="gzip")
                    else:
                        print(f"No RGB images to save for batch {h5_file_index}.")
                
                print(f"Saved {record_count} records and images in {h5_filename}")
                all_targets_records = [] 
                all_rgb_images = []
                h5_file_index += 1

        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON file: {json_path}. Skipping.")
        except KeyError as e:
            print(f"Error: Missing expected key ({e}) in file: {json_path}. Skipping.")
        except Exception as e:
            print(f"An unexpected error occurred while processing {json_path}: {e}. Skipping.")

    if all_targets_records:
        h5_filename = os.path.join(output_folder, f"driving_data_{h5_file_index:03d}.h5")
        with h5py.File(h5_filename, 'w') as hf:
            dt_targets = np.dtype([
                ('steer', 'f4'),
                ('gas', 'f4'),
                ('brake', 'f4'),
                ('cmd', 'f4'),
                ('speed', 'f4')
            ])
            structured_targets_array = np.array([
                (r['steer'], r['gas'], r['brake'], r['cmd'], r['speed'])
                for r in all_targets_records
            ], dtype=dt_targets)
            hf.create_dataset('targets', data=structured_targets_array)

            if all_rgb_images:
                rgb_data_array = np.array(all_rgb_images)
                hf.create_dataset('rgb', data=rgb_data_array, compression="gzip")
            else:
                print(f"No RGB images to save for the final batch {h5_file_index}.")

        print(f"Saved the remaining {len(all_targets_records)} records and images in {h5_filename}")

if __name__ == "__main__":
    #ejecutando dentro de la misma carpetaa  episode_0000
    json_folder_path = "episode_0000" 
    
    process_driving_data(json_folder_path)
    print("Processing completed.")
