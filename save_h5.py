import os
import h5py
import torch
import sys
import json
from PressNet.datasets.utilities.trajectory_solid import generate_trajectory_h5 as generate_trajectory
# Add the utilities directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utilities'))


def list_subfolders(directory):
    return [f.name for f in os.scandir(directory) if f.is_dir()]

def save_fatigue_dataset_to_h5(raw_folder_path, output_folder_path, device):

    basename = os.path.basename(raw_folder_path)
    h5_path = os.path.join(output_folder_path, f"{basename}_.h5")
    meta_json_path = os.path.join(output_folder_path, f"{basename}_.json")

    with h5py.File(h5_path, 'w') as f:
        group_to_folder_map = {}
        group_counter = 0

        # Each folder in raw_folder_path is a geometry folder
        geometry_folders = list_subfolders(raw_folder_path)
        print(f"Found {len(geometry_folders)} geometry folders")

        for geometry_folder in geometry_folders:
            geometry_path = os.path.join(raw_folder_path, geometry_folder)

            print(f"Processing geometry: {geometry_folder}")

            # Generate fatigue data from folder
            trajectory= generate_trajectory(geometry_path, device)
            #if len(trajectory) == 0:
            #        continue
            # Skip if no valid data
            if trajectory is None:
                print(f"Skipping invalid or empty: {geometry_folder}")
                continue

            # Create a unique group name
            group_name = f"group_{group_counter}"
            traj_group = f.create_group(group_name)

            # Save all tensors to the group
            for key, value in trajectory[0].items():
                traj_group.create_dataset(key, data=value.cpu().numpy())

            # Meta info
            num_nodes = trajectory[0]['mesh_pos'].shape[0]
            num_cells = trajectory[0]['cells'].shape[0]
            num_fatigue_life = trajectory[0]['fatigue_life'].shape[0]
            
            group_to_folder_map[group_name] = {
                'geometry_folder': geometry_folder,
                'number_of_nodes': num_nodes,
                'number_of_cells': num_cells,
                'number_of_fatigue_life': num_fatigue_life,
                'data_type': 'fatigue_life',
            }


            print(f"✓ Added group {group_name} for geometry {geometry_folder}")
            print(f"  - Nodes: {num_nodes}, Cells: {num_cells}")
            group_counter += 1

        # Save meta JSON
        with open(meta_json_path, 'w') as jf:
            json.dump(group_to_folder_map, jf, indent=4)
        print(f"Saved metadata to {meta_json_path}")
        print(f"Processed {group_counter} geometries successfully")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    raw_folder_path = r"D:\AnK\Project Fatigue\Notch shaft\fatigue_net\PressNet\datasets\Raw data\shaft"
    output_folder_path = r"D:\AnK\Project Fatigue\Notch shaft\fatigue_net\PressNet\datasets\extracted_data"

    # Check if raw folder exists
    if not os.path.exists(raw_folder_path):
        print(f"Error: Raw folder path does not exist: {raw_folder_path}")
        return

    # Create output directory if it doesn't exist
    os.makedirs(output_folder_path, exist_ok=True)
    print(f"Output will be saved to: {output_folder_path}")

    # Convert data
    save_fatigue_dataset_to_h5(raw_folder_path, output_folder_path, device)

if __name__ == '__main__':
    main()