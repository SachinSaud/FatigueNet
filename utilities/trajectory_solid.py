import os
import sys
import torch

# Add the utilities directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utilities'))

# Import the frame_solid185 module
import PressNet.datasets.utilities.frame_solid as frame

def generate_trajectory(single_sim_folder_path, device): # generate tragectory dataset for training 
    
    print(f"Extracting trajectory from: {single_sim_folder_path}")

    # Setup filenames and paths 
    folder_base_name = os.path.basename(single_sim_folder_path).replace("_data", "")
    dat_file_path = os.path.join(single_sim_folder_path, f"{folder_base_name}.dat")
    fatigue_life_path = os.path.join(single_sim_folder_path, f"{folder_base_name}.txt")

    # Check if files exist
    if not os.path.exists(dat_file_path):
        print(f"Error: .dat file not found: {dat_file_path}")
        return None
    
    if not os.path.exists(fatigue_life_path):
        print(f"Error: Fatigue_Life.txt not found: {fatigue_life_path}")
        return None

    try:
        # Read mesh and connectivity
        nodes, cells_detail, cells = frame.read_dat_file(dat_file_path)
        print("✓ .dat file loaded: nodes and cells extracted")

        mesh_pos_list = [list(node[1:4]) for node in nodes]  # XYZ only
        mesh_pos = torch.tensor(mesh_pos_list, dtype=torch.float32).to(device)

        node_type = frame.assign_body_id_to_nodes(nodes, cells_detail)
        node_type = torch.tensor(node_type, dtype=torch.float32).view(-1, 1).to(device)

        cells = torch.tensor(cells, dtype=torch.float32) - 1  # Convert to 0-based indexing
        cells = cells.to(device)

        # Read fatigue life data
        fatigue_life = frame.read_result_file(fatigue_life_path)
        fatigue_life = torch.tensor(fatigue_life, dtype=torch.float32).view(-1, 1).to(device)

        # Package into one trajectory step (no time steps)
        frame_data = {
            'mesh_pos': mesh_pos,
            'cells': cells,
            'node_type': node_type,         # Reusing naming convention
            'fatigue_life': fatigue_life,  # This is the prediction target
        }

        trajectory = [frame_data]  # Still returning as a list (like a 1-frame trajectory)

        print("✓ Fatigue frame extracted.\n")
        return trajectory
        
    except Exception as e:
        print(f"Error processing trajectory: {str(e)}")
        return None

def generate_trajectory_h5(single_sim_folder_path, device):
    print(f"Extracting fatigue H5-compatible trajectory from: {single_sim_folder_path}")

    folder_base_name = os.path.basename(single_sim_folder_path).replace("_data", "")
    dat_file_path = os.path.join(single_sim_folder_path, f"{folder_base_name}.dat")
    fatigue_life_path = os.path.join(single_sim_folder_path, f"{folder_base_name}.txt")

    if not os.path.exists(dat_file_path):
        print(f"Error: .dat file not found: {dat_file_path}")
        return None, None
    
    if not os.path.exists(fatigue_life_path):
        print(f"Error: Fatigue_Life.txt not found: {fatigue_life_path}")
        return None, None

    try:
        # Read mesh
        nodes, cells_detail, cells = frame.read_dat_file(dat_file_path)
        print("✓ .dat file loaded")

        mesh_pos_list = [list(node[1:4]) for node in nodes]
        mesh_pos = torch.tensor(mesh_pos_list, dtype=torch.float32).to(device)

        node_type = frame.assign_body_id_to_nodes(nodes, cells_detail)
        node_type = torch.tensor(node_type, dtype=torch.float32).view(-1, 1).to(device)

        cells = torch.tensor(cells, dtype=torch.float32).to(device) - 1

        fatigue_life = frame.read_result_file(fatigue_life_path)
        fatigue_life = torch.tensor(fatigue_life, dtype=torch.float32).view(-1, 1).to(device)

        # Wrap into H5-compatible frame
        frame_data = {
            'mesh_pos': mesh_pos,
            'cells': cells,
            'node_type': node_type,         # Reusing naming convention
            'fatigue_life': fatigue_life,
        }

        trajectory = [frame_data]  # still packaged as list for H5 saving
        return trajectory

    except Exception as e:
        print(f"Error creating h5 trajectory: {e}")
        return None



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Update this path to match your actual data structure
    folder = r"D:\AnK\Project Fatigue\Notch shaft\fatigue_net\PressNet\datasets\Raw data\shaft\life_D15_d5_r1_data"
    
    # Check if folder exists
    if not os.path.exists(folder):
        print(f"Error: Folder does not exist: {folder}")
        return

    trajectory = generate_trajectory(folder, device)

    if trajectory is not None:
        print("\n===== Fatigue Trajectory Summary =====")
        for key in trajectory[0]:
            print(f"{key} shape: {trajectory[0][key].shape}")
    else:
        print("Failed to generate trajectory")

if __name__ == '__main__':
    main()