import csv
import os
from pdb import main
import numpy as np
import torch
import re

def read_dat_file(dat_file_path):
    in_nodes = False
    in_elements = False
    nodes = []
    cells_detail = []

    with open(dat_file_path, "r") as file:
        lines = file.readlines()

    lines_iter = iter(lines)

    for line in lines_iter:

        line = line.strip()

        # Detect node block
        if line.lower().startswith("nblock"):
            in_nodes = True
            continue

        if in_nodes and line.strip() == "-1":
            in_nodes = False
            continue

        # Detect element block (EBLOCK)
        if line.lower().startswith("eblock"):
            in_elements = True
            continue

        if in_elements and line.strip() == "-1":
            in_elements = False
            continue

        # Parse nodes
        if in_nodes:
            parts = line.split()
            if len(parts) == 4:
                node_id = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])
                nodes.append((node_id, x, y, z))

        # Parse elements (EBLOCK, 2-line format)
        elif in_elements:
            parts = line.split()
            if len(parts) >= 12:
                try:
                    element_id = int(parts[10])
                    body_id = int(parts[11])
                    node_ids = [int(x) for x in parts[12:]]
                    next_line = next(lines_iter).strip()
                    node_ids += [int(x) for x in next_line.split()]
                    cells_detail.append([element_id, body_id] + node_ids)
                except Exception as e:
                    print(f"Warning: Failed to parse element line: {line}\nError: {e}")

    # Sort nodes and cells
    nodes.sort(key=lambda x: x[0])
    cells_detail.sort(key=lambda x: x[0])

    # Remove duplicates
    cells_detail = [list(item) for item in dict.fromkeys(tuple(cell) for cell in cells_detail)]

    # Extract 4-node face from each cell (can adjust as needed)
    cells = [[cell[2], cell[3], cell[4], cell[6]] for cell in cells_detail if len(cell) > 6]

    return nodes, cells_detail, cells

def assign_body_id_to_nodes(nodes, elements):
    node_body_mapping = {}
    for element in elements:
        body_id = element[1]
        node_ids = element[2:]
        for node_id in node_ids:
            if node_id not in node_body_mapping:
                node_body_mapping[node_id] = [node_id, body_id]

    node_type = list(node_body_mapping.values())
    node_type.sort(key=lambda x: x[0])
    node_type_bid = [node[1] for node in node_type]
    node_type_mgn = [0 if num == 1 else 1 if num == 4 else 1 for num in node_type_bid]
    return node_type_mgn

def assign_body_id_to_nodes_quarter(nodes, elements):
    node_body_mapping = {}
    for element in elements:
        body_id = element[1]
        node_ids = element[2:]
        for node_id in node_ids:
            if node_id not in node_body_mapping:
                node_body_mapping[node_id] = [node_id, body_id]

    node_type = list(node_body_mapping.values())
    node_type.sort(key=lambda x: x[0])

    for node in node_type:
        for original_node in nodes:
            if node[0] == original_node[0] and node[1] == 1:
                x, y = original_node[1], original_node[2]
                if x == 0 or (0 > x >= -100 and y == 0):
                    node[1] = 4
                break

    node_type_bid = [node[1] for node in node_type]
    node_type_mgn = [0 if num == 1 else 3 if num == 4 else 1 for num in node_type_bid]
    return node_type_mgn

def read_result_file(result_file_path, encoding='utf-8'):
    parameters = []
    with open(result_file_path, mode='r', encoding=encoding) as file:
        if result_file_path.endswith(".csv"):
            reader = csv.reader(file)
            next(reader)  # Skip header
            for row in reader:
                parameters.append(float(row[-1]))
        elif result_file_path.endswith(".txt"):
            for line in file:
                if "Node Number" in line:
                    continue
                columns = line.split()
                try:
                    parameters.append(float(columns[-1]))
                except ValueError:
                    continue
    return parameters
'''
def find_global_pos(mesh_pos, disp_x, disp_y, disp_z):
    global_pos = []
    for i, pos in enumerate(mesh_pos):
        global_pos.append([
            pos[0] + float(disp_x[i]),
            pos[1] + float(disp_y[i]),
            pos[2] + float(disp_z[i])
        ])
    return global_pos
'''
def generate_frame(folder_path, device):
    folder_base = os.path.basename(folder_path).replace("_data", "")
    dat_path = os.path.join(folder_path, f"{folder_base}.dat")
    fatigue_life_path = os.path.join(folder_path, f"{folder_base}.txt")

    if not os.path.exists(dat_path) or not os.path.exists(fatigue_life_path):
        print(f"Missing files in {folder_path}")
        return None

    try:
        #reading dat file and extracting nodes, cells, and time steps. and converting to tensors.
        nodes, cells_detail, cells, _ = read_dat_file(dat_path)
        mesh_pos_list = [list(node[1:4]) for node in nodes]
        mesh_pos = torch.tensor(mesh_pos_list, dtype=torch.float32).to(device)
        node_type = assign_body_id_to_nodes(nodes, cells_detail)
        node_type = torch.tensor(node_type, dtype=torch.float32).view(-1, 1).to(device)
        cells = torch.tensor(cells, dtype=torch.float32).to(device) - 1

        #reading result(fatigue life ) file and converting to tensors.
        fatigue_life = read_result_file(fatigue_life_path)
        fatigue_life = torch.tensor(fatigue_life, dtype=torch.float32).view(-1, 1).to(device)

        frame = {
            'mesh_pos': mesh_pos,
            'cells': cells,
            'node_type': node_type,
            'fatigue_life': fatigue_life
        }
        return frame
    except Exception as e:
        print(f"Error processing {folder_path}: {str(e)}")
        return None
    
    
def main():
        # Example folder path — change this to test with your own folder
        folder_path = r"D:\AnK\Project Fatigue\Notch shaft\fatigue_net\PressNet\datasets\Raw data\shaft\life_D15_d5_r1_data"

        # Use CPU (or set to 'cuda' if GPU is supported)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Try generating dataset from the folder
        frame = generate_frame(folder_path, device)

        if frame is None:
            print("Failed to generate dataset.")
            return

        print("\n===================================================")
        print("Fatigue Dataset Frame Contents and Tensor Shapes:")
        print("===================================================\n")
        
        for key, value in frame.items():
            print(f"{key:15} : {value.shape}")

        print("\nFrame generation successful!\n")


if __name__ == '__main__':
    main()
