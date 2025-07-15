import re

def read_dat_file(dat_file_path):
    in_nodes = False
    in_elements = False
    nodes = []
    cells_detail = []
    deltim_pattern = r'deltim,\s*([0-9e\-\.]+)'
    time_step = 1.0  # Default if not found

    with open(dat_file_path, "r") as file:
        lines = file.readlines()

    lines_iter = iter(lines)

    for line in lines_iter:
        match = re.search(deltim_pattern, line)
        if match:
            time_step = float(match.group(1))

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

    return nodes, cells_detail, cells, time_step


def main():
    dat_file_path = "D:\\AnK\\Project Fatigue\\Notch shaft\\fatigue_net\\PressNet\\datasets\\Raw data\\shaft\\life_D15_d5_r1_data\\life_D15_d5_r1.dat"  # Replace with your .dat file path
    try:
        nodes, cells_detail, cells, time_step = read_dat_file(dat_file_path)
        
        print("Data Read Confirmation:") 
        print(f"Time Step: {time_step}")
        print(f"Number of Nodes: {len(nodes)}")
        print("First 5 Nodes (ID, X, Y, Z):")
        for node in nodes[:5]:
            print(f"  {node}")
        if len(nodes) > 5:
            print("  ... (more nodes not shown)")
        
        print(f"Number of Cells: {len(cells)}")
        print("First 5 Cells (Node IDs):")
        for cell in cells[:5]:
            print(f"  {cell}")
        if len(cells) > 5:
            print("  ... (more cells not shown)")
        
        print(f"Number of Detailed Cells: {len(cells_detail)}")
        print("First 5 Detailed Cells (Element ID, Body ID, Node IDs):")
        for cell_detail in cells_detail[:5]:
            print(f"  {cell_detail}")
        if len(cells_detail) > 5:
            print("  ... (more detailed cells not shown)")
            
    except FileNotFoundError:
        print(f"Error: File '{dat_file_path}' not found.")
    except Exception as e:
        print(f"Error: An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()