import argparse
import csv
import numpy as np
from skimage.morphology import skeletonize, remove_small_objects
from scipy.ndimage import distance_transform_edt
import os
import yaml
import torch
from models.frangi import Frangi
from PIL import Image

def read_config(config_file):
    with open(config_file, 'r') as f:
        if config_file.endswith('.yml') or config_file.endswith('.yaml'):
            return yaml.safe_load(f)
        else:
            raise ValueError("Only YAML config files are supported.")

def prepare_output_dir(output_config):
    out_dir = output_config['directory']
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    return out_dir

def segment_vessels_with_frangi_2d(image_path, threshold=0.5):
    """
    Apply Frangi filter to a 2D PNG image and save the binary vessel mask as PNG.
    """
    img = Image.open(image_path).convert('L')
    data = np.array(img).astype(np.float32)
    # Normalize to [0,1]
    data_norm = (data - data.min()) / (data.max() - data.min() + 1e-8)
    tensor_img = torch.tensor(data_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    frangi = Frangi()
    vesselness = frangi(tensor_img)
    vessel_mask = (vesselness > threshold).squeeze().numpy().astype(np.uint8)
    print("Vessel mask sum:", vessel_mask.sum())
    # Save mask as PNG
    mask_path = image_path.replace(".png", "_frangi_mask.png")
    Image.fromarray((vessel_mask * 255).astype(np.uint8)).save(mask_path)
    return mask_path

def extract_vessel_graph_from_2d_image(image_path, out_csv_path, threshold=0.5, min_skel_size=20, norm_radius=True, pixel_spacing=1.0):
    """
    Extract vessel graph from a 2D vessel mask (PNG) and save as CSV.
    """
    img = Image.open(image_path).convert('L')
    data = np.array(img).astype(np.float32)
    vessel_mask = data > (threshold * 255 if data.max() > 1 else threshold)
    shape = data.shape  # (y, x)

    # 2D skeletonization
    skeleton = skeletonize(vessel_mask)
    # Remove small skeleton objects (noise)
    skeleton = remove_small_objects(skeleton, min_size=min_skel_size)
    print("Skeleton pixels after removal:", np.count_nonzero(skeleton))

    # Distance map for radius calculation
    distance_map = distance_transform_edt(vessel_mask)

    # For normalization, use the largest dimension in physical units
    if norm_radius:
        box_size = np.max(np.array(shape) * pixel_spacing)
    else:
        box_size = 1.0

    coords = np.argwhere(skeleton)
    edges_set = set()
    edges = []
    for y, x in coords:
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if (0 <= ny < skeleton.shape[0] and
                    0 <= nx < skeleton.shape[1] and
                    skeleton[ny, nx]):
                    edge_key = tuple(sorted([(y, x), (ny, nx)]))
                    if edge_key not in edges_set:
                        edges_set.add(edge_key)
                        node1 = [float(x)/shape[1], float(y)/shape[0]]
                        node2 = [float(nx)/shape[1], float(ny)/shape[0]]
                        radius = float(distance_map[y, x]) * pixel_spacing / box_size
                        edges.append((node1, node2, radius))
    print("Edges found:", len(edges))

    def format_coord(coord):
        return "[" + " ".join(f"{v:.8f}" for v in coord) + "]"

    with open(out_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['node1', 'node2', 'radius'])
        for node1, node2, radius in edges:
            writer.writerow([format_coord(node1), format_coord(node2), radius])

def main(config):
    input_dir = config['input']['directory']
    file_ext = config['input']['file_extension']
    output_config = config['output']
    output_dir = prepare_output_dir(output_config)
    threshold = config['input'].get('frangi_threshold', 0.5)
    min_skel_size = config['input'].get('min_skel_size', 20)
    pixel_spacing = config['input'].get('pixel_spacing', 0.015)  # Set to your pixel size in mm if known

    counter = 0
    for fname in os.listdir(input_dir):
        if not fname.endswith(file_ext):
            continue
        counter += 1
        print(f"Processing file: {counter}/ {len}: {fname}")
        image_path = os.path.join(input_dir, fname)
        out_csv_path = os.path.join(output_dir, os.path.splitext(fname)[0] + '_vessel_graph.csv')
        # Step 1: Segment vessels with Frangi filter
        mask_path = segment_vessels_with_frangi_2d(image_path, threshold=threshold)
        # Step 2: Extract vessel graph from mask
        extract_vessel_graph_from_2d_image(mask_path, out_csv_path, threshold=0.5, min_skel_size=min_skel_size, pixel_spacing=pixel_spacing)
        print(f"Extracted vessel graph saved to {out_csv_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract vessel graph from 2D OCTA PNG images using config file.')
    parser.add_argument('--config_file', type=str, required=True)
    args = parser.parse_args()

    assert os.path.isfile(args.config_file), f"Error: Your provided config path {args.config_file} does not exist!"
    config = read_config(args.config_file)
    main(config)