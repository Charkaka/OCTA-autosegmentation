import argparse
import csv
import os
import yaml
import glob
import warnings
import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage import img_as_bool
from multiprocessing import cpu_count
import concurrent.futures
from rich.console import Group, Console
from rich.live import Live
from rich.progress import Progress, TimeElapsedColumn
from utils.visualizer import DynamicDisplay
from vessel_graph_generation.utilities import prepare_output_dir, read_config

group = Group()

from scipy.ndimage import distance_transform_edt

def extract_vessel_graph_from_image(image_path, config, out_dir):
    # 1. Load image (grayscale)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Failed to load {image_path}")
        return

    # 2. Vessel segmentation (adaptive thresholding)
    vessel_mask = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10
    )

    # 3. Skeletonize
    skeleton = skeletonize(img_as_bool(vessel_mask))

    # 4. Compute distance transform (for radius estimation)
    vessel_mask_bool = vessel_mask > 0
    distance_map = distance_transform_edt(vessel_mask_bool)

    # --- FOV and pixel size ---
    fov_mm = float(config['input'].get('fov_mm', 6.0))  # Default to 6.0mm if not set
    img_h, img_w = img.shape
    pixel_size_mm_y = fov_mm / img_h
    pixel_size_mm_x = fov_mm / img_w

    # 5. Extract vessel graph as edge list (simple 4-neighbor connectivity)
    vessel_coords = np.column_stack(np.where(skeleton))
    edge_set = set()
    for y, x in vessel_coords:
        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
            ny, nx = y+dy, x+dx
            if 0 <= ny < skeleton.shape[0] and 0 <= nx < skeleton.shape[1]:
                if skeleton[ny, nx]:
                    # Convert pixel indices to physical (mm) locations
                    node1 = np.array([y * pixel_size_mm_y, x * pixel_size_mm_x])
                    node2 = np.array([ny * pixel_size_mm_y, nx * pixel_size_mm_x])
                    # Estimate radius in mm
                    radius = float((distance_map[y, x] + distance_map[ny, nx]) / 2) * ((pixel_size_mm_x + pixel_size_mm_y) / 2)
                    # Always store edges in sorted order to avoid duplicates
                    edge = tuple(sorted([tuple(node1), tuple(node2)])) + (radius,)
                    edge_set.add(edge)

    # 6. Save vessel graph as CSV
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    csv_path = os.path.join(out_dir, f"{base_name}_vessel_graph.csv")
    with open(csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["node1", "node2", "radius"])
        for node1, node2, radius in edge_set:
            # Write as NumPy-style arrays
            writer.writerow([f"[{node1[0]:.6f} {node1[1]:.6f}]", f"[{node2[0]:.6f} {node2[1]:.6f}]", radius])

def main(config):
    # Prepare output directory
    out_dir = prepare_output_dir(config['output'])
    with open(os.path.join(out_dir, 'config.yml'), 'w') as f:
        yaml.dump(config, f)

    input_dir = config['input']['directory']
    file_ext = config['input']['file_extension']
    image_paths = sorted(glob.glob(os.path.join(input_dir, f"*{file_ext}")))

    for image_path in image_paths:
        extract_vessel_graph_from_image(image_path, config, out_dir)
    
   
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract vessel graphs from OCTA500 images')
    parser.add_argument('--config_file', type=str, required=True)
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--threads', help="Number of parallel threads. By default all available threads but one are used.", type=int, default=-1)
    args = parser.parse_args()

    if args.debug:
        warnings.filterwarnings('error')

    assert os.path.isfile(args.config_file), f"Error: Your provided config path {args.config_file} does not exist!"
    config = read_config(args.config_file)

    assert config['output'].get('save_3D_volumes') in [None, 'npy', 'nifti'], \
        f"Your provided option {config['output'].get('save_3D_volumes')} for 'save_3D_volumes' does not exist. Choose one of 'null', 'npy' or 'nifti'."

    if args.threads == -1:
        cpus = cpu_count()
        threads = max(cpus - 1, 1)
    else:
        threads = args.threads

    with Live(group, console=Console(force_terminal=True), refresh_per_second=10):
        progress = Progress(*Progress.get_default_columns(), TimeElapsedColumn())
        image_dir = config['input']['directory']
        file_ext = config['input']['file_extension']
        image_paths = sorted(glob.glob(os.path.join(image_dir, f"*{file_ext}")))
        progress.add_task(f"Extracting vessel graphs from {len(image_paths)} images:", total=len(image_paths))
        with DynamicDisplay(group, progress):
            if threads > 1:
                with concurrent.futures.ProcessPoolExecutor(max_workers=threads) as executor:
                    future_dict = {
                        executor.submit(extract_vessel_graph_from_image, image_path, config, config['output']['directory']): i
                        for i, image_path in enumerate(image_paths)
                    }
                    for future in concurrent.futures.as_completed(future_dict):
                        progress.advance(task_id=0)
            else:
                for image_path in image_paths:
                    extract_vessel_graph_from_image(image_path, config, config['output']['directory'])
                    progress.advance(task_id=0)
        print("Vessel graph extraction completed.")