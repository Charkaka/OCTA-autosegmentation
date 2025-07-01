import argparse
import json
import torch
import os

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np
import csv

from monai.utils import set_determinism
import yaml
from models.model import define_model
from models.networks import init_weights
from data.image_dataset import get_dataset, get_post_transformation

from utils.visualizer import plot_sample, plot_single_image, DynamicDisplay
from utils.enums import Phase

from rich.live import Live
from rich.progress import Progress, TimeElapsedColumn
from rich.spinner import Spinner
from rich.console import  Group, Console
group = Group()

from torchvision.utils import save_image
import subprocess

# # Parse input arguments
# parser = argparse.ArgumentParser(description='')
# parser.add_argument('--config_file', type=str, required=True)
# parser.add_argument('--epoch', type=str, default="best")
# parser.add_argument('--num_samples', type=int, default=9999999)
# parser.add_argument('--num_workers', type=int, default=None, help="Number of cpu cores used for dataloading. By, use half of the available cores.")
# args = parser.parse_args()
# epoch_suffix = f"_{args.epoch}"
# assert args.num_samples>0

# # Read config file
# path: str = os.path.abspath(args.config_file)
# assert os.path.isfile(path), f"Your provided config path {args.config_file} does not exist!"
# with open(path, "r") as stream:
#     if path.endswith(".json"):
#         config: dict[str,dict] = json.load(stream)
#     else:
#         config: dict[str,dict] = yaml.safe_load(stream)
# if config["General"].get("seed") is not None:
#     set_determinism(seed=config["General"]["seed"])

# inference_suffix = "_"+config["General"]["inference"] if "inference" in config["General"] else ""
# save_dir = config[Phase.TEST].get("save_dir") or config["Output"]["save_dir"]+"/test"
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)

# device = torch.device(config["General"].get("device") or "cpu")
# scaler = torch.cuda.amp.GradScaler(enabled=False)
# # set_determinism(seed=0)

# with Live(group, console=Console(force_terminal=True), refresh_per_second=10):
#     with DynamicDisplay(group, Spinner("bouncingBall", text="Loading test data...")):
#         test_loader = get_dataset(config, Phase.TEST, num_workers=args.num_workers)
#         post_transformations_test = get_post_transformation(config, Phase.TEST)
#         test_mini_batch = next(iter(test_loader))
#         input_key = [k for k in test_mini_batch.keys() if not k.endswith("_path")][0]
#         test_mini_batch["image"] = test_mini_batch.pop(input_key)

#     with DynamicDisplay(group, Spinner("bouncingBall", text="Initializing model...")):
#         model = define_model(config, phase=Phase.TEST)
#         model.initialize_model_and_optimizer(test_mini_batch, init_weights, config, args, scaler, phase=Phase.TEST)
#     predictions = []

#     model.eval()
#     progress = Progress(*Progress.get_default_columns(), TimeElapsedColumn())
#     progress.add_task("Testing:", total=min(len(test_loader), args.num_samples))
#     with DynamicDisplay(group, progress):
#         with torch.no_grad():
#             num_sample=0
#             for test_mini_batch in test_loader:
#                 if num_sample>=args.num_samples:
#                     break
#                 num_sample+=1
#                 test_mini_batch["image"] = test_mini_batch.pop(input_key)
#                 outputs, _ = model.inference(test_mini_batch, post_transformations_test, device=device, phase=Phase.TEST)
#                 inference_mode = config["General"].get("inference") or "pred"
#                 image_name: str = test_mini_batch[f"{input_key}_path"][0].split("/")[-1]
                
#                 # plot_single_image(save_dir, inputs[0], image_name)
#                 plot_single_image(save_dir, outputs["prediction"][0], inference_mode + "_" + image_name)
#                 if config["Output"].get("save_comparisons"):
#                     plot_sample(save_dir, test_mini_batch[input_key][0], outputs["prediction"][0], None, test_mini_batch[f"{input_key}_path"][0], suffix=f"{inference_mode}_{image_name}", full_size=True)
#                 progress.advance(task_id=0)



# Parse input arguments
parser = argparse.ArgumentParser(description='')
parser.add_argument('--config_file', type=str, required=True)
parser.add_argument('--epoch', type=str, default="150")
parser.add_argument('--num_samples', type=int, default=9999999)
parser.add_argument('--num_workers', type=int, default=None, help="Number of cpu cores used for dataloading. By, use half of the available cores.")
parser.add_argument('--model_path', type=str, default=None, help="Path to the model checkpoint to load.")

args = parser.parse_args()
epoch_suffix = f"_{args.epoch}"
assert args.num_samples>0

# Read config file
path: str = os.path.abspath(args.config_file)
assert os.path.isfile(path), f"Your provided config path {args.config_file} does not exist!"
with open(path, "r") as stream:
    if path.endswith(".json"):
        config: dict[str,dict] = json.load(stream)
    else:
        config: dict[str,dict] = yaml.safe_load(stream)
if config["General"].get("seed") is not None:
    set_determinism(seed=config["General"]["seed"])

inference_suffix = "_"+config["General"]["inference"] if "inference" in config["General"] else ""
save_dir = config[Phase.TEST].get("save_dir") or config["Output"]["save_dir"]+"/test"
print(f"Saving test results to {save_dir}")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

device = torch.device(config["General"].get("device") or "cpu")
scaler = torch.cuda.amp.GradScaler(enabled=False)

with Live(group, console=Console(force_terminal=True), refresh_per_second=10):
    with DynamicDisplay(group, Spinner("bouncingBall", text="Loading test data...")):
        test_loader = get_dataset(config, Phase.TEST, num_workers=args.num_workers)
        post_transformations_test = get_post_transformation(config, Phase.TEST)
        test_mini_batch = next(iter(test_loader))
        input_key = [k for k in test_mini_batch.keys() if not k.endswith("_path")][0]
        test_mini_batch["image"] = test_mini_batch.pop(input_key)

    with DynamicDisplay(group, Spinner("bouncingBall", text="Initializing model...")):
        model = define_model(config, phase=Phase.TEST)
        model.initialize_model_and_optimizer(test_mini_batch, init_weights, config, args, scaler, phase=Phase.TEST)
        # Load the specified checkpoint if provided
        if args.model_path is not None:
            assert os.path.isfile(args.model_path), f"Model checkpoint {args.model_path} does not exist!"
            print(f"Loading model weights from {args.model_path}")
            state_dict = torch.load(args.model_path, map_location=device)
            model.netG_A.load_state_dict(state_dict)

    # Prepare directories for FID
    generated_dir = os.path.join(save_dir, "fid_generated")
    real_dir = os.path.join(save_dir, "fid_real")
    os.makedirs(generated_dir, exist_ok=True)
    os.makedirs(real_dir, exist_ok=True)

    model.eval()
    progress = Progress(*Progress.get_default_columns(), TimeElapsedColumn())
    progress.add_task("Testing:", total=min(len(test_loader), args.num_samples))
    with DynamicDisplay(group, progress):
        with torch.no_grad():
            num_sample=0
            for test_mini_batch in test_loader:
                if num_sample>=args.num_samples:
                    break
                num_sample+=1
                test_mini_batch["image"] = test_mini_batch.pop(input_key)
                outputs, _ = model.inference(test_mini_batch, post_transformations_test, device=device, phase=Phase.TEST)
                inference_mode = config["General"].get("inference") or "pred"
                image_name: str = test_mini_batch[f"{input_key}_path"][0].split("/")[-1]
                
                # Save generated image for FID
                gen_img_name = image_name.replace('.csv', '.png')
                plot_single_image(generated_dir, outputs["prediction"][0], gen_img_name)
                # Save also to main save_dir for user
                plot_single_image(save_dir, outputs["prediction"][0], inference_mode + "_" + image_name)
                
                # Save real image for FID if available
                if "real_B" in test_mini_batch:
                    real_img_name = image_name.replace('.csv', '.png')
                    real_img_path = os.path.join(real_dir, real_img_name)
                    save_image(test_mini_batch["real_B"][0], real_img_path)
                
                if config["Output"].get("save_comparisons"):
                    plot_sample(save_dir, test_mini_batch[input_key][0], outputs["prediction"][0], None, test_mini_batch[f"{input_key}_path"][0], suffix=f"{inference_mode}_{image_name}", full_size=True)
                
                progress.advance(task_id=0)

    # --- Compute FID ---
    print("Calculating FID between generated and real images...")
    try:
        fid_score = subprocess.check_output([
            "python", "-m", "pytorch_fid", generated_dir, real_dir
        ])
        print("FID:", fid_score.decode())
        with open(os.path.join(save_dir, "fid_score.txt"), "w") as f:
            f.write(fid_score.decode())
    except Exception as e:
        print("FID calculation failed:", e)