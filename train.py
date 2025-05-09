import argparse
import os
import re
import subprocess

from huggingface_hub import HfApi
from roboflow import Roboflow

from ruamel.yaml import YAML
from huggingface_hub import whoami
from loguru import logger
# ------------------------------------------------------------------------
# Download model
subprocess.run("venv/bin/pip install -e .[dev]", shell=True)
# -----------------------------------------------------

parser = argparse.ArgumentParser(description="Train SAM 2.1")
parser.add_argument("--roboflow_key", type=str, default="z6IkNNY2MUovMbOOeYaZ")
parser.add_argument("--roboflow_worksapce", type=str, default="brad-dwyer")
parser.add_argument("--roboflow_project", type=str, default="car-parts-pgo19")
parser.add_argument("--roboflow_version", type=int, default=6)
parser.add_argument("--roboflow_dataset", type=str, default="sam2")
parser.add_argument("--num_epochs", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument(
    "--yaml_path",
    type=str,
    default="./sam2/configs/train_example.yaml",
    help="Path to train.yaml",
)
parser.add_argument("--model", type=str, default="sam2.1_hiera_base_plus.pt")
parser.add_argument(
    "--hf_token", type=str, default="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"
)
parser.add_argument("--hf_model_id", type=str, default="sam2.1_hiera_base_plus.pt")
args = parser.parse_args()


# --- Update train.yaml with new values of num_epochs and batch_size ---
def update_train_yaml(yaml_path, num_epochs=None, batch_size=None, model=None):
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.indent(mapping=2, sequence=4, offset=2)
    yaml.explicit_start = False
    yaml.allow_unicode = True
    yaml.representer.add_representer(
        type(None),
        lambda dumper, data: dumper.represent_scalar("tag:yaml.org,2002:null", "null"),
    )
    # Read the YAML file
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.load(f)
    except Exception as e:
        print(f"Error reading YAML file: {e}")
        return

    # Update num_epochs
    if num_epochs is not None:
        if "scratch" in data:
            data["scratch"]["num_epochs"] = num_epochs
        else:
            print('Cannot find "scratch" section in yaml.')

    # Update batch_size
    if batch_size is not None:
        if "scratch" in data:
            data["scratch"]["train_batch_size"] = batch_size
        else:
            print('Cannot find "scratch" section in yaml.')

    # Update checkpoint_path
    if model is not None:
        if "trainer" in data and "checkpoint" in data["trainer"]:
            data["trainer"]["checkpoint"]["model_weight_initializer"]["state_dict"][
                "checkpoint_path"
            ] = f"./{model}"
        else:
            print('Cannot find "trainer.checkpoint" section in yaml.')

    # Write the YAML file
    try:
        with open("./sam2/configs/train.yaml", "w", encoding="utf-8") as f:
            yaml.dump(data, f)
    except Exception as e:
        print(f"Error writing YAML file: {e}")
        return


update_train_yaml(args.yaml_path, args.num_epochs, args.batch_size, args.model)
# 1. Download dataset
logger.info("Downloading dataset...")
rf = Roboflow(api_key=args.roboflow_key)
project = rf.workspace(args.roboflow_worksapce).project(args.roboflow_project)
version = project.version(args.roboflow_version)
dataset = version.download(args.roboflow_dataset, location="./data/")

# 2. Format dataset
for filename in os.listdir("./data/train"):
    # Replace all except last dot with underscore
    new_filename = filename.replace(".", "_", filename.count(".") - 1)
    if not re.search(r"_\d+\.\w+$", new_filename):
        # Add an int to the end of base name
        new_filename = new_filename.replace(".", "_1.")
    os.rename(
        os.path.join("./data/train", filename),
        os.path.join("./data/train", new_filename),
    )
logger.info("Dataset formatted successfully.")
# 3. Train
logger.info("Training model...")
command = (
    f"venv/bin/python training/train.py -c 'configs/train.yaml' --use-cluster 0 --num-gpus 1"
)
subprocess.run(command, shell=True)
logger.info("Model training completed successfully.")
# 4. Push logs and checkpoints to huggingface
api = HfApi()

username = whoami(token=args.hf_token)
repo_id = f"{username['name']}/{args.hf_model_id}"

# Create repo if not exists
try:
    api.create_repo(
        repo_id=args.hf_model_id, token=args.hf_token, exist_ok=True, private=False
    )
except Exception as e:
    print(f"Warning: Repo creation may have failed or already exists: {e}")

api.upload_file(
    path_or_fileobj="./sam2_logs/configs/train.yaml/checkpoints/checkpoint.pt",
    path_in_repo=args.model,
    repo_id=repo_id,
    token=args.hf_token,
    commit_message="Upload model checkpoint",
)

api.upload_folder(
    folder_path="./sam2_logs/",
    path_in_repo="runs",
    repo_id=repo_id,
    token=args.hf_token,
    commit_message="Upload TensorBoard traces",
)
