"""
reads config.yaml (shared settings) so every script uses the same values, and configs/<name>.yaml for one model's runs

import what you need, e.g. `from config import CLASSES, DATA_FOLDER, SEED`
"""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]  # repo root, two folders up from src/SeeNN/


def load_yaml(path):
    with open(ROOT / path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_config(name):
    """settings for one model's runs, from configs/<name>.yaml"""
    return load_yaml(Path("configs") / f"{name}.yaml")


CONFIG = load_yaml("config.yaml")

DATA_FOLDER = ROOT / CONFIG["paths"]["data"]
OUTPUT_FOLDER = ROOT / CONFIG["paths"]["standardized"]
RESULTS_FOLDER = ROOT / CONFIG["paths"]["results"]
CHECKPOINTS_FOLDER = ROOT / CONFIG["paths"]["checkpoints"]

SEED = CONFIG["seed"]
CLASSES = CONFIG["classes"]
VAL_FRACTION = CONFIG["val_fraction"]
CROP_HEIGHT = CONFIG["crop_height"]
IMAGE_SIZE = CONFIG["image_size"]
BATCH_SIZE = CONFIG["batch_size"]
