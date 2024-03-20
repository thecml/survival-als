from pathlib import Path
import numpy as np

# Directories
ROOT_DIR = Path(__file__).absolute().parent.parent
RAW_DATA_DIR = Path.joinpath(ROOT_DIR, "data/raw")
INTERIM_DATA_DIR = Path.joinpath(ROOT_DIR, 'data/interim')
PROCESSED_DATA_DIR = Path.joinpath(ROOT_DIR, 'data/processed')
MODELS_DIR = Path.joinpath(ROOT_DIR, 'models')
CONFIGS_DIR = Path.joinpath(ROOT_DIR, 'configs')

FILENAMES_RAW = {}